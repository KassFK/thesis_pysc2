# DQN training entry point. The script picks up the highest-numbered
# sc2_dqn_*.weights.h5 in weights/ if one exists, and otherwise begins
# from scratch. Per-episode rewards are appended to
# reward_outputs/training_rewards.dat (the format pgfplots reads) and
# loss/epsilon scalars go to TensorBoard at logs/dqn/.
import glob
import os
import re
import sys
import time
import datetime

import numpy as np
import tensorflow as tf
from absl import app, flags

from sc2_env import SC2Env
from dqn_agent import DQNAgent

FLAGS = flags.FLAGS
np.set_printoptions(threshold=sys.maxsize)


# pattern that matches sc2_dqn_<episode>.weights.h5 inside weights/
_CKPT_RE = re.compile(r"sc2_dqn_(\d+)\.weights\.h5$")


def _find_latest_checkpoint(model_dir):
    """Return (path, episode) of the highest-numbered checkpoint, or (None, -1)."""
    candidates = []
    for path in glob.glob(os.path.join(model_dir, "sc2_dqn_*.weights.h5")):
        m = _CKPT_RE.search(path)
        if m is not None:
            candidates.append((int(m.group(1)), path))
    if not candidates:
        return None, -1
    ep, path = max(candidates, key=lambda t: t[0])
    return path, ep


def _dump_session_block(fh, params, header_needed):
    """Write the per-session config block to the reward log."""
    if header_needed:
        fh.write("# columns: episode env steps reward avg_loss epsilon\n")
        fh.write("episode\tenv\tsteps\treward\tavg_loss\tepsilon\n")
    marker = "session start" if header_needed else "session resume"
    fh.write(f"# === {marker} :: {params['session_iso']} ===\n")
    for key in (
        "map", "step_mul_train", "num_envs",
        "state_shape", "num_actions",
        "learning_rate", "gamma",
        "buffer_size", "batch_size",
        "target_sync", "train_every", "save_frequency",
        "eps_start", "eps_end", "eps_decay_steps",
        "start_episode", "num_episodes", "target_episode",
    ):
        fh.write(f"## {key}: {params[key]}\n")


def train_dqn():
    # which mini-game to train on. pick from MoveToBeacon / CollectMineralShards
    # / DefeatRoaches / BuildMarines (any PySC2 map id will work in principle)
    map_name = "CollectMineralShards"
    num_envs = 4
    state_shape = (2, 84, 84)   # player_relative + selected
    num_actions = 524
    step_mul_train = 8

    num_episodes = 1000
    train_every = 4             # gradient step every N total env-steps
    save_frequency = 100        # checkpoint every N episodes

    learning_rate = 1e-4
    gamma = 0.99
    buffer_size = 10_000
    batch_size = 32
    target_sync = 1000     # target-network sync period (in train steps)
    eps_start = 1.0
    eps_end = 0.05
    eps_decay_steps = 500_000  # linear epsilon anneal over this many env-steps

    tensorboard_writer = tf.summary.create_file_writer("logs/dqn")

    agent = DQNAgent(
        state_shape=state_shape,
        num_actions=num_actions,
        screen_size=84,
        learning_rate=learning_rate,
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        target_sync=target_sync,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay_steps=eps_decay_steps,
    )

    model_dir = "weights"
    logs_dir = "logs"
    reward_outputs_dir = "reward_outputs"
    for d in (model_dir, logs_dir, reward_outputs_dir):
        os.makedirs(d, exist_ok=True)

    reward_log_path = os.path.join(reward_outputs_dir, "training_rewards.dat")

    # scan weights/ for the highest-numbered sc2_dqn_*.weights.h5 and resume there
    latest_path, latest_episode = _find_latest_checkpoint(model_dir)
    if latest_path is not None:
        agent.load(latest_path)
        start_episode = latest_episode + 1
        print(f"Resuming DQN training from episode {start_episode}.")
    else:
        start_episode = 0
        print("No checkpoint found. Training from scratch.")

    # log header on first write, then a per-session config dump
    header_needed = not os.path.exists(reward_log_path)
    session_iso = datetime.datetime.now().isoformat(timespec='seconds')
    session_params = {
        "session_iso": session_iso,
        "map": map_name, "step_mul_train": step_mul_train, "num_envs": num_envs,
        "state_shape": state_shape, "num_actions": num_actions,
        "learning_rate": learning_rate, "gamma": gamma,
        "buffer_size": buffer_size, "batch_size": batch_size,
        "target_sync": target_sync, "train_every": train_every,
        "save_frequency": save_frequency,
        "eps_start": eps_start, "eps_end": eps_end,
        "eps_decay_steps": eps_decay_steps,
        "start_episode": start_episode, "num_episodes": num_episodes,
        "target_episode": start_episode + num_episodes - 1,
    }
    with open(reward_log_path, "a") as fh:
        _dump_session_block(fh, session_params, header_needed)

    print(f"Spinning up {num_envs} SC2Env workers on {map_name} ...")
    envs = []
    for _ in range(num_envs):
        envs.append(SC2Env(map_name=map_name, step_mul=step_mul_train))

    env_states = list(map(lambda e: e.reset(), envs))
    env_episode_rewards = [0.0 for _ in range(num_envs)]
    env_episode_steps = [0 for _ in range(num_envs)]

    total_steps = 0
    episodes_done = start_episode
    episode_losses = []
    start_time = time.time()

    target_ep = start_episode + num_episodes
    print(f"Training: ep {start_episode} -> {target_ep} ({num_episodes} more).")

    while episodes_done < target_ep:
        for env_idx, env in enumerate(envs):
            available = env.get_available_actions()
            action, args = agent.select_action(env_states[env_idx], available, training=True)

            next_state, reward, done, _ = env.step(action, args)
            next_available = env.get_available_actions()

            # push transition into the replay buffer
            agent.store(env_states[env_idx], action, args, reward,
                        next_state, done, next_available)

            env_states[env_idx] = next_state
            env_episode_rewards[env_idx] += reward
            env_episode_steps[env_idx] += 1
            total_steps += 1

            # one gradient step every train_every env-steps
            if total_steps % train_every == 0:
                loss_dict = agent.train_step()
                if loss_dict is not None:
                    episode_losses.append(loss_dict['total_loss'])
                    _log_train_scalars(tensorboard_writer, loss_dict, total_steps)

            if done:
                ep_reward = env_episode_rewards[env_idx]
                ep_steps = env_episode_steps[env_idx]
                avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
                ep_eps = agent.epsilon()

                print(
                    f"[ep {episodes_done} env {env_idx}] "
                    f"steps={ep_steps} rew={ep_reward:.2f} "
                    f"loss={avg_loss:.6f} eps={ep_eps:.3f}"
                )

                # append per-episode row to the reward log
                with open(reward_log_path, "a") as fh:
                    fh.write(
                        f"{episodes_done}\t{env_idx}\t{ep_steps}\t"
                        f"{ep_reward:.4f}\t{avg_loss:.6f}\t{ep_eps:.4f}\n"
                    )

                # save model periodically and on the final episode
                is_periodic = (episodes_done % save_frequency == 0)
                is_final = (episodes_done == target_ep - 1)
                if is_periodic or is_final:
                    save_path = os.path.join(
                        model_dir, f"sc2_dqn_{episodes_done}.weights.h5")
                    agent.save(save_path)
                    print(f"  -> checkpoint written: {save_path}")

                with tensorboard_writer.as_default():
                    tf.summary.scalar('Rewards', ep_reward, step=episodes_done)

                episodes_done += 1
                episode_losses = []

                env_states[env_idx] = env.reset()
                env_episode_rewards[env_idx] = 0.0
                env_episode_steps[env_idx] = 0

                if episodes_done >= target_ep:
                    break

    elapsed_time = time.time() - start_time
    print(f"Done. Total wall time: {elapsed_time/60:.2f} min.")

    for env in envs:
        env.close()


def _log_train_scalars(writer, loss_dict, step):
    """Emit one TB scalar per loss component for the latest gradient step."""
    name_map = {
        'Total Loss':    'total_loss',
        'Loss Main':     'loss_main',
        'Loss Screen X': 'loss_screen_x',
        'Loss Screen Y': 'loss_screen_y',
        'Loss Screen2 X':'loss_screen2_x',
        'Loss Screen2 Y':'loss_screen2_y',
        'Loss Queue':    'loss_queue',
        'Epsilon':       'epsilon',
    }
    with writer.as_default():
        for tb_name, key in name_map.items():
            tf.summary.scalar(tb_name, loss_dict[key], step=step)


def main(argv):
    train_dqn()


if __name__ == "__main__":
    app.run(main)
