# A2C training entry point. Looks for the newest weights/sc2_a2c_*.weights.h5
# and resumes from it if one is there, otherwise starts from scratch. Each
# finished episode appends a row to reward_outputs/training_rewards.dat (the
# tab-separated format pgfplots reads) and losses go to TensorBoard at logs/a2c/.
import os
import sys
import time
import datetime

import numpy as np
import tensorflow as tf
from absl import app, flags

from sc2_env import SC2Env
from a2c_agent import A2CAgent

FLAGS = flags.FLAGS
np.set_printoptions(threshold=sys.maxsize)


def train_a2c():
    # PySC2 mini-game name (e.g. MoveToBeacon, CollectMineralShards, DefeatRoaches)
    map_name = "CollectMineralShards"
    num_envs = 4
    state_shape = (2, 84, 84)   # player_relative + selected
    num_actions = 524
    step_mul_train = 8

    num_episodes = 1000
    update_frequency = 20       # K = 20 transitions per env between gradient updates
    save_frequency = 100        # checkpoint every N episodes

    learning_rate = 7e-4
    gamma = 0.99
    gae_lambda = 0.95     # GAE: lambda=1 -> Monte-Carlo, lambda=0 -> 1-step TD
    entropy_coef = 5e-2     # entropy bonus on the main action distribution
    value_loss_coef = 0.5

    tensorboard_writer = tf.summary.create_file_writer("logs/a2c")

    agent = A2CAgent(
        state_shape=state_shape,
        num_actions=num_actions,
        screen_size=84,
        learning_rate=learning_rate,
        gamma=gamma,
        entropy_coef=entropy_coef,
        value_loss_coef=value_loss_coef,
        num_envs=num_envs,
        gae_lambda=gae_lambda,
    )

    model_dir = "weights"
    logs_dir = "logs"
    reward_outputs_dir = "reward_outputs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(reward_outputs_dir, exist_ok=True)

    reward_log_path = os.path.join(reward_outputs_dir, "training_rewards.dat")

    # auto-resume from latest checkpoint
    latest_model, latest_episode = None, -1
    for filename in os.listdir(model_dir):
        if filename.startswith("sc2_a2c_") and filename.endswith(".weights.h5"):
            try:
                ep = int(filename.split("_")[-1].split(".")[0])
                if ep > latest_episode:
                    latest_episode = ep
                    latest_model = filename
            except ValueError:
                continue

    if latest_model:
        agent.load(os.path.join(model_dir, latest_model))
        start_episode = latest_episode + 1
        print(f"Resuming training from episode {start_episode}")
    else:
        start_episode = 0
        print("Starting training from scratch.")

    # write a one-time header + a per-session config block to the reward log
    reward_log_existed = os.path.exists(reward_log_path)
    session_start_iso = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(reward_log_path, "a") as _f:
        if not reward_log_existed:
            _f.write("# columns: episode env steps reward avg_loss\n")
            _f.write("episode\tenv\tsteps\treward\tavg_loss\n")
        marker = "session start" if not reward_log_existed else "session resume"
        _f.write(f"# --- {marker}: {session_start_iso} ---\n")
        _f.write(f"# map={map_name}   step_mul_train={step_mul_train}   num_envs={num_envs}\n")
        _f.write(f"# state_shape={state_shape}   num_actions={num_actions}\n")
        _f.write(f"# learning_rate={learning_rate}   gamma={gamma}   "
                 f"gae_lambda={gae_lambda}   "
                 f"entropy_coef={entropy_coef}   value_loss_coef={value_loss_coef}\n")
        _f.write(f"# update_frequency={update_frequency}   save_frequency={save_frequency}\n")
        _f.write(f"# start_episode={start_episode}   num_episodes={num_episodes}   "
                 f"target_episode={start_episode + num_episodes - 1}\n")

    print(f"Launching {num_envs} parallel environments on map '{map_name}'...")
    envs = [SC2Env(map_name=map_name, step_mul=step_mul_train) for _ in range(num_envs)]

    env_states = [env.reset() for env in envs]
    env_episode_rewards = [0.0] * num_envs
    env_episode_steps = [0] * num_envs

    # transition buffer, shared across all envs
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    action_args = []

    total_steps = 0
    episode_count = start_episode
    episode_losses = []
    start_time = time.time()

    target_ep = start_episode + num_episodes
    print(f"Training starts at episode {start_episode}, target {target_ep} episodes.")

    while episode_count < start_episode + num_episodes:
        for i, env in enumerate(envs):
            available_actions = env.get_available_actions()
            action, args = agent.select_action(
                env_states[i], available_actions,
                multi_select_size=env.get_multi_select_size(),
                training=True)

            next_state, reward, done, env_args = env.step(action, args)

            states.append(env_states[i])
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            action_args.append(args)

            env_states[i] = next_state
            env_episode_rewards[i] += reward
            env_episode_steps[i] += 1
            total_steps += 1

            if done:
                ep_reward = env_episode_rewards[i]
                ep_steps = env_episode_steps[i]
                avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0

                print(f"Episode: {episode_count}, Env: {i}, Steps: {ep_steps}, "
                      f"Reward: {ep_reward:.2f}, Loss: {avg_loss:.6f}")

                # append the row for this finished episode
                with open(reward_log_path, "a") as _f:
                    _f.write(
                        f"{episode_count}\t{i}\t{ep_steps}\t"
                        f"{ep_reward:.4f}\t{avg_loss:.6f}\n"
                    )

                # checkpoint every save_frequency episodes, plus the very last one
                if (episode_count % save_frequency == 0
                        or episode_count == start_episode + num_episodes - 1):
                    save_path = os.path.join(model_dir, f"sc2_a2c_{episode_count}.weights.h5")
                    agent.save(save_path)
                    print(f"Model saved to {save_path}")

                with tensorboard_writer.as_default():
                    tf.summary.scalar('Rewards', ep_reward, step=episode_count)

                episode_count += 1
                episode_losses = []

                env_states[i] = env.reset()
                env_episode_rewards[i] = 0.0
                env_episode_steps[i] = 0

                if episode_count >= start_episode + num_episodes:
                    break

        # one gradient update every update_frequency*num_envs transitions in the buffer
        if len(states) >= update_frequency * num_envs:
            loss_dict = agent.train(
                np.array(states),
                np.array(actions),
                np.array(rewards),
                np.array(next_states),
                np.array(dones),
                action_args,
            )

            episode_losses.append(loss_dict['total_loss'])
            with tensorboard_writer.as_default():
                tf.summary.scalar('Main Policy Loss',
                                  np.mean(loss_dict['main_policy_loss']), step=total_steps)
                tf.summary.scalar('Main Value Loss',
                                  np.mean(loss_dict['main_value_loss']), step=total_steps)
                tf.summary.scalar('Screen XY Policy Loss',
                                  np.mean(loss_dict['screen_xy_policy_loss']), step=total_steps)
                tf.summary.scalar('Screen XY Value Loss',
                                  np.mean(loss_dict['screen_xy_value_loss']), step=total_steps)

            # wipe the buffer so the next batch starts fresh
            states, actions, rewards = [], [], []
            next_states, dones, action_args = [], [], []

    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time/60:.2f} minutes")

    for env in envs:
        env.close()


def main(argv):
    train_a2c()


if __name__ == "__main__":
    app.run(main)
