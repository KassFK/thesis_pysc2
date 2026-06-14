# Branching DQN: six independent Q-heads (one for the action id, four for the
# screen coords, one for the queue flag) sharing a single FullyConv backbone.
import tensorflow as tf
import numpy as np
import random
from collections import deque

from dqn_model import DQNNetwork


# simple FIFO replay buffer; deque with a max length drops the oldest transition
# automatically when we hit capacity
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# DQN with the three classic ingredients: epsilon-greedy exploration during training,
# a replay buffer to break sample correlation, and a target network for stable
# Bellman targets
class DQNAgent:

    def __init__(self, state_shape, num_actions, screen_size,
                 learning_rate=1e-4, gamma=0.99,
                 buffer_size=10_000, batch_size=32, target_sync=1000,
                 eps_start=1.0, eps_end=0.05, eps_decay_steps=500_000):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.screen_size = screen_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps

        self.network = DQNNetwork(num_actions, screen_size)
        self.target_network = DQNNetwork(num_actions, screen_size)

        # build both with a dummy forward pass so set_weights works
        dummy = tf.zeros(shape=(1,) + state_shape)
        self.network(dummy)
        self.target_network(dummy)
        self.target_network.set_weights(self.network.get_weights())

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.huber = tf.keras.losses.Huber()
        self.buffer = ReplayBuffer(buffer_size)

        # no_op and move_camera don't help on the mini-games we train on
        self.block_noop = True
        self.block_move_camera = True

        self.train_steps = 0
        self.env_steps = 0

        print("DQN Agent initialised.")
        print(self.network.summary())

    # how much we explore right now: starts at eps_start, slowly drops to
    # eps_end over the first eps_decay_steps environment steps
    def epsilon(self):
        frac = min(1.0, self.env_steps / max(1, self.eps_decay_steps))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def _build_main_mask(self, available_actions):
        mask = np.full(self.num_actions, -1e9, dtype=np.float32)
        mask[list(available_actions)] = 0.0
        if self.block_noop:
            mask[0] = -1e9
        if self.block_move_camera:
            mask[1] = -1e9
        return mask

    # pick an action: epsilon-greedy during training, top-N random pick at eval
    def select_action(self, state, available_actions, training=False):
        if len(available_actions) < 1:
            return 0, {}

        eps = self.epsilon() if training else 0.0
        if training:
            self.env_steps += 1

        outputs = self.network(np.expand_dims(state, axis=0))
        q_main = outputs['q_main'][0].numpy()
        mask = self._build_main_mask(available_actions)
        masked_q = q_main + mask
        valid = np.where(mask == 0.0)[0]
        if len(valid) == 0:
            valid = np.array(list(available_actions), dtype=np.int64)

        # main action
        if training:
            # epsilon-greedy: random valid action with prob epsilon, else argmax
            if np.random.random() < eps:
                action = np.int64(np.random.choice(valid))
            else:
                action = np.int64(np.argmax(masked_q))
        else:
            # eval: pick uniformly from top-5 valid Q-values (softer than argmax)
            n_action = 5
            n_eff = min(n_action, len(valid))
            valid_q = q_main[valid]
            top_n_in_valid = np.argpartition(valid_q, -n_eff)[-n_eff:]
            top_n_action_ids = valid[top_n_in_valid]
            action = np.int64(np.random.choice(top_n_action_ids))

        # small helper for the other heads: epsilon-greedy during training, and
        # at eval either argmax or a random pick from the top-N Q-values
        def pick(qvals, dim, n_top_eval=None):
            if training:
                if np.random.random() < eps:
                    return np.int64(np.random.randint(dim))
                return np.int64(np.argmax(qvals))
            if n_top_eval is None or n_top_eval >= dim:
                return np.int64(np.argmax(qvals))
            n_eff = min(n_top_eval, dim)
            top_n = np.argpartition(qvals, -n_eff)[-n_eff:]
            return np.int64(np.random.choice(top_n))

        # spatial args: top-2 over Q at eval for a small extra exploration
        x1 = pick(outputs['q_screen_x'][0].numpy(), self.screen_size, n_top_eval=2)
        y1 = pick(outputs['q_screen_y'][0].numpy(), self.screen_size, n_top_eval=2)
        x2 = pick(outputs['q_screen2_x'][0].numpy(), self.screen_size, n_top_eval=2)
        y2 = pick(outputs['q_screen2_y'][0].numpy(), self.screen_size, n_top_eval=2)
        q_arg = pick(outputs['q_queue'][0].numpy(), 2)   # only 2 options, keep greedy

        args_dict = {
            'queued': [q_arg],
            'screen': [x1, y1],
            'screen2': [x2, y2],
            'action_prob': 1.0,
            'value': float(masked_q[action]),
        }
        return action, args_dict

    # push one transition into the replay buffer.
    # we also pre-compute the next-state action mask here so train_step
    # doesn't need a reference to the env
    def store(self, state, action, args_dict, reward, next_state, done,
              next_available_actions):
        next_mask = np.zeros(self.num_actions, dtype=bool)
        next_mask[list(next_available_actions)] = True
        if self.block_noop:
            next_mask[0] = False
        if self.block_move_camera:
            next_mask[1] = False

        self.buffer.push((
            state.astype(np.float32),
            int(action),
            int(args_dict['screen'][0]),
            int(args_dict['screen'][1]),
            int(args_dict['screen2'][0]),
            int(args_dict['screen2'][1]),
            int(args_dict['queued'][0]),
            float(reward),
            next_state.astype(np.float32),
            float(done),
            next_mask,
        ))

    # sample a minibatch from the buffer and run one Bellman update on it
    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)
        (states, actions, x1s, y1s, x2s, y2s, qs,
         rewards, next_states, dones, next_masks) = zip(*batch)

        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        x1s = np.array(x1s, dtype=np.int32)
        y1s = np.array(y1s, dtype=np.int32)
        x2s = np.array(x2s, dtype=np.int32)
        y2s = np.array(y2s, dtype=np.int32)
        qs = np.array(qs, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        next_masks = np.array(next_masks, dtype=bool)

        # ask the (frozen) target network for the next-state Q-values
        next_outputs = self.target_network(next_states)
        next_q_main = next_outputs['q_main'].numpy()
        # ignore Q-values for actions the env won't let us pick next step
        next_q_main_masked = np.where(next_masks, next_q_main, -1e9)
        # in the rare case a row has no valid next action (can happen on
        # DefeatRoaches when all units die at once), fall back to 0
        any_valid = np.any(next_masks, axis=1)
        next_q_main_max = np.where(
            any_valid, np.max(next_q_main_masked, axis=1), 0.0)

        next_q_x_max = np.max(next_outputs['q_screen_x'].numpy(), axis=1)
        next_q_y_max = np.max(next_outputs['q_screen_y'].numpy(), axis=1)
        next_q_x2_max = np.max(next_outputs['q_screen2_x'].numpy(), axis=1)
        next_q_y2_max = np.max(next_outputs['q_screen2_y'].numpy(), axis=1)
        next_q_q_max = np.max(next_outputs['q_queue'].numpy(), axis=1)

        not_done = 1.0 - dones
        target_main = rewards + self.gamma * not_done * next_q_main_max
        target_x = rewards + self.gamma * not_done * next_q_x_max
        target_y = rewards + self.gamma * not_done * next_q_y_max
        target_x2 = rewards + self.gamma * not_done * next_q_x2_max
        target_y2 = rewards + self.gamma * not_done * next_q_y2_max
        target_q = rewards + self.gamma * not_done * next_q_q_max

        with tf.GradientTape() as tape:
            outputs = self.network(states)

            def gather(qmat, idx):
                rng = tf.range(tf.shape(qmat)[0], dtype=tf.int32)
                return tf.gather_nd(
                    qmat, tf.stack([rng, tf.cast(idx, tf.int32)], axis=1))

            q_main_chosen = gather(outputs['q_main'], actions)
            q_x_chosen = gather(outputs['q_screen_x'], x1s)
            q_y_chosen = gather(outputs['q_screen_y'], y1s)
            q_x2_chosen = gather(outputs['q_screen2_x'], x2s)
            q_y2_chosen = gather(outputs['q_screen2_y'], y2s)
            q_q_chosen = gather(outputs['q_queue'], qs)

            loss_main = self.huber(target_main, q_main_chosen)
            loss_x = self.huber(target_x, q_x_chosen)
            loss_y = self.huber(target_y, q_y_chosen)
            loss_x2 = self.huber(target_x2, q_x2_chosen)
            loss_y2 = self.huber(target_y2, q_y2_chosen)
            loss_q = self.huber(target_q, q_q_chosen)
            total_loss = loss_main + loss_x + loss_y + loss_x2 + loss_y2 + loss_q

        # apply the Bellman update; global-norm clip at 40 stops the rare
        # outlier from blowing up the Q-values
        grads = tape.gradient(total_loss, self.network.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 40.0)
        self.optimizer.apply_gradients(
            zip(grads, self.network.trainable_variables))

        # copy the live network's weights into the target every target_sync steps
        self.train_steps += 1
        if self.train_steps % self.target_sync == 0:
            self.target_network.set_weights(self.network.get_weights())

        return {
            'total_loss': float(total_loss.numpy()),
            'loss_main': float(loss_main.numpy()),
            'loss_screen_x': float(loss_x.numpy()),
            'loss_screen_y': float(loss_y.numpy()),
            'loss_screen2_x': float(loss_x2.numpy()),
            'loss_screen2_y': float(loss_y2.numpy()),
            'loss_queue': float(loss_q.numpy()),
            'epsilon': self.epsilon(),
        }

    def save(self, path):
        self.network.save_weights(path)

    def load(self, path):
        self.network.load_weights(path)
        self.target_network.set_weights(self.network.get_weights())
