# Synchronous A2C with a multi-head policy: one head picks the PySC2 function
# id, the others fill in the spatial coords, the queue flag, and the unit index.
import tensorflow as tf
import numpy as np
from pysc2.lib import actions as sc2_actions
from a2c_model import A2CNetwork


# synchronous A2C with a multi-head policy and GAE advantages
class A2CAgent:

    def __init__(self, state_shape, num_actions, screen_size, learning_rate,
                 gamma, entropy_coef, value_loss_coef,
                 num_envs=4, gae_lambda=0.95):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.screen_size = screen_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.num_envs = num_envs        # GAE uses this to reshape the buffer
        self.gae_lambda = gae_lambda

        self.network = A2CNetwork(num_actions, screen_size)
        dummy_state = tf.zeros(shape=(1,) + state_shape)
        self.network(dummy_state)       # warmup so all variables exist before training

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # no_op and move_camera don't help on the mini-games we train on,
        # so we just mask them out of the action distribution
        self.block_noop = True
        self.block_move_camera = True

        print("A2C Agent initialised.")
        print(self.network.summary())

    # pick an action id and the argument values that go with it
    def select_action(self, state, available_actions, multi_select_size=0, training=False):
        if len(available_actions) < 1:
            return 0, {}

        state = np.expand_dims(state, axis=0)
        outputs = self.network(state)

        # set unavailable actions to a very negative logit; softmax then ~ignores them
        action_logits = outputs['action_logits'][0].numpy()
        mask = np.full_like(action_logits, -1e9)
        mask[available_actions] = 0.0
        if self.block_noop:
            mask[0] = -1e9
        if self.block_move_camera:
            mask[1] = -1e9
        masked_logits = action_logits + mask
        action_probs = tf.nn.softmax(masked_logits).numpy()

        # main action: keep the top-15 most likely actions, then sample one
        # proportionally to its probability inside that pool
        n = 15
        available_probs = action_probs[available_actions]
        n_eff = min(n, len(available_actions))
        top_n_in_available = np.argpartition(available_probs, -n_eff)[-n_eff:]
        top_n_action_ids = np.array(available_actions)[top_n_in_available]
        top_n_probs = available_probs[top_n_in_available]
        top_n_probs = top_n_probs / top_n_probs.sum()
        action = np.random.choice(top_n_action_ids, p=top_n_probs)

        # queue flag (0 = do it now, 1 = put it in the queue)
        queue_probs = outputs['queue_probs'][0].numpy()
        queue_arg = np.int64(np.random.choice([0, 1], p=queue_probs))

        # spatial (x, y): same idea - top-50 pixels, then proportional sampling
        screen_xy_probs = outputs['screen_xy_probs'][0].numpy()
        flat_xy = screen_xy_probs.flatten()
        n_xy = 50
        top_n_indices = np.argpartition(flat_xy, -n_xy)[-n_xy:]
        top_n_xy_probs = flat_xy[top_n_indices]
        top_n_xy_probs = top_n_xy_probs / top_n_xy_probs.sum()
        flat_idx = np.random.choice(top_n_indices, p=top_n_xy_probs)
        y1, x1 = np.unravel_index(flat_idx, (self.screen_size, self.screen_size))
        x1, y1 = np.int64(x1), np.int64(y1)

        # screen2 corner (only select_rect uses this): full proportional sampling
        screen2_x_probs = outputs['screen2_x_probs'][0].numpy()
        screen2_y_probs = outputs['screen2_y_probs'][0].numpy()
        x2 = np.int64(np.random.choice(self.screen_size, p=screen2_x_probs))
        y2 = np.int64(np.random.choice(self.screen_size, p=screen2_y_probs))

        # select_unit_id: only the first multi_select_size indices are real units,
        # so we mask the rest and then proportional-sample over the top-min(5, valid)
        suid_logits = outputs['select_unit_id_logits'][0].numpy()
        head_dim = suid_logits.shape[0]
        valid_count = min(int(multi_select_size), head_dim)
        if valid_count > 0:
            suid_mask = np.full_like(suid_logits, -1e9)
            suid_mask[:valid_count] = 0.0
            suid_probs = tf.nn.softmax(suid_logits + suid_mask).numpy()
            n_suid = min(5, valid_count)
            top_n = np.argpartition(suid_probs[:valid_count], -n_suid)[-n_suid:]
            top_n_suid_probs = suid_probs[top_n]
            top_n_suid_probs = top_n_suid_probs / top_n_suid_probs.sum()
            select_unit_id = np.int64(np.random.choice(top_n, p=top_n_suid_probs))
        else:
            select_unit_id = np.int64(0)

        args_dict = {
            'queued': [queue_arg],
            'screen': [x1, y1],
            'screen2': [x2, y2],
            'select_unit_id': [select_unit_id],
            'action_prob': action_probs[action],
            'value': outputs['value'][0][0].numpy(),
        }
        return action, args_dict

    # one A2C update over a K-step rollout collected from num_envs parallel envs
    def train(self, states, actions, rewards, next_states, dones, action_args):
        if len(states) == 0:
            return {
                'total_loss': 0, 'policy_loss': 0, 'value_loss': 0,
                'entropy': 0, 'main_policy_loss': 0, 'main_value_loss': 0,
                'screen_xy_policy_loss': 0, 'screen_xy_value_loss': 0,
                'coord_policy_loss': 0, 'coord_value_loss': 0,
            }

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        batch_size = tf.shape(states)[0]

        with tf.GradientTape() as tape:
            outputs = self.network(states)
            next_outputs = self.network(next_states)

            # 1-step TD deltas (the building blocks for GAE below)
            values = tf.squeeze(outputs['value'], axis=1)
            next_values = tf.squeeze(next_outputs['value'], axis=1)
            deltas = rewards + self.gamma * next_values * (1 - dones) - values

            # GAE: walk back through each env's rollout and accumulate the deltas;
            # when K=1 we fall back to plain 1-step TD
            K_int = len(action_args) // self.num_envs
            if K_int > 1:
                deltas_2d = tf.reshape(deltas, (K_int, self.num_envs))
                dones_2d = tf.reshape(dones, (K_int, self.num_envs))
                gae_acc = tf.zeros((self.num_envs,), dtype=tf.float32)
                gae_steps = [None] * K_int
                for t in range(K_int - 1, -1, -1):
                    gae_acc = (deltas_2d[t]
                               + self.gamma * self.gae_lambda
                               * (1.0 - dones_2d[t]) * gae_acc)
                    gae_steps[t] = gae_acc
                td_advantages = tf.reshape(tf.stack(gae_steps, axis=0), (-1,))
            else:
                td_advantages = deltas

            # normalise advantages - used for the policy loss only,
            # the value loss below stays with the raw advantage
            adv_mean = tf.reduce_mean(td_advantages)
            adv_std = tf.math.reduce_std(td_advantages)
            td_advantages_norm = (td_advantages - adv_mean) / (adv_std + 1e-8)

            # main action policy loss (REINFORCE-style with the advantage baseline)
            action_probs = outputs['action_probs']
            action_indices = tf.stack(
                [tf.range(batch_size, dtype=tf.int32), actions], axis=1)
            chosen_action_probs = tf.gather_nd(action_probs, action_indices)
            main_log_probs = tf.math.log(chosen_action_probs + 1e-10)
            main_policy_loss = tf.reduce_mean(
                -main_log_probs * tf.stop_gradient(td_advantages_norm))

            # small entropy bonus so the policy keeps exploring
            main_entropy = -tf.reduce_mean(tf.reduce_sum(
                action_probs * tf.math.log(action_probs + 1e-10), axis=1))

            # value loss - we share one critic across all heads
            value_loss = tf.reduce_mean(tf.square(td_advantages))

            # per-head losses: only fire on transitions where the chosen action
            # actually consumed that argument (otherwise the head's output got
            # thrown away by the env, so we shouldn't push gradients through it)
            screen_xy_probs = outputs['screen_xy_probs']
            screen_xy_policy_losses = []
            screen2_x_policy_losses = []
            screen2_y_policy_losses = []
            select_unit_id_policy_losses = []
            coord_entropies = []

            for i, args in enumerate(action_args):
                if not args:
                    continue

                # which args the chosen action actually needs
                try:
                    arg_names = {a.name for a in sc2_actions.FUNCTIONS[int(actions[i])].args}
                except (IndexError, KeyError, TypeError):
                    arg_names = set()

                # primary spatial head
                if ('screen' in arg_names and 'screen' in args
                        and args['screen'] is not None and len(args['screen']) >= 2):
                    x_val = tf.convert_to_tensor([args['screen'][0]], dtype=tf.int32)
                    y_val = tf.convert_to_tensor([args['screen'][1]], dtype=tf.int32)
                    screen_xy = screen_xy_probs[i, :, :]
                    log_xy = tf.math.log(screen_xy[y_val[0], x_val[0]] + 1e-10)
                    screen_xy_policy_losses.append(
                        -log_xy * tf.stop_gradient(td_advantages_norm[i]))

                # screen2 x
                if ('screen2' in arg_names and 'screen2' in args
                        and args['screen2'] is not None and len(args['screen2']) >= 1):
                    x2_val = tf.convert_to_tensor([args['screen2'][0]], dtype=tf.int32)
                    s2x_probs = outputs['screen2_x_probs'][i]
                    x2_one_hot = tf.one_hot(x2_val, self.screen_size)
                    x2_log_prob = tf.reduce_sum(
                        tf.math.log(s2x_probs + 1e-10) * x2_one_hot)
                    screen2_x_policy_losses.append(
                        -x2_log_prob * tf.stop_gradient(td_advantages_norm[i]))
                    coord_entropies.append(-tf.reduce_sum(
                        s2x_probs * tf.math.log(s2x_probs + 1e-10)))

                # screen2 y
                if ('screen2' in arg_names and 'screen2' in args
                        and args['screen2'] is not None and len(args['screen2']) >= 2):
                    y2_val = tf.convert_to_tensor([args['screen2'][1]], dtype=tf.int32)
                    s2y_probs = outputs['screen2_y_probs'][i]
                    y2_one_hot = tf.one_hot(y2_val, self.screen_size)
                    y2_log_prob = tf.reduce_sum(
                        tf.math.log(s2y_probs + 1e-10) * y2_one_hot)
                    screen2_y_policy_losses.append(
                        -y2_log_prob * tf.stop_gradient(td_advantages_norm[i]))
                    coord_entropies.append(-tf.reduce_sum(
                        s2y_probs * tf.math.log(s2y_probs + 1e-10)))

                # select_unit_id
                if ('select_unit_id' in arg_names and 'select_unit_id' in args
                        and args['select_unit_id'] is not None
                        and len(args['select_unit_id']) >= 1):
                    suid_val = tf.convert_to_tensor(
                        [args['select_unit_id'][0]], dtype=tf.int32)
                    suid_probs = outputs['select_unit_id_probs'][i]
                    suid_one_hot = tf.one_hot(suid_val, 16)
                    suid_log_prob = tf.reduce_sum(
                        tf.math.log(suid_probs + 1e-10) * suid_one_hot)
                    select_unit_id_policy_losses.append(
                        -suid_log_prob * tf.stop_gradient(td_advantages_norm[i]))
                    coord_entropies.append(-tf.reduce_sum(
                        suid_probs * tf.math.log(suid_probs + 1e-10)))

            def _mean_or_zero(lst):
                if len(lst) > 0:
                    return tf.reduce_mean(tf.stack(lst))
                return tf.constant(0.0)

            screen_xy_policy_loss = _mean_or_zero(screen_xy_policy_losses)
            screen2_x_policy_loss = _mean_or_zero(screen2_x_policy_losses)
            screen2_y_policy_loss = _mean_or_zero(screen2_y_policy_losses)
            select_unit_id_policy_loss = _mean_or_zero(select_unit_id_policy_losses)
            coord_entropy = _mean_or_zero(coord_entropies)
            total_entropy = main_entropy + coord_entropy

            # add up every head's policy loss + value loss - entropy bonus
            total_loss = (main_policy_loss
                          + screen_xy_policy_loss
                          + screen2_x_policy_loss
                          + screen2_y_policy_loss
                          + select_unit_id_policy_loss
                          + self.value_loss_coef * value_loss
                          - self.entropy_coef * total_entropy)

        # one gradient step; clip the global gradient norm so updates stay sane
        grads = tape.gradient(total_loss, self.network.trainable_variables)
        grads_and_vars = [(g, v)
                          for g, v in zip(grads, self.network.trainable_variables)
                          if g is not None]
        if len(grads_and_vars) > 0:
            gs = [g for g, _ in grads_and_vars]
            vs = [v for _, v in grads_and_vars]
            gs, _ = tf.clip_by_global_norm(gs, 40.0)
            self.optimizer.apply_gradients(zip(gs, vs))

        coord_policy_loss = screen2_x_policy_loss + screen2_y_policy_loss
        return {
            'total_loss': total_loss.numpy(),
            'policy_loss': (main_policy_loss + screen_xy_policy_loss
                            + coord_policy_loss).numpy(),
            'value_loss': value_loss.numpy(),
            'entropy': total_entropy.numpy(),
            'main_policy_loss': main_policy_loss.numpy(),
            'main_value_loss': value_loss.numpy(),
            'screen_xy_policy_loss': screen_xy_policy_loss.numpy(),
            'screen_xy_value_loss': value_loss.numpy(),
            'coord_policy_loss': coord_policy_loss.numpy(),
            'coord_value_loss': value_loss.numpy(),
        }

    def save(self, path):
        self.network.save_weights(path)

    def load(self, path):
        self.network.load_weights(path)
