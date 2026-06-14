# Q-network used by the DQN agent. Same FullyConv backbone as the actor-critic
# model, so the A2C vs DQN comparison isolates the learning rule, not the net.
import tensorflow as tf


# Six-headed Q-network. The trunk (two convs + GAP + a wide Dense) is shared,
# and each action component branches off it through its own small MLP. The
# joint Q decomposes additively across heads so each branch trains against
# its own Bellman target independently.
class DQNNetwork(tf.keras.Model):

    def __init__(self, num_actions, screen_size):
        super(DQNNetwork, self).__init__()
        self.num_actions = num_actions
        self.screen_size = screen_size

        # trunk: two conv layers + global pool + a wide dense
        self.backbone_conv_a = tf.keras.layers.Conv2D(
            16, 5, strides=1, activation='relu', padding='same',
            data_format='channels_last', name='backbone_conv_a')
        self.backbone_conv_b = tf.keras.layers.Conv2D(
            32, 3, strides=1, activation='relu', padding='same',
            data_format='channels_last', name='backbone_conv_b')
        self.pool = tf.keras.layers.GlobalAveragePooling2D(
            data_format='channels_last', name='pool')
        self.trunk_dense = tf.keras.layers.Dense(
            256, activation='relu', name='trunk_dense')

        # head-builder helper: a non-linear projection followed by a linear
        # readout sized to that branch's action component
        def _branch(hidden, out_dim, tag):
            return (
                tf.keras.layers.Dense(hidden, activation='relu', name=f'{tag}_pre'),
                tf.keras.layers.Dense(out_dim, name=tag),
            )

        # primary action id over the 524 pysc2 functions
        self.q_main_pre, self.q_main = _branch(256, num_actions, 'q_main')

        # primary screen coordinates - two independent 1-d branches
        self.q_screen_x_pre, self.q_screen_x = _branch(128, screen_size, 'q_screen_x')
        self.q_screen_y_pre, self.q_screen_y = _branch(128, screen_size, 'q_screen_y')

        # screen2 (the second corner used by rect-style actions)
        self.q_screen2_x_pre, self.q_screen2_x = _branch(64, screen_size, 'q_screen2_x')
        self.q_screen2_y_pre, self.q_screen2_y = _branch(64, screen_size, 'q_screen2_y')

        # queue flag (0 = run now, 1 = append to action queue)
        self.q_queue_pre, self.q_queue = _branch(64, 2, 'q_queue')

    # one-line MLP head, used identically by every non-trunk branch
    def _head(self, trunk, pre, out):
        return out(pre(trunk))

    @tf.function
    def call(self, inputs):
        h = tf.cast(inputs, tf.float32)
        h = tf.transpose(h, perm=(0, 2, 3, 1))   # NCHW -> NHWC

        # trunk
        h = self.backbone_conv_a(h)
        h = self.backbone_conv_b(h)
        trunk = self.trunk_dense(self.pool(h))

        # branches - every head goes through the same _head(...) call shape
        return {
            'q_main':      self._head(trunk, self.q_main_pre,      self.q_main),
            'q_screen_x':  self._head(trunk, self.q_screen_x_pre,  self.q_screen_x),
            'q_screen_y':  self._head(trunk, self.q_screen_y_pre,  self.q_screen_y),
            'q_screen2_x': self._head(trunk, self.q_screen2_x_pre, self.q_screen2_x),
            'q_screen2_y': self._head(trunk, self.q_screen2_y_pre, self.q_screen2_y),
            'q_queue':     self._head(trunk, self.q_queue_pre,     self.q_queue),
        }
