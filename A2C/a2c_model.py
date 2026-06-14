# Keras model behind the A2C agent. FullyConv body, one spatial-policy head
# for screen coordinates, plus separate heads for the other action arguments.
import tensorflow as tf


# multi-head Actor-Critic network:
#   two conv layers pull spatial features from the screen, a 1x1 conv
#   on top of them makes the per-pixel spatial policy (used for the
#   screen argument), and Global Average Pooling gives us a vector
#   that feeds the other heads - main action, value baseline, the two
#   screen2 coordinates, the queue flag, and select_unit_id (which unit
#   we pick out of the current multi-select pool).
class A2CNetwork(tf.keras.Model):

    def __init__(self, num_actions, screen_size):
        super(A2CNetwork, self).__init__()
        self.num_actions = num_actions
        self.screen_size = screen_size

        # Convolutional feature extractor
        self.conv1 = tf.keras.layers.Conv2D(
            16, 5, strides=1, activation='relu', padding='same',
            data_format='channels_last', name='Conv1')
        self.conv2 = tf.keras.layers.Conv2D(
            32, 3, strides=1, activation='relu', padding='same',
            data_format='channels_last', name='Conv2')

        # spatial policy head: 1x1 conv keeps the screen grid,
        # then a softmax over the 84*84 = 7056 pixels picks the target
        self.screen_xy_conv = tf.keras.layers.Conv2D(1, 1, name='ScreenXYConv')
        self.screen_xy_flatten = tf.keras.layers.Flatten(name='ScreenXYFlatten')
        self.screen_xy_softmax = tf.keras.layers.Softmax(name='ScreenXYSoftmax')
        self.screen_xy_reshape = tf.keras.layers.Reshape(
            (screen_size, screen_size), name='ScreenXYReshape')

        # GAP collapses the conv features into a vector for the other heads
        self.gap = tf.keras.layers.GlobalAveragePooling2D(
            data_format='channels_last', name='GAP')
        self.shared_dense = tf.keras.layers.Dense(
            256, activation='relu', name='SharedDense')

        # main action head - softmax over the 524 PySC2 function ids
        self.policy_dense = tf.keras.layers.Dense(
            256, activation='relu', name='PolicyDense')
        self.action_logits_layer = tf.keras.layers.Dense(
            num_actions, name='ActionLogits')
        self.action_probs_layer = tf.keras.layers.Softmax(name='ActionProbs')

        # value baseline V(s) - one critic shared by every policy head
        self.value_dense = tf.keras.layers.Dense(
            256, activation='relu', name='ValueDense')
        self.value_output = tf.keras.layers.Dense(1, name='ValueOutput')

        # screen2 (x, y) - the second corner used by select_rect-style actions
        self.screen2_x_dense = tf.keras.layers.Dense(
            64, activation='relu', name='Screen2XDense')
        self.screen2_x_logits_layer = tf.keras.layers.Dense(
            screen_size, name='Screen2XLogits')
        self.screen2_y_dense = tf.keras.layers.Dense(
            64, activation='relu', name='Screen2YDense')
        self.screen2_y_logits_layer = tf.keras.layers.Dense(
            screen_size, name='Screen2YLogits')

        # queue flag (run now vs append to action queue)
        self.queue_dense = tf.keras.layers.Dense(
            64, activation='relu', name='QueueDense')
        self.queue_logits_layer = tf.keras.layers.Dense(2, name='QueueLogits')

        # select_unit_id head - picks one unit out of the multi_select pool.
        # output dim 16 is enough for the mini-games we use here
        self.select_unit_id_dense = tf.keras.layers.Dense(
            64, activation='relu', name='SelectUnitIdDense')
        self.select_unit_id_logits_layer = tf.keras.layers.Dense(
            16, name='SelectUnitIdLogits')

    @tf.function
    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)
        x = tf.transpose(x, [0, 2, 3, 1])   # NCHW -> NHWC

        # convolutional features
        x = self.conv1(x)
        x = self.conv2(x)

        # spatial policy: per-pixel softmax over the screen
        screen_xy_map = self.screen_xy_conv(x)
        screen_xy_flat = self.screen_xy_flatten(screen_xy_map)
        screen_xy_probs_flat = self.screen_xy_softmax(screen_xy_flat)
        screen_xy_probs = self.screen_xy_reshape(screen_xy_probs_flat)

        # pooled feature vector that feeds every non-spatial head
        gap_out = self.gap(x)
        shared = self.shared_dense(gap_out)

        # main action head
        policy_features = self.policy_dense(shared)
        action_logits = self.action_logits_layer(policy_features)
        action_probs = self.action_probs_layer(action_logits)

        # value baseline
        value_features = self.value_dense(shared)
        value = self.value_output(value_features)

        # screen2 x and y
        screen2_x_logits = self.screen2_x_logits_layer(self.screen2_x_dense(shared))
        screen2_x_probs = tf.nn.softmax(screen2_x_logits)
        screen2_y_logits = self.screen2_y_logits_layer(self.screen2_y_dense(shared))
        screen2_y_probs = tf.nn.softmax(screen2_y_logits)

        # queue head
        queue_logits = self.queue_logits_layer(self.queue_dense(shared))
        queue_probs = tf.nn.softmax(queue_logits)

        # select_unit_id head
        select_unit_id_logits = self.select_unit_id_logits_layer(
            self.select_unit_id_dense(shared))
        select_unit_id_probs = tf.nn.softmax(select_unit_id_logits)

        return {
            'action_logits': action_logits,
            'action_probs': action_probs,
            'value': value,
            'screen_xy_probs': screen_xy_probs,
            'screen2_x_logits': screen2_x_logits,
            'screen2_x_probs': screen2_x_probs,
            'screen2_y_logits': screen2_y_logits,
            'screen2_y_probs': screen2_y_probs,
            'queue_logits': queue_logits,
            'queue_probs': queue_probs,
            'select_unit_id_logits': select_unit_id_logits,
            'select_unit_id_probs': select_unit_id_probs,
        }
