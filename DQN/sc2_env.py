# Env adapter used by the DQN training loop. Wraps pysc2.env.sc2_env so the
# loop only needs reset / step / close, plus a helper that returns the action
# ids the env will accept this step.
from pysc2.env import sc2_env
from pysc2.lib import actions, features
import numpy as np


# argument-name -> how to format that arg coming out of the agent's dict.
# the keys here are the pysc2 arg_type.name values we expect to encounter;
# every other arg_type falls through to the default zero-fill below.
def _fmt_queued(args_in):
    return ('queued', [args_in['queued'][0].item()])


def _fmt_screen(args_in):
    return ('screen', [args_in['screen'][0].item(), args_in['screen'][1].item()])


def _fmt_screen2(args_in):
    return ('screen2', [args_in['screen2'][0].item(), args_in['screen2'][1].item()])


_ARG_FORMATTERS = {
    'queued':  _fmt_queued,
    'screen':  _fmt_screen,
    'minimap': _fmt_screen,   # minimap re-uses the screen coordinates
    'screen2': _fmt_screen2,
}


class SC2Env:
    def __init__(self, map_name, visualize=False, step_mul=8, realtime=False):
        # 84x84 screen, 64x64 minimap. FEATURES action space, which means the
        # agent emits raw pysc2 function ids and pixel coords (no high-level
        # macros), so the wrapper has to translate them into FunctionCalls.
        aif_kwargs = {
            'feature_dimensions': features.Dimensions(screen=84, minimap=64),
            'action_space': actions.ActionSpace.FEATURES,
            'use_feature_units': True,
        }
        self.env = sc2_env.SC2Env(
            map_name=map_name,
            players=[sc2_env.Agent(sc2_env.Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(**aif_kwargs),
            step_mul=step_mul,
            game_steps_per_episode=None,
            visualize=visualize,
            realtime=realtime,
        )

    def reset(self):
        self.current_obs = self.env.reset()[0]
        return self._process_state(self.current_obs)

    def step(self, action, args):
        action_id, args = self._process_action_model(action, args)
        fn_call = actions.FunctionCall(action_id, [v[1] for v in args])
        ts = self.env.step([fn_call])[0]
        self.current_obs = ts
        state = self._process_state(ts)
        return state, ts.reward, ts.last(), args

    # set of action ids the env will accept this step
    def get_available_actions(self):
        return self.current_obs.observation["available_actions"]

    # two-channel screen state: player_relative (who owns each pixel)
    # and selected (binary mask of pixels the player has selected)
    def _process_state(self, obs):
        screen = obs.observation.feature_screen
        layer_idx = (
            features.SCREEN_FEATURES.player_relative.index,
            features.SCREEN_FEATURES.selected.index,
        )
        return np.asarray([screen[i] for i in layer_idx], dtype=np.uint8)

    # builds the FunctionCall for a chosen action id. illegal ids (out of range
    # or not in available_actions for this step) collapse to no_op so the env
    # never sees a malformed call.
    def _process_action_model(self, action_id, args_input):
        action_id = int(action_id)
        if not (0 <= action_id < len(actions.FUNCTIONS)):
            return actions.FUNCTIONS.no_op.id, []
        if action_id not in self.current_obs.observation["available_actions"]:
            return actions.FUNCTIONS.no_op.id, []

        action_info = actions.FUNCTIONS[action_id]
        formatted = []
        for arg_type in action_info.args:
            fmt = _ARG_FORMATTERS.get(arg_type.name)
            if fmt is None:
                # nothing predicted for this arg type - default to zero
                formatted.append((arg_type.name, [0]))
            else:
                formatted.append(fmt(args_input))
        return action_id, formatted

    def close(self):
        if hasattr(self, 'env'):
            self.env.close()
