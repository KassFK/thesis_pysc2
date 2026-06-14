# Small adapter that gives the training loop a clean reset / step / close
# surface over pysc2.env.sc2_env, plus helpers for the available action ids
# and the size of the current multi_select pool. Conversion from the agent's
# argument dict to a pysc2 FunctionCall lives here too.
from pysc2.env import sc2_env
from pysc2.lib import actions, features
import numpy as np


class SC2Env:
    def __init__(self, map_name, visualize=False, step_mul=64, realtime=False):
        # 84x84 screen + 64x64 minimap feature layers, raw FEATURES action
        # space (the agent picks PySC2 function ids and screen coords directly)
        self.env = sc2_env.SC2Env(
            map_name=map_name,
            players=[sc2_env.Agent(sc2_env.Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=84, minimap=64),
                action_space=actions.ActionSpace.FEATURES,
                use_feature_units=True,
            ),
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
        self.current_obs = self.env.step(
            [actions.FunctionCall(action_id, [x[1] for x in args])])[0]
        state = self._process_state(self.current_obs)
        reward = self.current_obs.reward
        done = self.current_obs.last()
        return state, reward, done, args

    # which action ids the env allows on the current step
    def get_available_actions(self):
        return self.current_obs.observation["available_actions"]

    # how many units are currently in the multi_select pool - the agent's
    # select_unit_id head uses this to know which output indices are real
    def get_multi_select_size(self):
        return len(self.current_obs.observation.multi_select)

    # two-channel screen state: player_relative (who owns each pixel:
    # background / self / ally / neutral / enemy) and selected (a binary
    # mask of pixels belonging to the units the player has selected)
    def _process_state(self, obs):
        screen = obs.observation.feature_screen
        return np.stack([
            screen[features.SCREEN_FEATURES.player_relative.index],
            screen[features.SCREEN_FEATURES.selected.index],
        ])

    # turn the agent's args dict into the FunctionCall PySC2 expects.
    # we walk the chosen action's required arg list and pick each value
    # out of the agent's dict; anything the agent doesn't predict
    # (e.g. select_unit_act) defaults to 0. invalid action ids fall back to no_op.
    def _process_action_model(self, action_id, args_input):
        action_id = int(action_id)
        if not (0 <= action_id < len(actions.FUNCTIONS)):
            return actions.FUNCTIONS.no_op.id, []
        if action_id not in self.current_obs.observation["available_actions"]:
            return actions.FUNCTIONS.no_op.id, []

        action_info = actions.FUNCTIONS[action_id]
        args = []
        for _, arg_type in enumerate(action_info.args):
            if arg_type.name == 'queued':
                args.append(('queued', [args_input['queued'][0].item()]))
            elif arg_type.name == 'screen' or arg_type.name == 'minimap':
                args.append(('screen',
                             [args_input['screen'][0].item(),
                              args_input['screen'][1].item()]))
            elif arg_type.name == 'screen2':
                args.append(('screen2',
                             [args_input['screen2'][0].item(),
                              args_input['screen2'][1].item()]))
            elif arg_type.name == 'select_unit_id':
                v = args_input['select_unit_id'][0]
                args.append(('select_unit_id',
                             [v.item() if hasattr(v, 'item') else int(v)]))
            else:
                args.append((arg_type.name, [0]))
        return action_id, args

    def close(self):
        if hasattr(self, 'env'):
            self.env.close()
