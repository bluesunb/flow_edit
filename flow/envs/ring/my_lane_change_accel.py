from flow.envs.ring.lane_change_accel import LaneChangeAccelEnv
from flow.core import rewards

from gym.spaces.box import Box
from gym.spaces.tuple import Tuple
from gym.spaces.multi_discrete import MultiDiscrete

import numpy as np

from collections import defaultdict
from pprint import pprint

class MyLaneChangeAccelEnv(LaneChangeAccelEnv):
    """
    LC 학습을 위해 discrete한 action과 새로운 reward를 추가한 환경.
    """
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)
        self.accumulated_reward = None
        self.last_lc = None

    def _to_lc_action(self, rl_action):
        """Make direction components of rl_action to discrete"""
        if rl_action is None:
            return rl_action
        for i in range(1, len(rl_action), 2):
            if rl_action[i] < -1 + 2 / 3:
                rl_action[i] = -1
            elif rl_action[i] >= -1 + 4 / 3:
                rl_action[i] = 1
            else:
                rl_action[i] = 0
        return rl_action

    def _apply_rl_actions(self, actions):
        actions = self._to_lc_action(actions)

        acceleration = actions[::2]
        direction = actions[1::2]

        self.last_lc = self.k.vehicle.get_lane(self.k.vehicle.get_rl_ids())

        # re-arrange actions according to mapping in observation space
        sorted_rl_ids = [veh_id for veh_id in self.sorted_ids
            if veh_id in self.k.vehicle.get_rl_ids()]

        # represents vehicles that are allowed to change lanes
        non_lane_changing_veh = \
            [self.time_counter <=
             self.env_params.additional_params["lane_change_duration"]
             + self.k.vehicle.get_last_lc(veh_id)
             for veh_id in sorted_rl_ids]

        # vehicle that are not allowed to change have their directions set to 0
        direction[non_lane_changing_veh] = \
            np.array([0] * sum(non_lane_changing_veh))

        self.k.vehicle.apply_acceleration(sorted_rl_ids, acc=acceleration)
        self.k.vehicle.apply_lane_change(sorted_rl_ids, direction=direction)

    def compute_reward(self, rl_actions, **kwargs):
        lc_action = self._to_lc_action(rl_actions)
        reward = rewards.total_lc_reward(self, lc_action)
        if self.accumulated_reward is None:
            self.accumulated_reward = reward
        else:
            self.accumulated_reward += reward

        if self.time_counter == self.env_params.horizon \
                    + self.env_params.warmup_steps - 1:
            # reward = [rl_mean_speed, simple_lc_penalty, dc3, unsf, rl_action_p]
            pprint(self.initial_config.reward_params)
            print('=== now reward ===')
            print(f'[now]: {reward.round(3)}')
            print('=== accumulated reward ===')
            print(f'[accu]: {self.accumulated_reward.round(3)}')

        return sum(reward)

class TestLCEnv(MyLaneChangeAccelEnv):

    @property
    def action_space(self):
        """
        return: Tuple(Box, MultiDiscrete)
        0: direction -1
        1: direction 0
        2: direction 1
        """
        max_decel = self.env_params.additional_params["max_decel"]
        max_accel = self.env_params.additional_params["max_accel"]

        lb = [-abs(max_decel)] * self.initial_vehicles.num_rl_vehicles  # lower bound of acceleration
        ub = [max_accel] * self.initial_vehicles.num_rl_vehicles  # upper bound of acceleration

        lc_b = [3] * self.initial_vehicles.num_rl_vehicles  # boundary of lane change direction

        # return Box(np.array(lb), np.array(ub), dtype=np.float32)
        return Tuple((Box(np.array(lb), np.array(ub), dtype=np.float32), MultiDiscrete(np.array(lc_b))))

    def compute_reward(self, rl_actions, **kwargs):
        reward, rwds = rewards.full_reward(self, rl_actions)

        if self.accumulated_reward is None:
            self.accumulated_reward = defaultdict(int)
        else:
            for k in rwds.keys():
                self.accumulated_reward[k] += rwds[k]

        if self.time_counter == self.env_params.horizon \
                    + self.env_params.warmup_steps - 1:
            print('=== now reward ===')
            pprint(dict(rwds))
            print('=== accumulated reward ===')
            pprint(dict(self.accumulated_reward))

        return reward

    def _apply_rl_actions(self, actions):
        acceleration, raw_direction = actions
        direction = raw_direction - 1

        self.last_lc = self.k.vehicle.get_lane(
            self.k.vehicle.get_rl_ids())

        # re-arrange actions according to mapping in observation space
        sorted_rl_ids = [
            veh_id for veh_id in self.sorted_ids
            if veh_id in self.k.vehicle.get_rl_ids()
        ]

        # represents vehicles that are allowed to change lanes
        non_lane_changing_veh = \
            [self.time_counter <=
             self.env_params.additional_params["lane_change_duration"]
             + self.k.vehicle.get_last_lc(veh_id)
             for veh_id in sorted_rl_ids]
        # vehicle that are not allowed to change have their directions set to 0
        direction[non_lane_changing_veh] = \
            np.array([0] * sum(non_lane_changing_veh))

        self.k.vehicle.apply_acceleration(sorted_rl_ids, acc=acceleration)
        self.k.vehicle.apply_lane_change(sorted_rl_ids, direction=direction)

class X(TestLCEnv):

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)
        self.dir = 1

    def compute_reward(self, rl_actions, **kwargs):
        return np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))/1000

    def _apply_rl_actions(self, actions):
        timestep = self.k.vehicle.get_timestep(self.k.vehicle.get_rl_ids()[0])
        lane = self.k.vehicle.get_lane(self.k.vehicle.get_rl_ids()[0])
        num_rl = self.k.vehicle.num_rl_vehicles
        acceleration, raw_direction = actions
        direction = np.array([d-1 for d in raw_direction])
        sorted_rl_ids = [veh_id for veh_id in self.sorted_ids if veh_id in self.k.vehicle.get_rl_ids()]

        if timestep%100:
            self.dir = 0
        elif self.dir + lane < 0:
            self.dir = 1
        elif self.dir + lane >= self.net_params.additional_params["lanes"]:
            self.dir = -1

        self.k.vehicle.apply_acceleration(sorted_rl_ids, acceleration)
        self.k.vehicle.apply_lane_change(sorted_rl_ids, [self.dir]*num_rl)