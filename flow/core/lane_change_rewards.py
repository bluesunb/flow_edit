"""A series of reward functions."""

from gym.spaces import Box, Tuple
import numpy as np

from collections import defaultdict
from functools import reduce

def rl_mean_speed(env):
    vel = np.array(env.k.vehicle.get_speed(env.k.vehicle.get_rl_ids()))
    coeff = env.initial_config.reward_params.get('rl_mean_speed', 0)
    target_vel = env.env_params.additional_params['target_velocity']

    if coeff == 0:
        return 0

    if any(vel < -100):
        return 0
    if len(vel) == 0:
        return 0
    return np.mean(vel) * coeff / target_vel


def rl_action_penalty(env, rl_action):
    """

    Parameters
    ----------
    env: flow.envs.ring.my_lane_change_accel.MyLaneChangeAccelEnv
    rl_action

    Returns
    -------

    """
    action_penalty = env.initial_config.reward_params.get('rl_action_penalty', 0)
    if rl_action is None or action_penalty == 0:
        return 0

    rls = env.k.vehicle.get_rl_ids()
    lc_failed = np.array(env.last_lane) == np.array(env.k.vehicle.get_lane(rls))
    lc_rl_action = np.zeros_like(rl_action)

    if isinstance(env.action_space, Box):
        lc_rl_action = np.array(rl_action[1::2])
    elif isinstance(env.action_space, Tuple):
        lc_rl_action = rl_action[1] - 1
    if any(lc_rl_action) and any(lc_failed):
        ret = -action_penalty * sum(lc_failed)
    else:
        ret = 0
    return ret


def simple_lc_penalty(env):
    sim_lc_penalty = env.initial_config.reward_params.get('simple_lc_penalty', 0)
    if not sim_lc_penalty:
        return 0
    reward = 0
    for veh_id in env.k.vehicle.get_rl_ids():
        if env.k.vehicle.get_last_lc(veh_id) == env.time_counter:
            reward -= sim_lc_penalty
    return reward


def follower_decel_penalty(env):
    """Reward function used to reward the RL vehicles cause the emergency stop of non RL vehicles

    Parameters
    ----------
    env : flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.

    Returns
    -------
    float
        reward value
    """
    dc3_p = env.initial_config.reward_params.get('dc3_penalty', 0)
    if not dc3_p:
        return 0
    reward = 0
    threshold = -0.2
    rls = env.k.vehicle.get_rl_ids()
    max_decel = env.env_params.additional_params['max_decel']
    for rl in rls:
        follower = env.k.vehicle.get_follower(rl)
        if follower is not None:
            follower_accel = env.k.vehicle.get_accel(env.k.vehicle.get_follower(rl)) or 0
            rl_accel = env.k.vehicle.get_accel(rl) or 0
            accel = np.array([follower_accel, rl_accel])
            accel = accel[accel < threshold]
            if len(accel):
                def f(x):
                    x[x>1] = np.log(x[x>1]) + 1
                    return sum(x)
                pen = dc3_p * f(abs(2 * accel / max_decel))
                reward -= pen

    return reward


def unsafe_distance_penalty(env):
    from flow.core.kernel.network.base import VEHICLE_LENGTH
    unsafe_p = env.initial_config.reward_params.get('unsafe_penalty', 0)
    if not unsafe_p:
        return 0
    rls = env.k.vehicle.get_rl_ids()
    max_decel = env.env_params.additional_params['max_decel']
    reward = 0
    for rl in rls:
        follower = env.k.vehicle.get_follower(rl)
        if follower is not None:
            tailway = env.k.vehicle.get_tailway(rl)
            gap = VEHICLE_LENGTH + env.k.vehicle.get_speed(env.k.vehicle.get_follower(rl)) ** 2 / (
                    2 * max_decel)
            if tailway < gap:
                pen = unsafe_p * (gap - tailway) / gap
                reward -= pen

    return reward


def full_reward(env, rl_action):
    rewards = {
        'rl_mean_speed': rl_mean_speed(env),
        'simple_lc_penalty': simple_lc_penalty(env),
        'rl_action_penalty': rl_action_penalty(env, rl_action),
        'unsafe_distance': unsafe_distance_penalty(env),
        'decel_penalty': follower_decel_penalty(env),
    }

    return rewards


# def full_reward2(env, rl_action):
#     "total reward function v2"
#     rls = env.k.vehicle.get_rl_ids()
#     reward = 0
#
#     rl_mean_s = env.initial_config.reward_params.get('rl_mean_speed', 0)
#     simple_lc_p = env.initial_config.reward_params.get('simple_lc_penalty', 0)
#     unsafe_p = env.initial_config.reward_params.get('unsafe_penalty', 0)
#     dc3_p = env.initial_config.reward_params.get('dc3_penalty', 0)
#     rl_action_p = env.initial_config.reward_params.get('rl_action_penalty', 0)
#
#     rwds = defaultdict(int)
#
#     for rl in rls:
#         if rl_mean_s:
#             r = rl_mean_s * env.k.vehicle.get_speed(rl) / env.env_params.additional_params['target_velocity']
#             reward += r
#             rwds['rl_mean_speed'] += r
#         if simple_lc_p and env.time_counter == env.k.vehicle.get_last_lc(rl):
#             reward -= simple_lc_p
#             rwds['simple_lc_penalty'] -= simple_lc_p
#
#         follower = env.k.vehicle.get_follower(rl)
#         if follower is not None:
#             if unsafe_p:
#                 tailway = env.k.vehicle.get_tailway(rl)
#                 gap = 5 + env.k.vehicle.get_speed(env.k.vehicle.get_follower(rl)) ** 2 / (
#                         2 * env.env_params.additional_params['max_decel'])
#                 if tailway < gap:
#                     pen = unsafe_p * (gap - tailway) / gap
#                     reward -= pen
#                     rwds['unsafe_penalty'] -= pen
#             if dc3_p:
#                 accel = env.k.vehicle.get_accel(env.k.vehicle.get_follower(rl)) or 0
#                 if accel < -0.2:
#                     f = lambda x: x if x < 1 else np.log(x) + 1
#                     pen = dc3_p * f(abs(accel))
#                     reward -= pen
#                     rwds['dc3_penalty'] -= pen
#
#     if rl_action_p:
#         pen = rl_action_penalty(env, rl_action)
#         reward += pen
#         rwds['rl_action_penalty'] += pen
#
#     return reward, rwds
