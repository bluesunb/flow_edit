"""A series of reward functions."""

from gym.spaces import Box, Tuple
import numpy as np

from collections import defaultdict


def total_lc_reward(env, rl_action):
    """Put all of the reward functions we consider into list

    Parameters
    ----------
    env :
        the environment variable, which contains information on the current
        state of the system.

    rl_action : gym.spaces.Space

    Returns
    -------
    List
        list of reward values
    """
    reward_list = [
        rl_mean_speed(env),
        simple_lc_penalty(env),
        punish_emergency_decel2(env),
        unsafe_distance_penalty(env),
        rl_action_penalty(env, rl_action),

    ]
    return np.array(reward_list)

def rl_mean_speed(env):
    vel = np.array(env.k.vehicle.get_speed(env.k.vehicle.get_rl_ids()))
    coeff = env.initial_config.reward_params.get('rl_mean_speed', 0)
    target_vel = env.env_params.additional_params['target_velocity']
    if coeff == 0:
        return 0

    if any(vel < -100):
        return 0.
    if len(vel) == 0:
        return 0.
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
        return -action_penalty * sum(lc_failed)
    else:
        return 0

def simple_lc_penalty(env):
    sim_lc_penalty = env.initial_config.reward_params.get('simple_lc_penalty', 0)
    reward = 0
    for veh_id in env.k.vehicle.get_rl_ids():
        if env.k.vehicle.get_last_lc(veh_id) == env.time_counter:
            reward -= sim_lc_penalty
    return reward

def punish_emergency_decel2(env):
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

    decel2_coeff = env.initial_config.reward_params.get('dc2_penalty', 0)
    decel3_coeff = env.initial_config.reward_params.get('dc3_penalty', 0)
    if decel2_coeff == 0 and decel3_coeff == 0:
        return 0
    max_decel = -env.env_params.additional_params['max_decel']
    accel_list = [env.k.vehicle.get_accel(veh_id) for veh_id in env.k.vehicle.get_ids()
                  if env.k.vehicle.get_accel(veh_id) is not None and env.k.vehicle.get_accel(veh_id) < max_decel]
    accel = np.array(accel_list)
    decel_gap_list = max_decel - accel
    d_accel = decel_gap_list / abs(2 * max_decel)

    if decel3_coeff:
        d_accel[np.where(d_accel > 1)] = np.log(d_accel[d_accel > 1]) + 1
        return -decel3_coeff * sum(d_accel)

    accel_cliped = d_accel.clip(min=None, max=1)
    return -decel2_coeff * sum(accel_cliped)


def unsafe_distance_penalty(env):
    reward = 0
    max_decel = env.env_params.additional_params['max_decel']
    unsafe_penalty = env.initial_config.reward_params.get('unsafe_penalty', 0)
    if unsafe_penalty == 0:
        return 0
    get_safe_distance = lambda vid: env.k.vehicle.get_speed(vid) ** 2 / (
            2 * max_decel)
    base_distance = 5

    rls = env.k.vehicle.get_rl_ids()
    lane_tailways = env.k.vehicle.get_lane_tailways(rls)
    lanes = env.k.vehicle.get_lane(rls)
    tailways = np.array([lane_tailways[i][lanes[i]] for i in range(len(rls))])
    lane_followers_speed = env.k.vehicle.get_lane_followers_speed(rls)
    follower_speed = np.array([lane_followers_speed[i][lanes[i]] for i in range(len(rls))])
    safe_distances = follower_speed ** 2 / 2 * max_decel + base_distance
    penalties = -unsafe_penalty * max(0, safe_distances - tailways) / safe_distances

    reward = sum(penalties)

    return reward


def full_reward(env, rl_action):
    rls = env.k.vehicle.get_rl_ids()
    reward = 0

    rl_mean_s = env.initial_config.reward_params.get('rl_mean_speed', 0)
    simple_lc_p = env.initial_config.reward_params.get('simple_lc_penalty', 0)
    unsafe_p = env.initial_config.reward_params.get('unsafe_penalty', 0)
    dc3_p = env.initial_config.reward_params.get('dc3_penalty', 0)
    rl_action_p = env.initial_config.reward_params.get('rl_action_penalty', 0)

    rwds = defaultdict(int)

    for rl in rls:
        if rl_mean_s:
            r = rl_mean_s * env.k.vehicle.get_speed(rl) / env.env_params.additional_params['target_velocity']
            reward += r
            rwds['rl_mean_speed'] += r
        if simple_lc_p and env.time_counter == env.k.vehicle.get_last_lc(rl):
            reward -= simple_lc_p
            rwds['simple_lc_penalty'] -= simple_lc_p
        if unsafe_p and env.k.vehicle.get_tailway(rl) < 5:
            pen = unsafe_p * (5. - env.k.vehicle.get_tailway(rl)) / 5.
            reward -= pen
            rwds['unsafe_penalty'] -= pen
        if dc3_p and (env.k.vehicle.get_accel(env.k.vehicle.get_follower(rl)) or 0) < -0.2:
            pen = dc3_p * np.log(abs(env.k.vehicle.get_accel(env.k.vehicle.get_follower(rl))) + 1)
            reward -= pen
            rwds['dc3_penalty'] -= pen

    if rl_action_p:
        pen = rl_action_penalty(env, rl_action)
        reward += pen
        rwds['rl_action_penalty'] += pen

    return reward, rwds