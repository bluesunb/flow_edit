"""Used as an example of ring experiment.

This example consists of 22 IDM cars on a ring creating shockwaves.
"""

from flow.controllers import IDMController, ContinuousRouter
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.envs.ring.lane_change_accel import MyLaneChangeAccelEnv
from flow.networks.ring import RingNetwork, ADDITIONAL_NET_PARAMS

HORIZON = 3000

vehicles = VehicleParams()
vehicles.add(
    veh_id='outline',
    acceleration_controller=(IDMController, {'v0':2}),
    routing_controller=(ContinuousRouter, {}),
    initial_speed=3,
    num_vehicles=6,
    car_following_params=SumoCarFollowingParams(
        speed_mode='aggressive',
        min_gap=0
    )
)

# vehicles.add(
#     veh_id='inline',
#     acceleration_controller=(IDMController, {'v0':2}),
#     routing_controller=(ContinuousRouter, {}),
#     initial_speed=3,
#     num_vehicles=3,
#     car_following_params=SumoCarFollowingParams(
#         speed_mode='aggressive',
#         min_gap=0
#     )
# )

# vehicles.add(
#     veh_id="rl",
#     acceleration_controller=(RLController, {}),
#     routing_controller=(ContinuousRouter, {}),
#     lane_change_controller=(SimLaneChangeController, {}),
#     lane_change_params=SumoLaneChangeParams(
#         lane_change_mode=1621
#     ),
#     num_vehicles=1)

flow_params = dict(
    # name of the experiment
    exp_tag='ring',

    # name of the flow environment the experiment is running on
    # env_name=AccelEnv,
    env_name = MyLaneChangeAccelEnv,

    # name of the network class the experiment is running on
    network=RingNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        render=True,
        sim_step=0.1,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=750,
        clip_actions=False,
        additional_params={
            "max_accel": 3,
            "max_decel": 3,
            "ring_length": [220, 270],
            "lane_change_duration": 5,
            "target_velocity": 10,
            'sort_vehicles': False
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params={
            "length": 260,
            "lanes": 2,
            "speed_limit": 30,
            "resolution": 40,

        }, ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        lanes_distribution=1,
        # spacing='my2',
        spacing='my',
        additional_params={
            'inline_veh_nums' : sum(['inline' in vid for vid in vehicles.ids]),
            'outline_veh_nums' : sum(['outline' in vid for vid in vehicles.ids]),
        }
    ),
)
