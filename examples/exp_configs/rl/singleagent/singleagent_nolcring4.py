"""Ring road example.

Trains a single autonomous vehicle to stabilize the flow of 21 human-driven
vehicles in a variable length ring road.
"""
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams, SumoLaneChangeParams
from flow.controllers import RLController, IDMController, ContinuousRouter, SimLaneChangeController
from flow.envs import WaveAttenuationPOEnv, LaneChangeAccelEnv, MyLaneChangeAccelEnv, MyLaneChangeAccelEnv5
from flow.networks import RingNetwork

# time horizon of a single rollout
HORIZON = 3000
# number of rollouts per training iteration
N_ROLLOUTS = 20
# number of parallel workers
N_CPUS = 2

# We place one autonomous vehicle and 22 human-driven vehicles in the network
vehicles = VehicleParams()

vehicles.add(
    veh_id='outline',
    acceleration_controller=(IDMController, {'v0':5}),
    routing_controller=(ContinuousRouter, {}),
    initial_speed=5,
    num_vehicles=6,
    car_following_params=SumoCarFollowingParams(
        speed_mode='aggressive',
        min_gap=0,
    )
)
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    initial_speed=5,
    num_vehicles=1)

flow_params = dict(
    # name of the experiment
    exp_tag="no_lane_change_ring",

    # name of the flow environment the experiment is running on
    env_name=MyLaneChangeAccelEnv5,
    # env_name=WaveAttenuationPOEnv,

    # name of the network class the experiment is running on
    network=RingNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.1,
        render=False,
        restart_instance=False
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
        spacing='my',
        lanes_distribution=1,

    ),
)
