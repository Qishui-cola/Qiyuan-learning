# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to add and simulate on-board sensors for a robot.

We add the following sensors on the quadruped robot, ANYmal-C (ANYbotics):

* USD-Camera: This is a camera sensor that is attached to the robot's base.
* Height Scanner: This is a height scanner sensor that is attached to the robot's base.
* Contact Sensor: This is a contact sensor that is attached to the robot's feet.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/04_sensors/add_sensors_on_robot.py --enable_cameras

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on adding sensors on a robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
# from isaaclab_assets import CRAZYFLIE_NEW_CFG, CRAZYFLIE_CFG
from isaaclab.terrains import TerrainImporterCfg, TerrainImporter, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from isaaclab.actuators import ImplicitActuatorCfg

@configclass
class SensorsSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=r"/home/qiyuanlab-fyx/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/Crazyflie/cf2x.usd",
            activate_contact_sensors=True,      # 激活与其他物体的接触传感器，可以检测是否碰撞或着地。
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,   # 启用重力（让无人机自然掉落）
                max_depenetration_velocity=10.0,    # 控制“穿透恢复”速度（防止穿模）
                enable_gyroscopic_forces=True,      # 启用陀螺力矩，有助于更真实地模拟旋转行为
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, # 打开碰撞检测
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3), # 指定整体质量为 0.3 kg
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, # 是否允许自碰撞
                solver_position_iteration_count=4, # 解算器迭代次数（位置解算 4 次、速度解算 0 次）
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005, # 睡眠阈值控制什么时候物体变为“休眠”状态
                stabilization_threshold=0.001,
            ),
            copy_from_source=False, # 如果是 True，系统会将 USD 文件的原始结构复制到仿真场景中；设为 False 可节省内存/加载时间。
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 3), # 初始化位置
        ),
        actuators={
            "dummy": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0, # 相当于让关节自由运动（因为无人机四个 propeller 实际是速度控制或外力控制）
                damping=0.0,
            ),
        },
    )
    # sensors
    # camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/crazyflie/body/front_cam",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["rgb", "distance_to_image_plane"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(0.15, 0.0, 0.15), rot=(-0.707, 0.707, 0, 0), convention="world"),
    # )
    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/crazyflie/body",
    #     update_period=1 / 60,
    #     offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 0)),
    #     attach_yaw_only=True,
    #     pattern_cfg=patterns.LidarPatternCfg(
    #         channels=100, vertical_fov_range=[-90, 90], horizontal_fov_range=[-90, 90], horizontal_res=1.0
    #     ),
    #     debug_vis=True,
    #     mesh_prim_paths=["/World/ground"],
    # )



def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Fetch relevant parameters to make the quadcopter hover in place
    body_ids = scene["robot"].find_bodies("body")[0]  # 从机器人中查找名为 "body" 的刚体部分，用于之后施加力/力矩。
    robot_mass = scene["robot"].root_physx_view.get_masses().sum() # 获取整台机器人的总质量，用于计算浮空所需的总推力。
    gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm() # 获取当前仿真中的重力加速度大小（以 m/s² 计）。

    # Simulate physics
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset dof state
            joint_pos, joint_vel = scene["robot"].data.default_joint_pos, scene["robot"].data.default_joint_vel  # 获取机器人（即 cf2x 无人机）每个关节的初始位置和速度。
            scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)  # 将刚才获得的初始关节状态写入仿真中，更新四个电机的转速和位置。
            scene["robot"].write_root_pose_to_sim(scene["robot"].data.default_root_state[:, :7])   # 重置无人机的位置和姿态。
            scene["robot"].write_root_velocity_to_sim(scene["robot"].data.default_root_state[:, 7:]) # 重置无人机的线速度和角速度。
            scene["robot"].reset()
            # reset command
            print(">>>>>>>> Reset!")
            # apply action to the robot (make the robot float in place)
        thrust = torch.zeros(scene["robot"].num_instances, 1, 3, device=sim.device)
        moment = torch.zeros(scene["robot"].num_instances, 1, 3, device=sim.device)
        thrust[:, 0, 2] = robot_mass * gravity * 1.1
        moment[:, 0, :] = 0
        scene["robot"].set_external_force_and_torque(thrust, moment, body_ids=body_ids)
        scene.write_data_to_sim()

        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)
        # print information from the sensors
        # print("-------------------------------")
        # print(scene["camera"])
        # print("Received shape of rgb   image: ", scene["camera"].data.output["rgb"].shape)
        # print("Received shape of depth image: ", scene["camera"].data.output["distance_to_image_plane"].shape)
        # print("-------------------------------")
        # print(scene["height_scanner"])
        # print("Received max height value: ", torch.max(scene["height_scanner"].data.ray_hits_w[..., -1]).item())
        # print("-------------------------------")


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[10, 10, 10], target=[2.5, 0.0, 0.0])
    # design scene
    scene_cfg = SensorsSceneCfg(num_envs=args_cli.num_envs, env_spacing=5.0)
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()