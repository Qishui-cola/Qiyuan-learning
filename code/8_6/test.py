
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app



import torch
import numpy as np

from isaaclab.app import AppLauncher
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.actuators import ImplicitActuatorCfg

# 设置 cf2x.usd 的路径（确保路径存在）
USD_PATH = f"/home/qiyuanlab-fyx/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/Crazyflie/cf2x.usd"

# robot = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=USD_PATH,
#         activate_contact_sensors=True,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=False,   # 启用重力（让无人机自然掉落）
#             max_depenetration_velocity=10.0,    # 控制“穿透恢复”速度（防止穿模）
#             enable_gyroscopic_forces=True,      # 启用陀螺力矩，有助于更真实地模拟旋转行为
#         ),
#         collision_props=sim_utils.CollisionPropertiesCfg(
#             collision_enabled=True, # 打开碰撞检测
#         ),
#         mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=False, # 是否允许自碰撞
#             solver_position_iteration_count=4, # 解算器迭代次数（位置解算 4 次、速度解算 0 次）
#             solver_velocity_iteration_count=0,
#             sleep_threshold=0.005, # 睡眠阈值控制什么时候物体变为“休眠”状态
#             stabilization_threshold=0.001,
#         ),
#         copy_from_source=False,
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.25, -0.25, 0.0),
#     ),
    
#     actuators={
#         "dummy": ImplicitActuatorCfg(
#             joint_names_expr=[".*"],
#             stiffness=0.0,
#             damping=0.0,
#         ),
#     },
# )
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
        pos=(0.25, -0.25, 1), # 初始化位置，z值过低会发生碰撞改变方向
    ),
    actuators={
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*"], 
            effort_limit_sim=10000.0,
            velocity_limit_sim=20000.0,
            stiffness=0.0, # 相当于让关节自由运动（因为无人机四个 propeller 实际是速度控制或外力控制）
            damping=0.0,
        ),
    },
)

JETBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Jetbot/jetbot.usd"),
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)

class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    Jetbot = JETBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Jetbot")
    robot = robot.replace(prim_path="{ENV_REGEX_NS}/robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    
    # 初始推力大小（可调试）
    body_ids = scene["robot"].find_bodies("body")[0]  # 从机器人中查找名为 "body" 的刚体部分，用于之后施加力/力矩。
    
    positions = [
    scene["robot"].find_bodies("m1_prop")[0],
    scene["robot"].find_bodies("m2_prop")[0],
    scene["robot"].find_bodies("m3_prop")[0],
    scene["robot"].find_bodies("m4_prop")[0],
    ]

    robot_mass = scene["robot"].root_physx_view.get_masses().sum() # 获取整台机器人的总质量，用于计算浮空所需的总推力。
    gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm() # 获取当前仿真中的重力加速度大小（以 m/s² 计）。

    while simulation_app.is_running():
        # reset
        if count % 350 == 0:  # 时间
            # reset counters
            sim_time = 0.0
            count = 0
            # reset the scene entities to their initial positions offset by the environment origins
            root_jetbot_state = scene["Jetbot"].data.default_root_state.clone()
            root_jetbot_state[:, :3] += scene.env_origins

            # copy the default root state to the sim for the jetbot's orientation and velocity
            scene["Jetbot"].write_root_pose_to_sim(root_jetbot_state[:, :7])
            scene["Jetbot"].write_root_velocity_to_sim(root_jetbot_state[:, 7:])

            # copy the default joint states to the sim
            joint_pos, joint_vel = (
                scene["Jetbot"].data.default_joint_pos.clone(),
                scene["Jetbot"].data.default_joint_vel.clone(),
            )
            scene["Jetbot"].write_joint_state_to_sim(joint_pos, joint_vel)
            
            joint_pos, joint_vel = scene["robot"].data.default_joint_pos, scene["robot"].data.default_joint_vel  # 获取机器人（即 cf2x 无人机）每个关节的初始位置和速度。
            scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)  # 将刚才获得的初始关节状态写入仿真中，更新四个电机的转速和位置。
            scene["robot"].write_root_pose_to_sim(scene["robot"].data.default_root_state[:, :7])   # 重置无人机的位置和姿态。
            scene["robot"].write_root_velocity_to_sim(scene["robot"].data.default_root_state[:, 7:]) # 重置无人机的线速度和角速度。

            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting Jetbot and Dofbot state...")

        # drive around
        if count % 100 < 75:
            # Drive straight by setting equal wheel velocities
            action = torch.Tensor([[10.0, 10.0]])
        else:
            # Turn by applying different velocities
            action = torch.Tensor([[5.0, -5.0]])
        scene["Jetbot"].set_joint_velocity_target(action)
        
        # thrust = torch.zeros(scene["robot"].num_instances, 1, 3, device=sim.device)
        # moment = torch.zeros(scene["robot"].num_instances, 1, 3, device=sim.device)
        # thrust[:, 0, 2] = robot_mass * gravity * 1
        # moment[:, 0, 2] = 0
        # scene["robot"].set_external_force_and_torque(thrust, moment, body_ids=body_ids)

        forces = torch.zeros(scene["robot"].num_instances, 1, 3, device=sim.device)
        moment = torch.zeros(scene["robot"].num_instances, 1, 3, device=sim.device)
        forces[:, :, 2] = robot_mass * gravity * 1 / 4
        moment[:, 0, 2] = 0.005
        
        scene["robot"].set_external_force_and_torque(forces, moment, body_ids=positions[0])
        scene["robot"].set_external_force_and_torque(forces, moment, body_ids=positions[1])
        scene["robot"].set_external_force_and_torque(forces, moment, body_ids=positions[2])
        # forces[:, :, 2] = robot_mass * gravity * 1.1 / 4
        scene["robot"].set_external_force_and_torque(forces, moment, body_ids=positions[3])
        
        # for i in range(4):
        #     scene["robot"].set_external_force_and_torque(forces, moment, body_ids=positions[i])

        scene.write_data_to_sim()
        
        print("Root pos:", scene["robot"].data.root_pos_w.cpu().numpy())  # 查看坐标
        
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)

def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view(eye=[10, 10, 10], target=[2.5, 0.0, 0.0])
    # design scene
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=5.0)
    scene = InteractiveScene(scene_cfg)
    
    # Play the simulator
    sim.reset()

    print("[DEBUG] Jetbot all joints:", scene["Jetbot"].data.joint_names)
    print("[DEBUG] CF2X all joints:", scene["robot"].data.joint_names) #[DEBUG] CF2X all joints: ['m1_joint', 'm2_joint', 'm3_joint', 'm4_joint']

    # Now we are ready!
    
    print("[INFO]: Setup complete...")

    # Run the simulator
    run_simulator(sim, scene)
    
if __name__ == "__main__":
    main()
    simulation_app.close()

