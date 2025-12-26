from meshes.mesh_world import MeshWorld
from gaussians.gaussian_world import GaussianWorld
import argparse
import numpy as np
import os
import torch


def robo4d_parse():
    parser = argparse.ArgumentParser(description="Robo4D")
    parser.add_argument("--scene_name", type=str, default="basket_world")
    parser.add_argument("--instruction", type=str, default="put the green cucumber into the basket")
    parser.add_argument("--name", type=str, default='demo')
    parser.add_argument("--image_size", type=int, default=100)
    parser.add_argument("--total_steps", type=int, default=2)
    # parser.add_argument("--total_steps", type=int, default=10)
    parser.add_argument("--camera_view_id", type=int, default=1)
    parser.add_argument("--plane_action", action="store_true")
    parser.add_argument("--cem_iteration", type=int, default=1)
    # parser.add_argument("--cem_iteration", type=int, default=3)
    parser.add_argument("--num_sample_each_group", type=int, default=6)
    parser.add_argument("--num_sample_actions", type=int, default=81)
    parser.add_argument("--num_sample_vlm", type=int, default=36)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--try_release", action="store_true")
    parser.add_argument("--replan", type=bool, default=False)
    # parser.add_argument("--replan", action="store_true")
    
    parser.add_argument("--enable_torch_profiler", type=bool, default=False)
    parser.add_argument("--torch_profiler_dir", type=str, default="profiling_results")
    parser.add_argument("--try_times", type=int, default=1)
    # parser.add_argument("--try_times", type=int, default=5)
    return parser

parser = robo4d_parse()
args = parser.parse_args()

robot_translation = [-0.45, 0.0, 0.0]
distance = 1.5
gaussian_world = GaussianWorld(args.scene_name, parser, post_process=False)
radius = gaussian_world.radius * distance
robot_uids = 'PandaRobotiqHand'
if args.scene_name is None:
    output_path = os.path.join('results', f'{args.instruction}/{args.scene_id}/{args.name}')
else:
    output_path = os.path.join('results', f'{args.instruction}/{args.scene_name}/{args.name}')
close_gripper = False

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
# camera config
# elevation angles of 4 fixed cameras
elev = torch.tensor([-70, 0, 70, 0], device=device)
# azimuth angles of 4 fixed cameras
azim = torch.tensor([0, 70, 0, 0], device=device)

cameras_config = []
for i in range(4):
    cameras_config.append({
        'elev': elev[i].item(),
        'azim': azim[i].item(),
    })
    
mesh_world = MeshWorld(args.scene_name, num_envs=args.num_sample_actions, scene_traslation=-np.array(robot_translation), radius=radius, \
                       image_size=args.image_size, record_video=args.record_video, robot_uids=robot_uids, need_render=True, dir=output_path, \
                        close_gripper=close_gripper, cameras_config=cameras_config)

action_dimenstions = 6
means = np.zeros(action_dimenstions)
covariance = np.zeros((action_dimenstions, action_dimenstions))

samples = np.random.multivariate_normal(means, covariance, size=args.num_sample_actions)

joint_angles_list, action_object_transformations, post_samples, robot_images, robot_depth_images = mesh_world.sample_action_distribution_batch(samples, profile=True)