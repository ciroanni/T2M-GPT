from Motion.InverseKinematics import animation_from_positions
from Motion import BVH
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Process motion data from npy file.')
parser.add_argument('--prompt', type=str, required=False, help='Prompt')
parser.add_argument('--input_file', type=str, required=True, help='Path to the input npy file')
parser.add_argument('--output_dir', type=str, required=False, help='Path to the output bvh files')
parser.add_argument('--iterations', type=int, required=False, help='Number of iterations for the IK solver')
args = parser.parse_args()

# Load data from the provided file path
npy_file = args.input_file
iterations = args.iterations
if args.prompt is None:
    prompt = "anim"
else:
    prompt = args.prompt.replace(' ', '_')
data = np.load(npy_file, allow_pickle=True)
# Extract the motion component
motion_component = data#.item().get('motion')
# print(f'Loaded motion with shape {motion_component.shape}')
pos = motion_component
#pos = pos.transpose(0, 3, 1, 2) # samples x joints x coord x frames ==> samples x frames x joints x coord
print(f'Loaded motion with shape {pos.shape}')
parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
prompt_path = args.output_dir + "\\" + prompt
SMPL_JOINT_NAMES = [
    'Hips', # 0
    'LeftUpperLeg', # 1
    'RightUpperLeg', # 2
    'Spine', # 3
    'LeftLowerLeg', # 4
    'RightLowerLeg', # 5
    'Chest', # 6
    'LeftFoot', # 7
    'RightFoot', # 8
    'UpperChest', # 9
    'L_Foot', # 10
    'R_Foot', # 11
    'Neck', # 12
    'LeftShoulder', # 13
    'RightShoulder', # 14
    'Head', # 15
    'LeftUpperArm', # 16
    'RightUpperArm', # 17
    'LeftLowerArm', # 18
    'RightLowerArm', # 19
    'L_Wrist', # 20
    'R_Wrist', # 21
    # 'L_Hand', # 22
    # 'R_Hand', # 23
]
for i, p in enumerate(pos):
    print(f'starting anim no. {i}')
    anim, sorted_order, _ = animation_from_positions(p, parents, iterations=iterations)
    prompt_i = prompt_path + "_" + str(i) + ".bvh"
    BVH.save(prompt_i, anim, names=np.array(SMPL_JOINT_NAMES)[sorted_order])
    
    
# SMPL_JOINT_NAMES = [
# 'Pelvis', # 0
# 'L_Hip', # 1
# 'R_Hip', # 2
# 'Spine1', # 3
# 'L_Knee', # 4
# 'R_Knee', # 5
# 'Spine2', # 6
# 'L_Ankle', # 7
# 'R_Ankle', # 8
# 'Spine3', # 9
# 'L_Foot', # 10
# 'R_Foot', # 11
# 'Neck', # 12
# 'L_Collar', # 13
# 'R_Collar', # 14
# 'Head', # 15
# 'L_Shoulder', # 16
# 'R_Shoulder', # 17
# 'L_Elbow', # 18
# 'R_Elbow', # 19
# 'L_Wrist', # 20
# 'R_Wrist', # 21
# # 'L_Hand', # 22
# # 'R_Hand', # 23
# ]