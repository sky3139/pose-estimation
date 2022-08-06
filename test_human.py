import torch
from models import VAE
import os
import pickle as pkl
import numpy as np
from icecream import ic as print
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import *

# Hyperparameters
data_path = "./data/human"
model_path = "./data/"
mode = 'test'
ae_features = 25
def human_skeleton2tensor(skeleton_3d):
    # joint positions
    # x_0, y_0, z_0 = skeleton_3d[11]  # LShoulder
    # x_1, y_1, z_1 = skeleton_3d[14]  # RShoulder
    # x_2, y_2, z_2 = skeleton_3d[0]  # LHip
    # x_3, y_3, z_3 = skeleton_3d[1]  # RHip
    # x_4, y_4, z_4 = skeleton_3d[12]  # LElbow
    # x_5, y_5, z_5 = skeleton_3d[13]  # LWrist
    # x_6, y_6, z_6 = skeleton_3d[15]  # RElbow
    # x_7, y_7, z_7 = skeleton_3d[16]  # RWrist
    # x_8, y_8, z_8 = skeleton_3d[5]  # LKnee
    # x_9, y_9, z_9 = skeleton_3d[6]  # LAnkle
    # x_10, y_10, z_10 = skeleton_3d[2]  # Rknee
    # x_11, y_11, z_11 = skeleton_3d[3]  # RAnkle
    x_0, y_0, z_0 = skeleton_3d[17]  # LShoulder
    x_1, y_1, z_1 = skeleton_3d[25]  # RShoulder
    x_2, y_2, z_2 = skeleton_3d[0]  # LHip
    x_3, y_3, z_3 = skeleton_3d[1]  # RHip
    x_4, y_4, z_4 = skeleton_3d[18]  # LElbow
    x_5, y_5, z_5 = skeleton_3d[19]  # LWrist
    x_6, y_6, z_6 = skeleton_3d[26]  # RElbow
    x_7, y_7, z_7 = skeleton_3d[27]  # RWrist
    x_8, y_8, z_8 = skeleton_3d[7]  # LKnee
    x_9, y_9, z_9 = skeleton_3d[8]  # LAnkle
    x_10, y_10, z_10 = skeleton_3d[2]  # Rknee
    x_11, y_11, z_11 = skeleton_3d[3]  # RAnkle
    # convert to spherical representation
    r_1, theta_1, phi_1 = cartesian2spherical(
        np.array([x_1, y_1, z_1]) - np.array([x_0, y_0, z_0]))
    r_2, theta_2, phi_2 = cartesian2spherical(
        np.array([x_2, y_2, z_2]) - np.array([(x_0+x_1)/2, (y_0+y_1)/2, (z_0+z_1)/2]))
    r_3, theta_3, phi_3 = cartesian2spherical(
        np.array([x_3, y_3, z_3]) - np.array([x_2, y_2, z_2]))
    r_4, theta_4, phi_4 = cartesian2spherical(
        np.array([x_4, y_4, z_4]) - np.array([x_0, y_0, z_0]))
    r_5, theta_5, phi_5 = cartesian2spherical(
        np.array([x_5, y_5, z_5]) - np.array([x_4, y_4, z_4]))
    r_6, theta_6, phi_6 = cartesian2spherical(
        np.array([x_6, y_6, z_6]) - np.array([x_1, y_1, z_1]))
    r_7, theta_7, phi_7 = cartesian2spherical(
        np.array([x_7, y_7, z_7]) - np.array([x_6, y_6, z_6]))
    r_8, theta_8, phi_8 = cartesian2spherical(
        np.array([x_8, y_8, z_8]) - np.array([x_2, y_2, z_2]))
    r_9, theta_9, phi_9 = cartesian2spherical(
        np.array([x_9, y_9, z_9]) - np.array([x_8, y_8, z_8]))
    r_10, theta_10, phi_10 = cartesian2spherical(
        np.array([x_10, y_10, z_10]) - np.array([x_3, y_3, z_3]))
    r_11, theta_11, phi_11 = cartesian2spherical(
        np.array([x_11, y_11, z_11]) - np.array([x_10, y_10, z_10]))

    x = torch.Tensor([x_0, y_0, z_0, theta_1, phi_1, theta_2, phi_2, theta_3, phi_3, theta_4, phi_4, theta_5,
                        phi_5, theta_6, phi_6, theta_7, phi_7, theta_8, phi_8, theta_9, phi_9, theta_10, phi_10, theta_11, phi_11])
    r = torch.Tensor([r_1, r_2, r_3, r_4, r_5, r_6,
                        r_7, r_8, r_9, r_10, r_11])
    return x, r
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
vae = VAE(in_features=ae_features).to(device)
# Load checkpoint
vae.load_state_dict(torch.load(os.path.join(model_path, "ae_best.pth")))
vae.eval()
import cdflib
# Generate data
folder_path = os.path.join(data_path, mode)
pkl_files = [os.path.join(folder_path, pkl) for pkl in os.listdir(folder_path)]
x_list = []
y_list = []
for pkl_file in pkl_files:
    cdf = cdflib.CDF(pkl_file)
# print(cdf.cdf_info())
    info = cdf.varget("Pose")
    x_list = []
    
    for i, jointPositions in enumerate(info[0]):
        points = jointPositions.reshape(-1, 3)*0.001  # 世界坐标系的点
        x_t, r_t = human_skeleton2tensor(points)
        x_t = x_t.unsqueeze(0).to(device) 
        r_t = r_t.unsqueeze(0).to(device)
    #         
    #         
        # x_list.append(tensor2skeleton(*skeleton2tensor(skeleton_t)).cpu().numpy())
        x_list.append(tensor2skeleton(*human_skeleton2tensor(points)).cpu().numpy())
        with torch.no_grad():
            _, x_recon, _, _ = vae(x_t)
            # print(x_recon)
            y_list.append(tensor2skeleton(x_recon, r_t).squeeze(0).detach().cpu().numpy())                
#             # print(y_list[-1])
    #         # print(x_list, y_list)
            # break
    total_frames=len(x_list)
    break

x_list = np.array(x_list)
y_list = np.array(y_list)
# print(x_list.shape, y_list.shape)

def run(t):
    # print(t)
    # left
    x_0, y_0, z_0 = x_list[t, 0]
    x_1, y_1, z_1 = x_list[t, 1]
    x_2, y_2, z_2 = x_list[t, 2]
    x_3, y_3, z_3 = x_list[t, 3]
    x_4, y_4, z_4 = x_list[t, 4]
    x_5, y_5, z_5 = x_list[t, 5]
    x_6, y_6, z_6 = x_list[t, 6]
    x_7, y_7, z_7 = x_list[t, 7]
    x_8, y_8, z_8 = x_list[t, 8]
    x_9, y_9, z_9 = x_list[t, 9]
    x_10, y_10, z_10 = x_list[t, 10]
    x_11, y_11, z_11 = x_list[t, 11]
    line_x_t = [[x_0, x_1], [(x_0+x_1)/2, x_2], [(x_0+x_1)/2, x_3], [x_0, x_4], [x_4, x_5], [x_1, x_6], [x_6, x_7], [x_2, x_8], [x_8, x_9], [x_3, x_10], [x_10, x_11]]
    line_y_t = [[y_0, y_1], [(y_0+y_1)/2, y_2], [(y_0+y_1)/2, y_3], [y_0, y_4], [y_4, y_5], [y_1, y_6], [y_6, y_7], [y_2, y_8], [y_8, y_9], [y_3, y_10], [y_10, y_11]]
    line_z_t = [[z_0, z_1], [(z_0+z_1)/2, z_2], [(z_0+z_1)/2, z_3], [z_0, z_4], [z_4, z_5], [z_1, z_6], [z_6, z_7], [z_2, z_8], [z_8, z_9], [z_3, z_10], [z_10, z_11]]
    for line, line_x, line_y, line_z in zip(lines_left, line_x_t, line_y_t, line_z_t):
        line.set_data(np.array([line_x, line_y]))
        line.set_3d_properties(np.array(line_z))
    # time.sleep(0.1)
    # right
    x_0, y_0, z_0 = y_list[t, 0]
    x_1, y_1, z_1 = y_list[t, 1]
    x_2, y_2, z_2 = y_list[t, 2]
    x_3, y_3, z_3 = y_list[t, 3]
    x_4, y_4, z_4 = y_list[t, 4]
    x_5, y_5, z_5 = y_list[t, 5]
    x_6, y_6, z_6 = y_list[t, 6]
    x_7, y_7, z_7 = y_list[t, 7]
    x_8, y_8, z_8 = y_list[t, 8]
    x_9, y_9, z_9 = y_list[t, 9]
    x_10, y_10, z_10 = y_list[t, 10]
    x_11, y_11, z_11 = y_list[t, 11]
    line_x_t = [[x_0, x_1], [(x_0+x_1)/2, x_2], [(x_0+x_1)/2, x_3], [x_0, x_4], [x_4, x_5], [x_1, x_6], [x_6, x_7], [x_2, x_8], [x_8, x_9], [x_3, x_10], [x_10, x_11]]
    line_y_t = [[y_0, y_1], [(y_0+y_1)/2, y_2], [(y_0+y_1)/2, y_3], [y_0, y_4], [y_4, y_5], [y_1, y_6], [y_6, y_7], [y_2, y_8], [y_8, y_9], [y_3, y_10], [y_10, y_11]]
    line_z_t = [[z_0, z_1], [(z_0+z_1)/2, z_2], [(z_0+z_1)/2, z_3], [z_0, z_4], [z_4, z_5], [z_1, z_6], [z_6, z_7], [z_2, z_8], [z_8, z_9], [z_3, z_10], [z_10, z_11]]
    for line, line_x, line_y, line_z in zip(lines_right, line_x_t, line_y_t, line_z_t):
        line.set_data(np.array([line_x, line_y]))
        line.set_3d_properties(np.array(line_z))
    lines = lines_left + lines_right
    return lines
# attach 3D axis to figure
# left
fig = plt.figure(figsize=plt.figaspect(0.5))
ax_left = fig.add_subplot(1, 2, 1, projection='3d')
ax_left.set_title('Ground Truth')
ax_left.view_init(elev=-90, azim=-90)
ax_left.set_xlim3d([-1, 1])
ax_left.set_xlabel('X')
ax_left.set_ylim3d([-1, 1])
ax_left.set_ylabel('Y')
ax_left.set_zlim3d([3, 5])
ax_left.set_zlabel('Z')
# right
ax_right = fig.add_subplot(1, 2, 2, projection='3d')
ax_right.set_title('VAE Reconstruction')
ax_right.view_init(elev=-90, azim=-90)
ax_right.set_xlim3d([-1, 1])
ax_right.set_xlabel('X')
ax_right.set_ylim3d([-1, 1])
ax_right.set_ylabel('Y')
ax_right.set_zlim3d([3, 5])
ax_right.set_zlabel('Z')
# create animation
lines_left = [ax_left.plot([], [], [], 'royalblue', marker='o')[0] for i in range(11)]
lines_right = [ax_right.plot([], [], [], 'royalblue', marker='o')[0] for i in range(11)]
ani = animation.FuncAnimation(fig, run, np.arange(total_frames), interval=200)
FFwriter = animation.FFMpegWriter(fps=5)
ani.save('./saved/vae_reconstruction.mp4', writer=FFwriter)
# plt.show()