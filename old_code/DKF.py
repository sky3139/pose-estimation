import numpy as np
from numpy import sin, cos
from icecream import ic as print
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import time
import torch
from torch.autograd.functional import jacobian
from torch.distributions.multivariate_normal import MultivariateNormal

from models import VAE, Transition
from utils import *

# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparams
# pkl_file = './data/3DPW/sequenceFiles/train/courtyard_arguing_00.pkl'
pkl_file="data/data.pkl"
vae_checkpoint = "/home/u20/d2/code/pose-estimation/data/vae_best.pth"
vae_in_features = 25

trans_checkpoint = '/home/u20/d2/code/pose-estimation/data/transition_best.pth'
trans_in_features = 64
actor_id = 0
num_joints = 12
num_edges = 11
num_states = 64
num_z = num_joints*2
num_pos = num_joints*3
num_uv = num_joints*2
# l_noise = (0.2**2)*torch.eye(trans_in_features).to(device)
# pos_noise = (0.02**2)*torch.eye(num_pos).to(device)
uv_noise = (10**2)*torch.eye(num_uv).to(device)

####################################################
# load models
####################################################
# transition model
transition = Transition(in_features=trans_in_features).to(device)
transition.load_state_dict(torch.load(trans_checkpoint))
# vae model
vae = VAE(in_features=vae_in_features).to(device)
vae.load_state_dict(torch.load(vae_checkpoint))

####################################################
# parse data file
####################################################
u = pkl._Unpickler(open(pkl_file,'rb'))
u.encoding = 'latin1'
seq = u.load()
jointPositions = seq['jointPositions'][actor_id]
total_frames = jointPositions.shape[0]
jointPositions = jointPositions.reshape(total_frames, -1, 3)     # size: [T, 24, 3]
cam_poses = seq['cam_poses']                                     # size: [T, 4, 4]
poses2d = seq['poses2d'][actor_id]                               # size: [T, 3, 18]
intrinsics = seq['cam_intrinsics']                           # size: [3, 3]
intrinsics = torch.from_numpy(intrinsics.astype('float32')).to(device)
print(jointPositions.shape, cam_poses.shape, poses2d.shape, intrinsics.shape)

####################################################
# kalman filter
####################################################
# initialize state
skeleton_0 = jointPos2camPos(cam_poses[0], jointPositions[0])
x_0, r_0 = skeleton2tensor(skeleton_0)
pos_0 = tensor2skeleton(x_0, r_0)

x = x_0.to(device)
r = r_0.to(device)
with torch.no_grad():
    l_mu, l_logvar = vae.encode(x)
l_std = torch.exp(0.5 * l_logvar)
l_var = torch.diag_embed(l_std**2)
# l_mu is state & l_var is covariance matrix
print(l_mu.shape, l_var.shape)

line_x_all = []
line_y_all = []
line_z_all = []
line_x_all_gt = []
line_y_all_gt = []
line_z_all_gt = []
total_frames = 100
for t in range(1, total_frames):
    print(t)
    # observation
    uv_t = torch.from_numpy(pose2uv(poses2d[t]).astype('float32')).to(device)
    skeleton_gt = jointPos2camPos(cam_poses[t], jointPositions[t])
    x_gt, r_gt = skeleton2tensor(skeleton_gt)
    pos_gt = tensor2skeleton(x_gt, r_gt).to(device)
    uv_gt = intrinsics @ pos_gt.T
    uv_gt /= uv_gt[2,:].clone()
    uv_gt = uv_gt[:2].T
    # print(uv_gt - uv_t)
    # print(uv_t)
    # uv_t = uv_gt

    ####################################################
    # prediction
    ####################################################
    # l--->l t+1
    l_mu_hat, l_var_hat = transition(l_mu.unsqueeze(0), l_var.unsqueeze(0))
    # l_mu_hat, l_logvar_hat = vae.encode(x_gt.unsqueeze(0).to(device))
    # l_var_hat = torch.diag_embed(torch.exp(l_logvar_hat))
    l_var = l_var_hat.squeeze()
    # l--->x
    x_mu_hat, x_var_hat = vae.decode_distribution(l_mu_hat, l_var_hat)
    # x--->pos
    pos_mu_hat, pos_var_hat = x2pos_distribution(x_mu_hat, x_var_hat, r.unsqueeze(0))
    # pos--->uv
    uv_mu_hat, uv_var_hat = pos2uv_distribution(pos_mu_hat, pos_var_hat, intrinsics.unsqueeze(0))

    # sampling
    def l2uv(l):
        # l--->x
        x = vae.decode(l)
        # x--->pos
        pos = tensor2skeleton(x, r.unsqueeze(0))
        # pos--->uv
        uv = pos2uv(pos, intrinsics.unsqueeze(0))
        return torch.masked_select(uv, mask).reshape(-1, 2)#uv
    # l_distrib = MultivariateNormal(loc=l_mu_hat, covariance_matrix=l_var_hat)
    # l_hat = l_distrib.rsample()
    # uv_hat = l2uv(l_hat)

    ####################################################
    # chi-square test
    ####################################################
    # visible keypoints
    mask = (uv_t != 0)
    num_z = torch.sum(mask.int())
    uv_t = torch.masked_select(uv_t, mask).reshape(-1, 2)
    uv_mu_hat = l2uv(l_mu.unsqueeze(0))
    uv_noise = (10**2)*torch.eye(num_z).to(device)

    ####################################################
    # observation matrix
    ####################################################
    H = jacobian(l2uv, l_mu.unsqueeze(0), vectorize=True)
    H = H.squeeze().reshape(num_z, -1)
    # print(H.shape)
    
    ####################################################
    # update
    ####################################################
    K = l_var @ H.T @ torch.inverse(H @ l_var @ H.T + uv_noise)
    l_mu = l_mu_hat.squeeze() + (K @ (uv_t - uv_mu_hat.squeeze()).reshape(num_z, -1)).squeeze()
    l_var = l_var - K @ H @ l_var
    # print(l_mu.shape, l_var.shape)

    ####################################################
    # plot
    ####################################################
    # l--->x
    x_mu = vae.decode(l_mu)
    # x--->pos
    pos_mu = tensor2skeleton(x_mu, r).detach().cpu().numpy()
    x_0, y_0, z_0 = pos_mu[0]
    x_1, y_1, z_1 = pos_mu[1]
    x_2, y_2, z_2 = pos_mu[2]
    x_3, y_3, z_3 = pos_mu[3]
    x_4, y_4, z_4 = pos_mu[4]
    x_5, y_5, z_5 = pos_mu[5]
    x_6, y_6, z_6 = pos_mu[6]
    x_7, y_7, z_7 = pos_mu[7]
    x_8, y_8, z_8 = pos_mu[8]
    x_9, y_9, z_9 = pos_mu[9]
    x_10, y_10, z_10 = pos_mu[10]
    x_11, y_11, z_11 = pos_mu[11]
    line_x_all.append([[x_0, x_1], [(x_0+x_1)/2, x_2], [(x_0+x_1)/2, x_3], [x_0, x_4], [x_4, x_5], [x_1, x_6], [x_6, x_7], [x_2, x_8], [x_8, x_9], [x_3, x_10], [x_10, x_11]])
    line_y_all.append([[y_0, y_1], [(y_0+y_1)/2, y_2], [(y_0+y_1)/2, y_3], [y_0, y_4], [y_4, y_5], [y_1, y_6], [y_6, y_7], [y_2, y_8], [y_8, y_9], [y_3, y_10], [y_10, y_11]])
    line_z_all.append([[z_0, z_1], [(z_0+z_1)/2, z_2], [(z_0+z_1)/2, z_3], [z_0, z_4], [z_4, z_5], [z_1, z_6], [z_6, z_7], [z_2, z_8], [z_8, z_9], [z_3, z_10], [z_10, z_11]])
    
    # ground truth
    skeleton_gt = jointPos2camPos(cam_poses[t], jointPositions[t])
    x_gt, r_gt = skeleton2tensor(skeleton_gt)
    pos_gt = tensor2skeleton(x_gt, r_gt).cpu().numpy()
    x_0, y_0, z_0 = pos_gt[0]
    x_1, y_1, z_1 = pos_gt[1]
    x_2, y_2, z_2 = pos_gt[2]
    x_3, y_3, z_3 = pos_gt[3]
    x_4, y_4, z_4 = pos_gt[4]
    x_5, y_5, z_5 = pos_gt[5]
    x_6, y_6, z_6 = pos_gt[6]
    x_7, y_7, z_7 = pos_gt[7]
    x_8, y_8, z_8 = pos_gt[8]
    x_9, y_9, z_9 = pos_gt[9]
    x_10, y_10, z_10 = pos_gt[10]
    x_11, y_11, z_11 = pos_gt[11]
    line_x_all_gt.append([[x_0, x_1], [(x_0+x_1)/2, x_2], [(x_0+x_1)/2, x_3], [x_0, x_4], [x_4, x_5], [x_1, x_6], [x_6, x_7], [x_2, x_8], [x_8, x_9], [x_3, x_10], [x_10, x_11]])
    line_y_all_gt.append([[y_0, y_1], [(y_0+y_1)/2, y_2], [(y_0+y_1)/2, y_3], [y_0, y_4], [y_4, y_5], [y_1, y_6], [y_6, y_7], [y_2, y_8], [y_8, y_9], [y_3, y_10], [y_10, y_11]])
    line_z_all_gt.append([[z_0, z_1], [(z_0+z_1)/2, z_2], [(z_0+z_1)/2, z_3], [z_0, z_4], [z_4, z_5], [z_1, z_6], [z_6, z_7], [z_2, z_8], [z_8, z_9], [z_3, z_10], [z_10, z_11]])


def run(t):
    # left
    line_x_t = line_x_all_gt[t]
    line_y_t = line_y_all_gt[t]
    line_z_t = line_z_all_gt[t]
    for line, line_x, line_y, line_z in zip(lines_left, line_x_t, line_y_t, line_z_t):
        line.set_data(np.array([line_x, line_y]))
        line.set_3d_properties(np.array(line_z))
    # right
    line_x_t = line_x_all[t]
    line_y_t = line_y_all[t]
    line_z_t = line_z_all[t]
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
ax_left.view_init(elev=-20, azim=-90)
ax_left.set_xlim3d([-1, 1])
ax_left.set_xlabel('X')
ax_left.set_ylim3d([-1, 1])
ax_left.set_ylabel('Y')
ax_left.set_zlim3d([3, 5])
ax_left.set_zlabel('Z')
# right
ax_right = fig.add_subplot(1, 2, 2, projection='3d')
ax_right.set_title('Latent Kalman Filter')
ax_right.view_init(elev=-20, azim=-90)
ax_right.set_xlim3d([-1, 1])
ax_right.set_xlabel('X')
ax_right.set_ylim3d([-1, 1])
ax_right.set_ylabel('Y')
ax_right.set_zlim3d([3, 5])
ax_right.set_zlabel('Z')
# create animation
lines_left = [ax_left.plot([], [], [], 'royalblue', marker='o')[0] for i in range(11)]
lines_right = [ax_right.plot([], [], [], 'royalblue', marker='o')[0] for i in range(11)]
ani = animation.FuncAnimation(fig, run, np.arange(total_frames-1), interval=30)
FFwriter = animation.FFMpegWriter(fps=30)
ani.save('./saved/videos/DKF.mp4', writer=FFwriter)
# plt.show()
