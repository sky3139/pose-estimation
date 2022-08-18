import imp
import os
import enum
import numpy as np
from numpy import sin, cos
from icecream import ic as print
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import torch
from torch.autograd.functional import jacobian

from models import VAE, Transition
from utils import *


def skeleton2tensor(skeleton_3d):
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
    x_2, y_2, z_2 = skeleton_3d[6]  # LHip
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
        np.array([x_2, y_2, z_2]) - np.array([(x_0 + x_1) / 2, (y_0 + y_1) / 2, (z_0 + z_1) / 2]))
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
                      phi_5, theta_6, phi_6, theta_7, phi_7, theta_8, phi_8, theta_9, phi_9, theta_10, phi_10, theta_11,
                      phi_11])
    r = torch.Tensor([r_1, r_2, r_3, r_4, r_5, r_6,
                      r_7, r_8, r_9, r_10, r_11])
    return x, r


# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparams
# pkl_file = './data/3DPW/sequenceFiles/train/courtyard_arguing_00.pkl'
# pkl_file = "data/data.pkl"
# vae_checkpoint = "./saved/models/vae_best.pth"
vae_in_features = 25

# trans_checkpoint = './saved/models/transition_best.pth'

vae_checkpoint = "/home/u20/d2/code/pose-estimation/saved/models/vae-noise/ae_best.pth"
# vae_in_features = 25

trans_checkpoint = '/home/u20/d2/code/pose-estimation/data/transition_best.pth'

trans_in_features = 64
actor_id = 0
num_joints = 12
num_edges = 11
num_states = 64
num_z = num_joints * 2
num_pos = num_joints * 3
num_uv = num_joints * 2
# l_noise = (0.2**2)*torch.eye(trans_in_features).to(device)
# pos_noise = (0.02**2)*torch.eye(num_pos).to(device)
uv_noise = (10 ** 2) * torch.eye(num_uv).to(device)

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
# u = pkl._Unpickler(open(pkl_file, 'rb'))
# u.encoding = 'latin1'
# seq = u.load()
# jointPositions = seq['jointPositions'][actor_id]
# total_frames = jointPositions.shape[0]
# jointPositions = jointPositions.reshape(
# total_frames, -1, 3)     # size: [T, 24, 3]
# size: [T, 4, 4]
# cam_poses = seq['cam_poses']
# size: [T, 3, 18]
# poses2d = seq['poses2d'][actor_id]

####################################################
# kalman filter
####################################################
# initialize state
data_3d = np.load("/home/u20/d2/code/PoseFormer_bak/data/data_3d_h36m.npz", allow_pickle=True)
# data_3d = np.load("./data/data_3d_h36m.npz", allow_pickle=True)
data3d_item = data_3d['positions_3d'].item()

# Hyperparameters
data_path = "./data"
mode = 'test'
if mode == 'train':
    train_list = ["S1", "S5", "S6", "S7", "S8"]
else:
    train_list = ["S9", "S11"]

line_x_all = []
line_y_all = []
line_z_all = []
line_x_all_gt = []
line_y_all_gt = []
line_z_all_gt = []
total_frames = 100

np.set_printoptions(suppress=True)

folder_path = "data/human/test"
x_list = []
r_list = []
intrinsics_list = []
uv_1_list = []
uv_2_list = []
uv_3_list = []
uv_4_list = []
uv_5_list = []
pos_1_list = []
pos_2_list = []
pos_3_list = []
pos_4_list = []
pos_5_list = []
x_1_list = []
x_2_list = []
x_3_list = []
x_4_list = []
x_5_list = []
r_1_list = []
r_2_list = []
r_3_list = []
r_4_list = []
r_5_list = []


def get12(skeleton_3d):
    ans = []
    ans.append(skeleton_3d[17])  # LShoulder
    ans.append(skeleton_3d[25])  # RShoulder
    ans.append(skeleton_3d[6])  # LHip
    ans.append(skeleton_3d[1])  # RHip
    ans.append(skeleton_3d[18])  # LElbow
    ans.append(skeleton_3d[19])  # LWrist
    ans.append(skeleton_3d[26])  # RElbow
    ans.append(skeleton_3d[27])  # RWrist
    ans.append(skeleton_3d[7])  # LKnee
    ans.append(skeleton_3d[8])  # LAnkle
    ans.append(skeleton_3d[2])  # Rknee
    ans.append(skeleton_3d[3])  # RAnkle
    return np.array(ans)
def quaternion_to_rotation_matrix(quat,t):
    q = quat.copy()
    n = np.dot(q, q)
    if n < np.finfo(q.dtype).eps:
        return np.identity(4)
    q = q * np.sqrt(2.0 / n)
    q = np.outer(q, q)
    rot_matrix = np.array(
        [[1.0 - q[2, 2] - q[3, 3], q[1, 2] + q[3, 0], q[1, 3] - q[2, 0], t[0]],
        [q[1, 2] - q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] + q[1, 0], t[1]],
        [q[1, 3] + q[2, 0], q[2, 3] - q[1, 0], 1.0 - q[1, 1] - q[2, 2], t[2]],
        [0.0, 0.0, 0.0, 1.0]],
        dtype=q.dtype)
    return rot_matrix
cam_1R=quaternion_to_rotation_matrix(np.array([0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]),[1.8411070556640625, 4.95528466796875, 1.5634454345703125])

def word2cam(wordp3):
    g_p3=np.insert(wordp3,3,1,axis=1).T
    cam_p3s=np.dot(cam_1R,g_p3).T
    skeleton_t=np.delete(cam_p3s,3,axis=1)
    return skeleton_t
cam_name = []
cams_K = []
for v in data3d_item["cam"]:
    cam_name.append(v['id'])
    K = np.array([v["focal_length"][0], 0, v["center"][0], 0,
                  v["focal_length"][1], v["center"][1], 0, 0, 1]).reshape(3, 3)
    cams_K.append(K)
data = dict()
intrinsics = intrinsics = torch.from_numpy(cams_K[0].astype('float32').reshape(3, 3)).to(device)
# print(cam_name,cams_K)
for subject in train_list:
    print(mode, subject)
    for aid, action in enumerate(data3d_item[subject]):
        # for cam_idx, kps in enumerate(keypoints[subject][action][0]):
        #     print(cam_idx,len(kps))
        print(mode, subject, action)
        # print(cam_para[subject][0]["intrinsic"])
        cam_intrinsics = np.array(cams_K[0]).reshape(3, 3)
        # print(cam_intrinsics)
        p3s, p2s_4cam = data3d_item[subject][action]
        p2s = p2s_4cam[cam_name[0]]

        for t, skeleton_gt_w in enumerate(p3s):
            # print(t)
            if (t == 0):
                skeleton_0 = word2cam(skeleton_gt_w)
                x_0, r_0 = skeleton2tensor(skeleton_0)
                pos_0 = tensor2skeleton(x_0, r_0)

                x = x_0.to(device)
                r = r_0.to(device)
                with torch.no_grad():
                    l_mu, l_logvar = vae.encode(x)
                l_std = torch.exp(0.5 * l_logvar)
                l_var = torch.diag_embed(l_std ** 2)
                # l_mu is state & l_var is covariance matrix
                print(l_mu.shape, l_var.shape)
            # x_t, r_t = skeleton2tensor(skeleton_t)
            # print(len(p3s[frame]),len(p2s[frame]))
            #     break
            # break

            # for t in range(1, total_frames):

            # observation
            uv_t = torch.from_numpy(get12(p2s[t]).astype('float32')).to(device)
            # skeleton_gt = jointPos2camPos(cam_poses[t], jointPositions[t])
            x_gt, r_gt = skeleton2tensor(word2cam(skeleton_gt_w))
            pos_gt = tensor2skeleton(x_gt, r_gt).to(device)
            uv_gt = intrinsics @ pos_gt.T
            uv_gt /= uv_gt[2, :].clone()
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
            pos_mu_hat, pos_var_hat = x2pos_distribution(
                x_mu_hat, x_var_hat, r.unsqueeze(0))
            # print(pos_mu_hat)
            # pos--->uv
            uv_mu_hat, uv_var_hat = pos2uv_distribution(
                pos_mu_hat, pos_var_hat, intrinsics.unsqueeze(0))


            # sampling
            def l2uv(l):
                # l--->x
                x = vae.decode(l)
                # x--->pos
                pos = tensor2skeleton(x, r.unsqueeze(0))
                # pos--->uv
                uv = pos2uv(pos, intrinsics.unsqueeze(0))
                return torch.masked_select(uv, mask).reshape(-1, 2)  # uv


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
            uv_noise = (10 ** 2) * torch.eye(num_z).to(device)

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
            l_mu = l_mu_hat.squeeze() + (K @ (uv_t - uv_mu_hat.squeeze()
                                              ).reshape(num_z, -1)).squeeze()
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
            # print('our', x_0, y_0, z_0)
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
            line_x_all.append([[x_0, x_1], [(x_0 + x_1) / 2, x_2], [(x_0 + x_1) / 2, x_3], [x_0, x_4],
                               [x_4, x_5], [x_1, x_6], [x_6, x_7], [x_2, x_8], [x_8, x_9], [x_3, x_10], [x_10, x_11]])
            line_y_all.append([[y_0, y_1], [(y_0 + y_1) / 2, y_2], [(y_0 + y_1) / 2, y_3], [y_0, y_4],
                               [y_4, y_5], [y_1, y_6], [y_6, y_7], [y_2, y_8], [y_8, y_9], [y_3, y_10], [y_10, y_11]])
            line_z_all.append([[z_0, z_1], [(z_0 + z_1) / 2, z_2], [(z_0 + z_1) / 2, z_3], [z_0, z_4],
                               [z_4, z_5], [z_1, z_6], [z_6, z_7], [z_2, z_8], [z_8, z_9], [z_3, z_10], [z_10, z_11]])

            # ground truth

            pos_gt = pos_gt.cpu().numpy()
            x_0, y_0, z_0 = pos_gt[0]
            # print('gt', x_0, y_0, z_0)
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
            line_x_all_gt.append([[x_0, x_1], [(x_0 + x_1) / 2, x_2], [(x_0 + x_1) / 2, x_3], [x_0, x_4], [
                x_4, x_5], [x_1, x_6], [x_6, x_7], [x_2, x_8], [x_8, x_9], [x_3, x_10], [x_10, x_11]])
            line_y_all_gt.append([[y_0, y_1], [(y_0 + y_1) / 2, y_2], [(y_0 + y_1) / 2, y_3], [y_0, y_4], [
                y_4, y_5], [y_1, y_6], [y_6, y_7], [y_2, y_8], [y_8, y_9], [y_3, y_10], [y_10, y_11]])
            line_z_all_gt.append([[z_0, z_1], [(z_0 + z_1) / 2, z_2], [(z_0 + z_1) / 2, z_3], [z_0, z_4], [
                z_4, z_5], [z_1, z_6], [z_6, z_7], [z_2, z_8], [z_8, z_9], [z_3, z_10], [z_10, z_11]])
        break
    break


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
ax_left.view_init(elev=0, azim=45)
ax_left.set_xlim3d([-2, 2])
ax_left.set_xlabel('X')
ax_left.set_ylim3d([-1, 3])
ax_left.set_ylabel('Y')
ax_left.set_zlim3d([0, 3])
ax_left.set_zlabel('Z')
# right
ax_right = fig.add_subplot(1, 2, 2, projection='3d')
ax_right.set_title('Latent Kalman Filter')
ax_right.view_init(elev=0, azim=45)
ax_right.set_xlim3d([-5, 5])
ax_right.set_xlabel('X')
ax_right.set_ylim3d([-5, 5])
ax_right.set_ylabel('Y')
ax_right.set_zlim3d([0, 5])
ax_right.set_zlabel('Z')
# create animation
lines_left = [ax_left.plot([], [], [], 'royalblue', marker='o')[
                  0] for i in range(11)]
lines_right = [ax_right.plot([], [], [], 'royalblue', marker='o')[
                   0] for i in range(11)]
# import io
# import PIL,cv2
# for i in range(30):


#     run(i);
#     buffer = io.BytesIO()#using buffer,great way!
# #把plt的内容保存在内存中
#     plt.savefig(buffer,format = 'png')
#     #用PIL或CV2从内存中读取
#     dataPIL = PIL.Image.open(buffer)
#     #转换为nparrary
#     data = np.asarray(dataPIL)
#     cv2.imshow('image', data)
#     cv2.waitKey(1)
ani = animation.FuncAnimation(fig, run, np.arange(len(line_x_all_gt) - 1), interval=30)
FFwriter = animation.FFMpegWriter(fps=30)
ani.save('./save/videos/DKF.mp4', writer=FFwriter)