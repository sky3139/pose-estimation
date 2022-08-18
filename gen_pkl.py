# import cdflib
# import torch
# from models import VAE
import enum
import os
import pickle as pkl
import numpy as np
# from icecream import ic as print
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


# Hyperparameters
data_path = "./data/human"
mode = 'test'
if mode == 'train':
    train_list = ["S1", "S5", "S6", "S7", "S8"]
else:
    train_list = ["S9", "S11"]
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
    ans.append(skeleton_3d[16])  # LShoulder
    ans.append(skeleton_3d[17])  # RShoulder
    ans.append(skeleton_3d[1])  # LHip
    ans.append(skeleton_3d[2])  # RHip
    ans.append(skeleton_3d[18])  # LElbow
    ans.append(skeleton_3d[20])  # LWrist
    ans.append(skeleton_3d[19])  # RElbow
    ans.append(skeleton_3d[21])  # RWrist
    ans.append(skeleton_3d[4])  # LKnee
    ans.append(skeleton_3d[7])  # LAnkle
    ans.append(skeleton_3d[5])  # Rknee
    ans.append(skeleton_3d[8])  # RAnkle
    return np.array(ans)


data_3d = np.load(
    "/home/u20/d2/code/PoseFormer/data/data_3d_h36m.npz", allow_pickle=True)
# data_2d = np.load(
#     "/home/u20/d2/code/VideoPose3D/data/data_2d_h36m_gt.npz", allow_pickle=True)
# keypoints_metadata = data_2d['metadata'].item()
# keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
# cam_para = keypoints_metadata['cam']
# print(cam_para)
# kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])

# joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
# for v in p_i['cam']:
#     print(v)
# print(data)
data3d_item = data_3d['positions_3d'].item()

cam_name = []
cams_K=[]
for v in data3d_item["cam"]:
    cam_name.append(v['id'])
    K = np.array([v["focal_length"][0], 0, v["center"][0], 0,
                 v["focal_length"][1], v["center"][1], 0, 0, 1]).reshape(3, 3)
    cams_K.append(K)
data=dict()
# print(cam_name,cams_K)
for subject in train_list:
    print(mode, subject)
    for aid,action in enumerate( data3d_item[subject]):
        # for cam_idx, kps in enumerate(keypoints[subject][action][0]):
        #     print(cam_idx,len(kps))
        print(mode, subject,action)
        # print(cam_para[subject][0]["intrinsic"])
        cam_intrinsics = np.array(cams_K[0]).reshape(3, 3)
        # print(cam_intrinsics)
        p3s,p2s_4cam=data3d_item[subject][action]
        p2s=p2s_4cam[cam_name[0]]
        

        # for frame, skeleton_t in enumerate(p3s):
            # x_t, r_t = skeleton2tensor(skeleton_t)
            # print(len(p3s[frame]),len(p2s[frame]))
        break
    break
# with open("data/data.pkl", "wb") as f:
#     pkl.dump(data, f)





