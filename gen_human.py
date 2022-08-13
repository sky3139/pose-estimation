# import cdflib
# import torch
# from models import VAE
import os
import pickle as pkl
import numpy as np
# from icecream import ic as print
from utils import *
def skeleton2tensor( skeleton_3d):
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
mode = 'train'
np.set_printoptions(suppress=True)

folder_path = "data/human/test"
# pkl_files = [os.path.join(folder_path, pkl) for pkl in os.listdir(folder_path)]
# # print(pkl_files)
# frame_datas = []

# for file_path in pkl_files:
#     print(file_path)

#     u = pkl._Unpickler(open(file_path, 'rb'))
#     u.encoding = 'latin1'
#     seq = u.load()
#     x_list = []
#     ps3d=seq['jointPositions']
#     ps2d=seq['poses2d']
#     cam=seq['CAM_P3D']
#     cam_intrinsics=seq['cam_intrinsics']
#     frame_data=[]
#     for i,skeleton_3d in enumerate(cam):
#         frame_data.append([ps3d[i], cam[i], ps2d[i]])
#     frame_datas.append(frame_data)
# Generate data
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





import numpy as np


data_3d=np.load("/home/u20/d2/code/VideoPose3D/data/data_3d_h36m.npz",allow_pickle=True)
data_2d=np.load("/home/u20/d2/code/VideoPose3D/data/data_2d_h36m_gt.npz",allow_pickle=True)
keypoints_metadata = data_2d['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
cam_para=keypoints_metadata['cam']
# print(cam_para)
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
# joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
keypoints = data_2d['positions_2d'].item()

# print(data)
data3d_item=data_3d['positions_3d'].item()
for subject in data3d_item:
    print(subject)
    for action in data3d_item[subject]:
        # p3=
        # for cam_idx, kps in enumerate(keypoints[subject][action][0]):
        #     print(cam_idx,len(kps))
        for frame,p3 in enumerate( data3d_item[subject][action]):
            if frame > len(data3d_item[subject][action])-7:
                continue
            skeleton_t=keypoints[subject][action][0][0][frame] #cam 3d
            cam_intrinsics=cam_para[subject]
            pix2d=keypoints[subject][action][1][0][frame] #pix 2d
            # print(len(cam3d),len(pix2d))

            x_t, r_t = skeleton2tensor(skeleton_t)
            # print(poses2d)
            uv_1 = get12(keypoints[subject][action][1][0][frame+1])
            uv_2 = get12(keypoints[subject][action][1][0][frame+2])
            uv_3 = get12(keypoints[subject][action][1][0][frame+3])
            uv_4 = get12(keypoints[subject][action][1][0][frame+4])
            uv_5 = get12(keypoints[subject][action][1][0][frame+5])
            
            skeleton_1 = keypoints[subject][action][0][0][frame+1]
            skeleton_2 = keypoints[subject][action][0][0][frame+2]
            skeleton_3 = keypoints[subject][action][0][0][frame+3]
            skeleton_4 = keypoints[subject][action][0][0][frame+4]
            skeleton_5 = keypoints[subject][action][0][0][frame+5]
            # print(len(skeleton_1),skeleton_1)
            pos_1 = tensor2skeleton(*skeleton2tensor(skeleton_1))
            pos_2 = tensor2skeleton(*skeleton2tensor(skeleton_2))
            pos_3 = tensor2skeleton(*skeleton2tensor(skeleton_3))
            pos_4 = tensor2skeleton(*skeleton2tensor(skeleton_4))
            pos_5 = tensor2skeleton(*skeleton2tensor(skeleton_5))
            x_1, r_1 = skeleton2tensor(skeleton_1)
            x_2, r_2 = skeleton2tensor(skeleton_2)
            x_3, r_3 = skeleton2tensor(skeleton_3)
            x_4, r_4 = skeleton2tensor(skeleton_4)
            x_5, r_5 = skeleton2tensor(skeleton_5)
            if (uv_1 == 0).any() or (uv_2 == 0).any() or (uv_3 == 0).any() or (uv_4 == 0).any() or (uv_5 == 0).any():
                continue

            x_list.append(x_t.cpu().numpy())
            r_list.append(r_t)
            intrinsics_list.append(cam_intrinsics)
            uv_1_list.append(uv_1)
            uv_2_list.append(uv_2)
            uv_3_list.append(uv_3)
            uv_4_list.append(uv_4)
            uv_5_list.append(uv_5)
            pos_1_list.append(pos_1)
            pos_2_list.append(pos_2)
            pos_3_list.append(pos_3)
            pos_4_list.append(pos_4)
            pos_5_list.append(pos_5)
            x_1_list.append(x_1)
            x_2_list.append(x_2)
            x_3_list.append(x_3)
            x_4_list.append(x_4)
            x_5_list.append(x_5)
            r_1_list.append(r_1)
            r_2_list.append(r_2)
            r_3_list.append(r_3)
            r_4_list.append(r_4)
            r_5_list.append(r_5)
        # print(x_1_list)
#     break
# break
# Save data
x_list = np.stack(x_list, axis=0)
r_list = np.stack(r_list, axis=0)
intrinsics_list = np.stack(intrinsics_list, axis=0)
# print(intrinsics_list)
uv_1_list = np.stack(uv_1_list, axis=0)
uv_2_list = np.stack(uv_2_list, axis=0)
uv_3_list = np.stack(uv_3_list, axis=0)
uv_4_list = np.stack(uv_4_list, axis=0)
uv_5_list = np.stack(uv_5_list, axis=0)
pos_1_list = np.stack(pos_1_list, axis=0)
pos_2_list = np.stack(pos_2_list, axis=0)
pos_3_list = np.stack(pos_3_list, axis=0)
pos_4_list = np.stack(pos_4_list, axis=0)
pos_5_list = np.stack(pos_5_list, axis=0)
x_1_list = np.stack(x_1_list, axis=0)
x_2_list = np.stack(x_2_list, axis=0)
x_3_list = np.stack(x_3_list, axis=0)
x_4_list = np.stack(x_4_list, axis=0)
x_5_list = np.stack(x_5_list, axis=0)
r_1_list = np.stack(r_1_list, axis=0)
r_2_list = np.stack(r_2_list, axis=0)
r_3_list = np.stack(r_3_list, axis=0)
r_4_list = np.stack(r_4_list, axis=0)
r_5_list = np.stack(r_5_list, axis=0)
print(x_list.shape, r_list.shape, intrinsics_list.shape,
      uv_1_list.shape, uv_2_list.shape, uv_3_list.shape, uv_4_list.shape, uv_5_list.shape,
      pos_1_list.shape, pos_2_list.shape, pos_3_list.shape, pos_4_list.shape, pos_5_list.shape,
      x_1_list.shape, x_2_list.shape, x_3_list.shape, x_4_list.shape, x_5_list.shape,
      r_1_list.shape, r_2_list.shape, r_3_list.shape, r_4_list.shape, r_5_list.shape)
data = {'x': x_list, 'r': r_list, 'intrinsics': intrinsics_list,
        'uv_1': uv_1_list, 'uv_2': uv_2_list, 'uv_3': uv_3_list, 'uv_4': uv_4_list, 'uv_5': uv_5_list,
        'pos_1': pos_1_list, 'pos_2': pos_2_list, 'pos_3': pos_3_list, 'pos_4': pos_4_list, 'pos_5': pos_5_list,
        'x_1': x_1_list, 'x_2': x_2_list, 'x_3': x_3_list, 'x_4': x_4_list, 'x_5': x_5_list,
        'r_1': r_1_list, 'r_2': r_2_list, 'r_3': r_3_list, 'r_4': r_4_list, 'r_5': r_5_list, }
with open('./data/human/latent-{}-multi.pkl'.format("train1"), 'wb') as handle:
    pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)
with open('./data/human/latent-{}-multi.pkl'.format("test1"), 'wb') as handle:
    pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)
with open('./data/human/latent-{}-multi.pkl'.format("validation1"), 'wb') as handle:
    pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)
