import torch
from models import VAE
import os
import pickle as pkl
import numpy as np
from icecream import ic as print

from utils import *

# Hyperparameters
# data_path = "./data/3DPW/sequenceFiles"
data_path = "data/human"
mode = 'test'

# Generate data
folder_path = os.path.join(data_path, mode)
pkl_files = [os.path.join(folder_path, pkl) for pkl in os.listdir(folder_path)]
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

for pkl_file in pkl_files:
    u = pkl._Unpickler(open(pkl_file,'rb'))
    u.encoding = 'latin1'
    seq = u.load()
    # print(seq.keys())
    # print(seq['poses2d'][0].shape, seq['jointPositions'][0].shape)
    for jointPositions, poses2d in zip(seq['jointPositions'], seq['poses2d']):
        print(jointPositions)
        total_frames = jointPositions.shape[0]
        jointPositions = jointPositions.reshape(total_frames, -1, 3)
        # print(jointPositions.shape)
        for t in range(total_frames-5):
            # frame t
            # 
            skeleton_t = jointPos2camPos(seq['cam_poses'][t], jointPositions[t])
            # uv_hat = seq['cam_intrinsics'] @ tensor2skeleton(*skeleton2tensor(skeleton_t)).T
            # uv_hat /= uv_hat[-1]
            # uv_hat = uv_hat[:2].T
            # print(uv_hat)
            # print(pose2uv(poses2d[t]))
            # print(uv_hat - pose2uv(poses2d[t]))
            x_t, r_t = skeleton2tensor(skeleton_t)
            uv_1 = pose2uv(poses2d[t+1])
            uv_2 = pose2uv(poses2d[t+2])
            uv_3 = pose2uv(poses2d[t+3])
            uv_4 = pose2uv(poses2d[t+4])
            uv_5 = pose2uv(poses2d[t+5])
            skeleton_1 = jointPos2camPos(seq['cam_poses'][t+1], jointPositions[t+1])
            skeleton_2 = jointPos2camPos(seq['cam_poses'][t+2], jointPositions[t+2])
            skeleton_3 = jointPos2camPos(seq['cam_poses'][t+3], jointPositions[t+3])
            skeleton_4 = jointPos2camPos(seq['cam_poses'][t+4], jointPositions[t+4])
            skeleton_5 = jointPos2camPos(seq['cam_poses'][t+5], jointPositions[t+5])
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
            intrinsics_list.append(seq['cam_intrinsics'])
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

# Save data   
x_list = np.stack(x_list, axis=0)
r_list = np.stack(r_list, axis=0)
intrinsics_list = np.stack(intrinsics_list, axis=0)
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
        'r_1': r_1_list, 'r_2': r_2_list, 'r_3': r_3_list, 'r_4': r_4_list, 'r_5': r_5_list,}
with open('./data/latent/latent-{}-multi.pkl'.format(mode), 'wb') as handle:
    pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)
