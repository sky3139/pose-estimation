import cdflib
# import torch
# from models import VAE
import os
import pickle as pkl
import numpy as np
# from icecream import ic as print

from utils import *
import test_human
skeleton2tensor = test_human.human_skeleton2tensor
# Hyperparameters
data_path = "./data/human"
mode = 'train'







import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
# data=pd.read_csv('s_01_act_02_subact_01_ca_01.txt',sep =r' ',header=None) # 注意目录层级
# print(data)
file = open('s_01_act_02_subact_01_ca_03.txt')
line = file.readline()
frame = int(line)
line = file.readline().split(' ')[1:-1]
resolution = [int(i) for i in line]
print(resolution)
line = file.readline().split(' ')[1:-1]
rotation = np.array([float(i) for i in line]).reshape(3, 3)
# print(rotation)
line = file.readline().split(' ')[1:-1]
translation = [float(i) for i in line]

line = file.readline().split(' ')[1:-1]
focal_length = [float(i) for i in line]

line = file.readline().split(' ')[1:-1]
c_param = [float(i) for i in line]

line = file.readline().split(' ')[1:-1]
k_param = [float(i) for i in line]

line = file.readline().split(' ')[1:-1]
p_param = [float(i) for i in line]
line = file.readline()

cam_pos = np.column_stack((rotation, np.array(translation)*0.001))
cam_pos = np.row_stack((cam_pos, np.array([0, 0, 0, 1])))
cam_K = np.array([focal_length[0], 0, c_param[0],
                  0, focal_length[0], c_param[1], 0, 0, 1]).reshape(3,3)
print(cam_K)
D2_Pos=[]
D3_Pos=[]
pix_pos=[]
while 1:
    line = file.readline()
    if not line:
        break
    line = line.split(' ')[1:-1]
    pos = np.array([float(i) for i in line]).reshape(-1, 3)*0.001 #3D
    pos = np.insert(pos, 3, values=1, axis=1)
    # print(cam_pos,pos)
    cam_join_pos = np.dot(cam_pos, pos.T).T 
    cam_join_pos=np.delete(cam_join_pos,3,axis=1) #2D
    # print(cam_join_pos)
    gui_pos=[]
    for p in cam_join_pos:
        _p=p/p[1]
        temp=_p[2]
        _p[2]=_p[1]
        _p[1]=temp
        gui_pos.append(_p)
        # print(p,_p)
    # print(np.array(gui_pos))
    pix_pos=np.dot(cam_K,np.array(gui_pos).T).T
    cam_join_pos=np.delete(pix_pos,2,axis=1) #像素
    # print(len(cam_join_pos))

    break




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
  
cam_intrinsics=np.array([[1145.049405,0,512.541505 ],[0,1143.781096 ,515.451487],[0,0,1]])
extrinsics = np.array([[-0.915362, 0.051548, -0.399319, 1.841107028],
                       [0.401808, 0.180374, -0.897784, 4.955284623],
                       [0.025748, -0.982246, -0.185820, 1.563445396],
                       [0, 0, 0, 1]])
def jointPos2camPos(jointPositions):

    jointPositions = jointPositions.reshape(-1, 3)*0.001
    jointPositions = np.insert(jointPositions, 3, values=1, axis=1).T
    skeleton_t = np.dot(extrinsics, jointPositions).T
    skeleton_t = np.delete(skeleton_t, -1, axis=1)
    return skeleton_t
np.set_printoptions(suppress=True)

p2_pathh = "/home/u20/gitee/ahuman-pose-estimation/data"
for pkl_file in pkl_files:
    cdf = cdflib.CDF("data/human/train/Directions 1.cdf")
    p2_cdf = cdflib.CDF("data/D2_Positions/Directions 1.54138969.cdf")
# print(cdf.cdf_info())
    info = cdf.varget("Pose")
    p2_info = p2_cdf.varget("Pose")
    print(p2_info[0])
    # print(seq.keys())
    # print(seq['poses2d'][0].shape, seq['jointPositions'][0].shape)
    total_frames = len(info[0])
    print(info[0][0])
    for i, jointPositions in enumerate(info[0]):
        if(i > total_frames-5):
            break
        # for jointPositions, poses2d in zip(seq['jointPositions'], seq['poses2d']):
        poses2d = p2_info[0][i]
        # print(poses2d)
        skeleton_t=jointPos2camPos(jointPositions)
        x_t, r_t = skeleton2tensor(skeleton_t)
        # print(poses2d)
        uv_1 = poses2d[i+1]
        uv_2 = poses2d[i+2]
        uv_3 = poses2d[i+3]
        uv_4 = poses2d[i+4]
        uv_5 = poses2d[i+5]

        skeleton_1 = jointPos2camPos(info[0][i+1])
        skeleton_2 = jointPos2camPos(info[0][i+2])
        skeleton_3 = jointPos2camPos(info[0][i+3])
        skeleton_4 = jointPos2camPos(info[0][i+4])
        skeleton_5 = jointPos2camPos(info[0][i+5])
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

    #     break
    # break
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
