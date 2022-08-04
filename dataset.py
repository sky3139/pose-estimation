import imp
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import pickle as pkl
import numpy as np
import cdflib 
from utils import *


class Dataset_AE(Dataset):
    def __init__(self, data_path, mode='train', transform=None):
        super(Dataset_AE, self).__init__()
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        self.folder_path = os.path.join(self.data_path, self.mode)
        self.pkl_files = [os.path.join(self.folder_path, pkl)
                          for pkl in os.listdir(self.folder_path)]
        self.x_list = []
        for pkl_file in self.pkl_files:
            x_list = self.read_pickle(pkl_file)
            self.x_list.extend(x_list)
        self.total_frames = len(self.x_list)

    def read_pickle(self, pkl_file):
        u = pkl._Unpickler(open(pkl_file, 'rb'))
        u.encoding = 'latin1'
        seq = u.load()
        x_list = []
        for jointPositions in seq['jointPositions']:
            total_frames = jointPositions.shape[0]
            print(jointPositions, jointPositions.shape[0])

            jointPositions = jointPositions.reshape(total_frames, -1, 3)
            # print(jointPositions.shape)
            for t in range(total_frames):
                skeleton_3d = jointPos2camPos(
                    seq['cam_poses'][t], jointPositions[t])
                x, _ = skeleton2tensor(skeleton_3d)
                # print(seq['cam_poses'][t],"and",jointPositions[t],"\n")
                # transform
                print(len(skeleton_3d[0]))
                if self.transform is not None:
                    x = self.transform(x)

                x_list.append(x)

        return x_list

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        random_noise = torch.zeros_like(self.x_list[idx])
        random_noise[:3] = torch.rand(3)*20 - 10
        return self.x_list[idx] + random_noise


class Dataset_HUMAN(Dataset):
    def __init__(self, data_path, mode='train', transform=None):
        super(Dataset_HUMAN, self).__init__()
        self.mode = mode
        self.transform = transform
        self.folder_path = os.path.join(data_path, self.mode)
        self.pkl_files = [os.path.join(self.folder_path, pkl)
                          for pkl in os.listdir(self.folder_path)]
        self.x_list = []
        for pkl_file in self.pkl_files:
            print(pkl_file)
            x_list = self.read_pickle(pkl_file)
            self.x_list.extend(x_list)
            # break
        self.total_frames = len(self.x_list)

    def read_pickle(self, pkl_file):

        cdf = cdflib.CDF(pkl_file)
# print(cdf.cdf_info())
        info = cdf.varget("Pose")

        # u = pkl._Unpickler(open(pkl_file,'rb'))
        # u.encoding = 'latin1'
        fr = open(pkl_file, 'rb')  # open的参数是pkl文件的路径
        x_list = []
        for i, jointPositions in enumerate(info[0]):
            points = jointPositions.reshape(-1, 3)*0.001  # 世界坐标系的点
            # print( points)

            # total_frames = int(jointPositions.shape[0]/3)
            # skeleton_3d = jointPositions.reshape(-1, 3)
            # print(skeleton_3d[0])
            # for t in range(total_frames):
            #     # print(seq['cam_poses'][t],jointPositions[t])
            #     skeleton_3d = jointPos2camPos(seq['cam_poses'][t], jointPositions[t])
            x, _ = self.skeleton2tensor(points)

            #     # transform
            #     if self.transform is not None:
            #         x = self.transform(x)

            x_list.append(x)

        return x_list
#reference https://blog.csdn.net/weixin_45436729/article/details/124770186
    def skeleton2tensor(self, skeleton_3d):
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

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        random_noise = torch.zeros_like(self.x_list[idx])
        random_noise[:3] = torch.rand(3)*20 - 10
        return self.x_list[idx] + random_noise


class Dataset_Transition(Dataset):
    def __init__(self, data_path, mode='train'):
        super(Dataset_Transition, self).__init__()
        self.data_path = data_path
        self.mode = mode
        self.pkl_file = os.path.join(
            self.data_path, 'latent-{}-multi.pkl'.format(mode))
        with open(self.pkl_file, 'rb') as handle:
            data = pkl.load(handle)
        self.x_list = torch.from_numpy(data['x'].astype('float32'))
        self.r_list = torch.from_numpy(data['r'].astype('float32'))
        self.intrinsics_list = torch.from_numpy(
            data['intrinsics'].astype('float32'))
        self.uv_1_list = torch.from_numpy(data['uv_1'].astype('float32'))
        self.uv_2_list = torch.from_numpy(data['uv_2'].astype('float32'))
        self.uv_3_list = torch.from_numpy(data['uv_3'].astype('float32'))
        self.uv_4_list = torch.from_numpy(data['uv_4'].astype('float32'))
        self.uv_5_list = torch.from_numpy(data['uv_5'].astype('float32'))
        self.pos_1_list = torch.from_numpy(data['pos_1'].astype('float32'))
        self.pos_2_list = torch.from_numpy(data['pos_2'].astype('float32'))
        self.pos_3_list = torch.from_numpy(data['pos_3'].astype('float32'))
        self.pos_4_list = torch.from_numpy(data['pos_4'].astype('float32'))
        self.pos_5_list = torch.from_numpy(data['pos_5'].astype('float32'))
        self.x_1_list = torch.from_numpy(data['x_1'].astype('float32'))
        self.x_2_list = torch.from_numpy(data['x_2'].astype('float32'))
        self.x_3_list = torch.from_numpy(data['x_3'].astype('float32'))
        self.x_4_list = torch.from_numpy(data['x_4'].astype('float32'))
        self.x_5_list = torch.from_numpy(data['x_5'].astype('float32'))
        self.r_1_list = torch.from_numpy(data['r_1'].astype('float32'))
        self.r_2_list = torch.from_numpy(data['r_2'].astype('float32'))
        self.r_3_list = torch.from_numpy(data['r_3'].astype('float32'))
        self.r_4_list = torch.from_numpy(data['r_4'].astype('float32'))
        self.r_5_list = torch.from_numpy(data['r_5'].astype('float32'))
        self.total_frames = len(self.x_list)

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        return self.x_list[idx], self.r_list[idx], self.intrinsics_list[idx], \
            self.uv_1_list[idx], self.uv_2_list[idx], self.uv_3_list[idx], self.uv_4_list[idx], self.uv_5_list[idx], \
            self.pos_1_list[idx], self.pos_2_list[idx], self.pos_3_list[idx], self.pos_4_list[idx], self.pos_5_list[idx], \
            self.x_1_list[idx], self.x_2_list[idx], self.x_3_list[idx], self.x_4_list[idx], self.x_5_list[idx]


if __name__ == '__main__':
    dataset_ae = Dataset_AE(
        data_path='./data/3DPW/sequenceFiles', mode='train')
    print(len(dataset_ae), dataset_ae[0])

    dataset_transition = Dataset_Transition(
        data_path='./data/latent', mode='train')
    print(len(dataset_transition), dataset_transition[0])
