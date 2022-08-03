import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import pickle as pkl
import numpy as np

from utils import *

class Dataset_AE(Dataset):
    def __init__(self, data_path, mode='train', transform=None):
        super(Dataset_AE, self).__init__()
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        self.folder_path = os.path.join(self.data_path, self.mode)
        self.pkl_files = [os.path.join(self.folder_path, pkl) for pkl in os.listdir(self.folder_path)]
        self.x_list = []
        for pkl_file in self.pkl_files:
            x_list = self.read_pickle(pkl_file)
            self.x_list.extend(x_list)
        self.total_frames = len(self.x_list)
    
    def read_pickle(self, pkl_file):
        u = pkl._Unpickler(open(pkl_file,'rb'))
        u.encoding = 'latin1'
        seq = u.load()
        x_list = []
        for jointPositions in seq['jointPositions']:
            total_frames = jointPositions.shape[0]
            jointPositions = jointPositions.reshape(total_frames, -1, 3)
            # print(jointPositions.shape)
            for t in range(total_frames):
                skeleton_3d = jointPos2camPos(seq['cam_poses'][t], jointPositions[t])
                x, _ = skeleton2tensor(skeleton_3d)
                
                # transform
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


class Dataset_Transition(Dataset):
    def __init__(self, data_path, mode='train'):
        super(Dataset_Transition, self).__init__()
        self.data_path = data_path
        self.mode = mode
        self.pkl_file = os.path.join(self.data_path, 'latent-{}-multi.pkl'.format(mode))
        with open(self.pkl_file, 'rb') as handle:
            data = pkl.load(handle)
        self.x_list = torch.from_numpy(data['x'].astype('float32'))
        self.r_list = torch.from_numpy(data['r'].astype('float32'))
        self.intrinsics_list = torch.from_numpy(data['intrinsics'].astype('float32'))
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
    dataset_ae = Dataset_AE(data_path='./data/3DPW/sequenceFiles', mode='train')
    print(len(dataset_ae), dataset_ae[0])

    dataset_transition = Dataset_Transition(data_path='./data/latent', mode='train')
    print(len(dataset_transition), dataset_transition[0])