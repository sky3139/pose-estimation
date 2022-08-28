# import cdflib
# import torch
# from models import VAE
import enum
import os
import pickle as pkl
import numpy as np
# from icecream import ic as print
from utils import *


def cal_k_p(Points_cali):
    a = np.zeros((32, 11))
    ii = 0
    for i in range(32):
        if (i % 2 == 0):
            a[i][0] = Points_cali[ii][0]
            a[i][1] = Points_cali[ii][1]
            a[i][2] = Points_cali[ii][2]
            a[i][3] = 1
            a[i][8] = -Points_cali[ii][0] * Points_cali[ii][3]
            a[i][9] = -Points_cali[ii][1] * Points_cali[ii][3]
            a[i][10] = -Points_cali[ii][2] * Points_cali[ii][3]
        else:
            a[i][4] = Points_cali[ii][0]
            a[i][5] = Points_cali[ii][1]
            a[i][6] = Points_cali[ii][2]
            a[i][7] = 1
            a[i][8] = -Points_cali[ii][0] * Points_cali[ii][4]
            a[i][9] = -Points_cali[ii][1] * Points_cali[ii][4]
            a[i][10] = -Points_cali[ii][2] * Points_cali[ii][4]
            ii += 1
    # print(a)
    u = np.zeros((32, 1))
    ii = 0
    for i in range(16):
        u[i*2] = Points_cali[ii][3]
        u[i*2 + 1] = Points_cali[ii][4]
        ii += 1
    L = np.linalg.inv(np.dot(a.T, a))
    L = np.dot(np.dot(L, a.T), u)
    # print(L)
    _tp = pow(L[8][0], 2) + pow(L[9][0], 2) + pow(L[10][0], 2)
    u0 = (L[0][0] * L[8][0] + L[1][0] * L[9][0] + L[2][0] * L[10][0]) / _tp
    v0 = (L[4][0] * L[8][0] + L[5][0] * L[9][0]+L[6][0] * L[10][0]) / _tp

    fu = np.sqrt((pow(u0 * L[8][0] - L[0][0], 2) + pow(u0 * L[9]
                 [0] - L[1][0], 2) + pow(u0 * L[10][0] - L[2][0], 2)) / _tp)
    fv = np.sqrt((pow(v0 * L[8][0] - L[4][0], 2) + pow(v0 * L[9]
                 [0] - L[5][0], 2) + pow(v0 * L[10][0] - L[6][0], 2)) / _tp)
    K = np.array([fu, 0, u0, 0, fv, v0, 0, 0, 1]).reshape(3, 3)
    # print(K)
    K_34 = np.append(L, 1.0).reshape(3, 4)
    # print(K_34)
    pose = np.dot(np.linalg.inv(K), K_34)/np.sqrt(_tp)
    bt = np.array([0, 0, 0, 1])
    pose = np.vstack((pose, bt))
    return pose, K


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

def quaternion_to_rotation_matrix(quat, t):
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

def cam2uv(CK, p3ds):
    pg = []
    for p3 in p3ds:
        pg.append(p3/p3[2])
    pg = np.array(pg).reshape(-1, 3)
    ans = np.dot(CK, pg.T).T
    ans = np.delete(ans, 2, axis=1)
    return ans


def word2cam(cam_1R, wordp3):
    g_p3 = np.insert(wordp3, 3, 1, axis=1).T
    cam_p3s = np.dot(cam_1R, g_p3).T
    # print(cam_p3s)
    skeleton_t = np.delete(cam_p3s, 3, axis=1)
    return skeleton_t




def main():
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


    data_3d = np.load(
        "/home/u20/d2/code/PoseFormer_bak/data/data_3d_h36m.npz", allow_pickle=True)
    data3d_item = data_3d['positions_3d'].item()
    cam_1R = quaternion_to_rotation_matrix(np.array([0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]), [
                                        1.8411070556640625, 4.95528466796875, 1.5634454345703125])
    print(cam_1R)
    cam_name = []
    cams_K = []

    for v in data3d_item["cam"]:
        cam_name.append(v['id'])
        K = np.array([v["focal_length"][0], 0, v["center"][0], 0,
                    v["focal_length"][1], v["center"][1], 0, 0, 1]).reshape(3, 3)
        cams_K.append(K)
    # print(cam_name[0],cams_K[0],cam_1R)
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
            # p2s_=np.insert(p2s,2,value=1,axis=1)
            _p, _k = cal_k_p(np.hstack((p3s.reshape(-1, 3), p2s.reshape(-1, 2))))
            # print(_k,_p)
            frame_len = len(p3s)-7
            for frame, skeleton_t in enumerate(p3s):
                if frame > frame_len:
                    break
                # skeleton_t=get12(skeleton_t)
                cam_p3d = word2cam(_p, skeleton_t)
                uv = cam2uv(_k, cam_p3d)
                # print(uv, p2s[frame])
                # print(skeleton_t)
                # for frame,p3 in p3s:
                # ___a=np.hstack((points, points_2d))
                # pose,K=cal_k_p(___a)
                # skeleton_t=
                # skeleton_t = get12(p3)  # cam 3d
                # pix2d =get12(p2s[frame]) # p2s[frame]  # pix 2d
                # print(skeleton_t,pix2d)

                # print(cam_p3d)

                # print(cam_p3d,np.dot(cams_K[0],cam_p3d.T).T)
                x_t, r_t = skeleton2tensor(cam_p3d)
                # print(poses2d)

                skeleton_1 = word2cam(_p, p3s[frame+1])
                skeleton_2 = word2cam(_p, p3s[frame+2])
                skeleton_3 = word2cam(_p, p3s[frame+3])
                skeleton_4 = word2cam(_p, p3s[frame+4])
                skeleton_5 = word2cam(_p, p3s[frame+5])

                uv_1 = get12(cam2uv(_k,skeleton_1))
                uv_2 = get12(cam2uv(_k,skeleton_2))
                uv_3 = get12(cam2uv(_k,skeleton_3))
                uv_4 = get12(cam2uv(_k,skeleton_4))
                uv_5 = get12(cam2uv(_k,skeleton_5))

                # print(len(skeleton_1),skeleton_1)
                x_1, r_1 = skeleton2tensor(skeleton_1)
                x_2, r_2 = skeleton2tensor(skeleton_2)
                x_3, r_3 = skeleton2tensor(skeleton_3)
                x_4, r_4 = skeleton2tensor(skeleton_4)
                x_5, r_5 = skeleton2tensor(skeleton_5)
                pos_1 = tensor2skeleton(x_1, r_1)
                pos_2 = tensor2skeleton(x_2, r_2)
                pos_3 = tensor2skeleton(x_3, r_3)
                pos_4 = tensor2skeleton(x_4, r_4)
                pos_5 = tensor2skeleton(x_5, r_5)

                if (uv_1 == 0).any() or (uv_2 == 0).any() or (uv_3 == 0).any() or (uv_4 == 0).any() or (uv_5 == 0).any():
                    continue

                x_list.append(x_t.cpu().numpy())
                r_list.append(r_t)
                intrinsics_list.append(_k)
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
    with open('./data/human/latent-{}-multi.pkl'.format(mode), 'wb') as handle:
        pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)
if __name__ == "main":
    main()