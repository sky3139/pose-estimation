import numpy as np
import torch
#from jacobian import compute_jacobian_multi
from functorch import jacrev, vmap


def cartesian2spherical(pos):
    r = np.linalg.norm(pos)
    theta = np.arccos(pos[2]/r)
    phi = np.arctan2(pos[1], pos[0])
    return r, theta, phi

def spherical2cartesian(r, theta, phi):
    if r.dtype is torch.float32:
        x = r*torch.sin(theta)*torch.cos(phi)
        y = r*torch.sin(theta)*torch.sin(phi)
        z = r*torch.cos(theta)
        return torch.stack([x, y, z], dim=-1)
    else:
        x = r*np.sin(theta)*np.cos(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(theta)
        return np.stack([x, y, z], axis=-1)

def skeleton2tensor(skeleton_3d):
    # joint positions
    x_0, y_0, z_0 = skeleton_3d[16] #LShoulder
    x_1, y_1, z_1 = skeleton_3d[17] #RShoulder
    x_2, y_2, z_2 = skeleton_3d[1] #LHip
    x_3, y_3, z_3 = skeleton_3d[2] #RHip
    x_4, y_4, z_4 = skeleton_3d[18] #LElbow
    x_5, y_5, z_5 = skeleton_3d[20] #LWrist
    x_6, y_6, z_6 = skeleton_3d[19] #RElbow
    x_7, y_7, z_7 = skeleton_3d[21] #RWrist
    x_8, y_8, z_8 = skeleton_3d[4] #LKnee
    x_9, y_9, z_9 = skeleton_3d[7] #LAnkle
    x_10, y_10, z_10 = skeleton_3d[5] #Rknee
    x_11, y_11, z_11 = skeleton_3d[8] #RAnkle

    # convert to spherical representation
    r_1, theta_1, phi_1 = cartesian2spherical(np.array([x_1, y_1, z_1]) - np.array([x_0, y_0, z_0]))
    r_2, theta_2, phi_2 = cartesian2spherical(np.array([x_2, y_2, z_2]) - np.array([(x_0+x_1)/2, (y_0+y_1)/2, (z_0+z_1)/2]))
    r_3, theta_3, phi_3 = cartesian2spherical(np.array([x_3, y_3, z_3]) - np.array([x_2, y_2, z_2]))
    r_4, theta_4, phi_4 = cartesian2spherical(np.array([x_4, y_4, z_4]) - np.array([x_0, y_0, z_0]))
    r_5, theta_5, phi_5 = cartesian2spherical(np.array([x_5, y_5, z_5]) - np.array([x_4, y_4, z_4]))
    r_6, theta_6, phi_6 = cartesian2spherical(np.array([x_6, y_6, z_6]) - np.array([x_1, y_1, z_1]))
    r_7, theta_7, phi_7 = cartesian2spherical(np.array([x_7, y_7, z_7]) - np.array([x_6, y_6, z_6]))
    r_8, theta_8, phi_8 = cartesian2spherical(np.array([x_8, y_8, z_8]) - np.array([x_2, y_2, z_2]))
    r_9, theta_9, phi_9 = cartesian2spherical(np.array([x_9, y_9, z_9]) - np.array([x_8, y_8, z_8]))
    r_10, theta_10, phi_10 = cartesian2spherical(np.array([x_10, y_10, z_10]) - np.array([x_3, y_3, z_3]))
    r_11, theta_11, phi_11 = cartesian2spherical(np.array([x_11, y_11, z_11]) - np.array([x_10, y_10, z_10]))

    x = torch.Tensor([x_0, y_0, z_0, theta_1, phi_1, theta_2, phi_2, theta_3, phi_3, theta_4, phi_4, theta_5, phi_5, theta_6, phi_6, theta_7, phi_7, theta_8, phi_8, theta_9, phi_9, theta_10, phi_10, theta_11, phi_11])
    r = torch.Tensor([r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8, r_9, r_10, r_11])
    return x, r

def tensor2skeleton(x, r):
    batch_input = len(x.size()) > 1
    if batch_input:
        new_order = [-1] + [i for i in range(len(x.shape)-1)]
        x_0, y_0, z_0, theta_1, phi_1, theta_2, phi_2, theta_3, phi_3, theta_4, phi_4, theta_5, phi_5, theta_6, phi_6, theta_7, phi_7, theta_8, phi_8, theta_9, phi_9, theta_10, phi_10, theta_11, phi_11 = x.permute(*new_order)
        r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8, r_9, r_10, r_11 = r.permute(*new_order)
    else:
        x_0, y_0, z_0, theta_1, phi_1, theta_2, phi_2, theta_3, phi_3, theta_4, phi_4, theta_5, phi_5, theta_6, phi_6, theta_7, phi_7, theta_8, phi_8, theta_9, phi_9, theta_10, phi_10, theta_11, phi_11 = x
        r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8, r_9, r_10, r_11 = r

    # convert to cartesian representation
    xyz_0 = torch.stack([x_0, y_0, z_0], dim=-1) #LShoulder
    xyz_1 = xyz_0 + spherical2cartesian(r_1, theta_1, phi_1) #RShoulder
    xyz_2 = (xyz_0 + xyz_1)/2 + spherical2cartesian(r_2, theta_2, phi_2) #LHip
    xyz_3 = xyz_2 + spherical2cartesian(r_3, theta_3, phi_3) #RHip
    xyz_4 = xyz_0 + spherical2cartesian(r_4, theta_4, phi_4) #LElbow
    xyz_5 = xyz_4 + spherical2cartesian(r_5, theta_5, phi_5) #LWrist
    xyz_6 = xyz_1 + spherical2cartesian(r_6, theta_6, phi_6) #RElbow
    xyz_7 = xyz_6 + spherical2cartesian(r_7, theta_7, phi_7) #RWrist
    xyz_8 = xyz_2 + spherical2cartesian(r_8, theta_8, phi_8) #LKnee
    xyz_9 = xyz_8 + spherical2cartesian(r_9, theta_9, phi_9) #LAnkle
    xyz_10 = xyz_3 + spherical2cartesian(r_10, theta_10, phi_10) #Rknee
    xyz_11 = xyz_10 + spherical2cartesian(r_11, theta_11, phi_11) #RAnkle

    if batch_input:
        pos = torch.stack([xyz_0, xyz_1, xyz_2, xyz_3, xyz_4, xyz_5, xyz_6, xyz_7, xyz_8, xyz_9, xyz_10, xyz_11], dim=len(x.shape)-1)
    else:
        pos = torch.stack([xyz_0, xyz_1, xyz_2, xyz_3, xyz_4, xyz_5, xyz_6, xyz_7, xyz_8, xyz_9, xyz_10, xyz_11])

    return pos

def pose2uv(pose):
    uv = np.stack([
        pose[:2,5], #LShoulder
        pose[:2,2], #RShoulder
        pose[:2,11], #LHip
        pose[:2,8], #RHip
        pose[:2,6], #LElbow
        pose[:2,7], #LWrist
        pose[:2,3], #RElbow
        pose[:2,4], #RWrist
        pose[:2,12], #LKnee
        pose[:2,13], #LAnkle
        pose[:2,9], #Rknee
        pose[:2,10], #RAnkle
    ], axis=0)
    return uv

def jointPos2camPos(extrinsics, jointPositions):
    homo_pos = np.concatenate((jointPositions, np.ones((jointPositions.shape[0], 1))), axis=-1)
    cam_pos = extrinsics @ homo_pos.T
    return cam_pos[:3].T

def pos2uv(pos, intrinsics):
    new_order = [i for i in range(len(pos.shape)-2)] + [len(pos.shape)-1, len(pos.shape)-2]
    uv = torch.matmul(intrinsics, pos.permute(*new_order)).permute(*new_order)
    uv = torch.stack([uv[...,0]/uv[...,2], uv[...,1]/uv[...,2]], dim=-1)
    # print(uv)
    return uv

def pos2uv_distribution(pos_mu, pos_var, intrinsics):
    ###############################################
    # udpate mu
    ###############################################
    uv_mu = pos2uv(pos_mu, intrinsics)
    
    ###############################################
    # update covariance
    ###############################################
    # calculate jacobian matrix
    J = vmap(jacrev(pos2uv))(pos_mu, intrinsics)
    J = J.reshape(J.shape[0], J.shape[1], J.shape[2], -1)
    J = J.reshape(J.shape[0], -1, J.shape[-1])
    # covariance matrix of uv
    uv_var = torch.matmul(torch.matmul(J, pos_var), torch.transpose(J, -2, -1))

    return uv_mu, uv_var

def x2pos_distribution(x_mu, x_var, r):
    ###############################################
    # udpate mu
    ###############################################
    pos_mu = tensor2skeleton(x_mu, r)
    
    ###############################################
    # update covariance
    ###############################################
    # calculate jacobian matrix
    J = vmap(jacrev(tensor2skeleton))(x_mu, r)
    J = J.reshape(J.shape[0], -1, J.shape[-1])
    # covariance matrix of pos
    pos_var = torch.matmul(torch.matmul(J, x_var), torch.transpose(J, -2, -1))

    return pos_mu, pos_var