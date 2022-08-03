import numpy as np
from numpy import sin, cos
from icecream import ic as print
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import time

with open('camera_3d_infrared.npy', 'rb') as f:
    skeleton_3d = np.load(f)
    # print(skeleton_3d, skeleton_3d.shape)

# camera matrix
cam_matrix = np.array([[435.547921, 0.000000, 323.890159],
                       [0.000000, 434.576666, 237.879057],
                       [0.000000, 0.000000, 1.000000]])
fx, s, v = cam_matrix[0]
fy, w = cam_matrix[1][1:]

#LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
#LHip, RHip, LKnee, Rknee, LAnkle, RAnkle
x_0, y_0, z_0 = skeleton_3d[34] #LShoulder
x_1, y_1, z_1 = skeleton_3d[33] #RShoulder
x_2, y_2, z_2 = skeleton_3d[28] #LHip
x_3, y_3, z_3 = skeleton_3d[27] #RHip
x_4, y_4, z_4 = skeleton_3d[35] #LElbow
x_5, y_5, z_5 = skeleton_3d[36] #LWrist
x_6, y_6, z_6 = skeleton_3d[32] #RElbow
x_7, y_7, z_7 = skeleton_3d[31] #RWrist
x_8, y_8, z_8 = skeleton_3d[29] #LKnee
x_9, y_9, z_9 = skeleton_3d[30] #LAnkle
x_10, y_10, z_10 = skeleton_3d[26] #Rknee
x_11, y_11, z_11 = skeleton_3d[25] #RAnkle


def cartesian2spherical(pos):
    r = np.linalg.norm(pos)
    theta = np.arccos(pos[2]/r)
    phi = np.arctan2(pos[1], pos[0])
    return r, theta, phi

def spherical2cartesian(r, theta, phi):
    x = r*sin(theta)*cos(phi)
    y = r*sin(theta)*sin(phi)
    z = r*cos(theta)
    return x, y, z

# position
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
# print(r_2, theta_2, phi_2)
# print(r_3, theta_3, phi_3)
# exit(0)

# velocity
x_dot_0, y_dot_0, z_dot_0 = 0, 0, 0
theta_dot_1, phi_dot_1 = 0, 0
theta_dot_2, phi_dot_2 = 0, 0
theta_dot_3, phi_dot_3 = 0, 0
theta_dot_4, phi_dot_4 = 0, 0
theta_dot_5, phi_dot_5 = 0, 0
theta_dot_6, phi_dot_6 = 0, 0
theta_dot_7, phi_dot_7 = 0, 0
theta_dot_8, phi_dot_8 = 0, 0
theta_dot_9, phi_dot_9 = 0, 0
theta_dot_10, phi_dot_10 = 0, 0
theta_dot_11, phi_dot_11 = 0, 0

# define state
x = np.array([x_0, x_dot_0,
              y_0, y_dot_0,
              z_0, z_dot_0,
              theta_1, theta_dot_1,
              phi_1, phi_dot_1,
              theta_2, theta_dot_2,
              phi_2, phi_dot_2,
              theta_3, theta_dot_3,
              # phi_3, phi_dot_3,
              theta_4, theta_dot_4,
              phi_4, phi_dot_4,
              theta_5, theta_dot_5,
              phi_5, phi_dot_5,
              theta_6, theta_dot_6,
              phi_6, phi_dot_6,
              theta_7, theta_dot_7,
              phi_7, phi_dot_7,
              theta_8, theta_dot_8,
              phi_8, phi_dot_8,
              theta_9, theta_dot_9,
              phi_9, phi_dot_9,
              theta_10, theta_dot_10,
              phi_10, phi_dot_10,
              theta_11, theta_dot_11,
              phi_11, phi_dot_11,
              ])

def state2pos(x, normalize=True):
    x_0, y_0, z_0 = x[0], x[2], x[4] #LShoulder
    x_1, y_1, z_1 = np.array([x_0, y_0, z_0]) + spherical2cartesian(r_1, x[6], x[8]) #RShoulder
    x_2, y_2, z_2 = np.array([(x_0+x_1)/2, (y_0+y_1)/2, (z_0+z_1)/2]) + spherical2cartesian(r_2, x[10], x[12]) #LHip
    x_3, y_3, z_3 = np.array([x_2, y_2, z_2]) + spherical2cartesian(r_3, x[14], phi_3) #RHip
    x_4, y_4, z_4 = np.array([x_0, y_0, z_0]) + spherical2cartesian(r_4, x[16], x[18]) #LElbow
    x_5, y_5, z_5 = np.array([x_4, y_4, z_4]) + spherical2cartesian(r_5, x[20], x[22]) #LWrist
    x_6, y_6, z_6 = np.array([x_1, y_1, z_1]) + spherical2cartesian(r_6, x[24], x[26]) #RElbow
    x_7, y_7, z_7 = np.array([x_6, y_6, z_6]) + spherical2cartesian(r_7, x[28], x[30]) #RWrist
    x_8, y_8, z_8 = np.array([x_2, y_2, z_2]) + spherical2cartesian(r_8, x[32], x[34]) #LKnee
    x_9, y_9, z_9 = np.array([x_8, y_8, z_8]) + spherical2cartesian(r_9, x[36], x[38]) #LAnkle
    x_10, y_10, z_10 = np.array([x_3, y_3, z_3]) + spherical2cartesian(r_10, x[40], x[42]) #Rknee
    x_11, y_11, z_11 = np.array([x_10, y_10, z_10]) + spherical2cartesian(r_11, x[44], x[46]) #RAnkle
    if normalize:
        return np.array([
            [x_0/z_0, y_0/z_0, 1],
            [x_1/z_1, y_1/z_1, 1],
            [x_2/z_2, y_2/z_2, 1],
            [x_3/z_3, y_3/z_3, 1],
            [x_4/z_4, y_4/z_4, 1],
            [x_5/z_5, y_5/z_5, 1],
            [x_6/z_6, y_6/z_6, 1],
            [x_7/z_7, y_7/z_7, 1],
            [x_8/z_8, y_8/z_8, 1],
            [x_9/z_9, y_9/z_9, 1],
            [x_10/z_10, y_10/z_10, 1],
            [x_11/z_11, y_11/z_11, 1],
        ])
    else:
        return np.array([
            [x_0, y_0, z_0],
            [x_1, y_1, z_1],
            [x_2, y_2, z_2],
            [x_3, y_3, z_3],
            [x_4, y_4, z_4],
            [x_5, y_5, z_5],
            [x_6, y_6, z_6],
            [x_7, y_7, z_7],
            [x_8, y_8, z_8],
            [x_9, y_9, z_9],
            [x_10, y_10, z_10],
            [x_11, y_11, z_11],
        ])

# params
num_joints = 12
num_edges = 11
num_states = 4*num_joints
num_z = 3*num_joints+3
dt = 0.05
# propagation
F = np.eye(num_states)
for i in range(0, num_states, 2):
    F[i,i+1] = dt
P = np.eye(num_states)*1/100.
# R = np.eye(num_z)*10
R = np.eye(num_z)
for i in range(num_z):
    # u, v
    if i < 2*num_joints:
        R[i,i] *= 15
    # constraints
    elif i >= 2*num_joints and i < 2*num_joints+3:
        R[i,i] *= 1
    # depth
    else:
        R[i,i] *= 10
G = np.zeros((num_states, 2*num_joints))
for i in range(1, num_states, 2):
    G[i,i//2] = 1
V_NOISE = np.zeros(2*num_joints)
V_NOISE[:3] = 1.0
V_NOISE[3:] = 0.1
Qd = np.eye(2*num_joints)*0.1

# import json
# with open('./AlphaPose/examples/res1/alphapose-results.json', 'r') as json_file:
#     skeleton_2d = json.load(json_file)
# # print(type(skeleton_2d), len(skeleton_2d))
# # skeleton_2d = np.stack(([np.array(data['keypoints']).reshape(-1, 3) for data in skeleton_2d]))
# # clean data
# skeleton_2d_list = []
# for data in skeleton_2d:
#     keypoints = np.array(data['keypoints']).reshape(-1, 3)
#     if len(skeleton_2d_list) == 0:
#         skeleton_2d_list.append(keypoints)
#     else:
#         error = np.abs(keypoints - skeleton_2d_list[-1]).mean()
#         # print(error)
#         if error < 10:
#             skeleton_2d_list.append(keypoints)
# skeleton_2d = np.stack(skeleton_2d_list)
# print(skeleton_2d.shape)

with open('uv_with_disparity.npy', 'rb') as f:
    skeleton_2d = np.load(f)

total_frames = skeleton_2d.shape[0]
print(total_frames)

###############################
# Test
###############################
# uv_0 = cam_matrix @ np.array([x_0, y_0, z_0])
# uv_0 /= uv_0[2]
# print(x_0, y_0, z_0)
# print(uv_0, ((skeleton_2d[0][5] + skeleton_2d[0][6])/2)[:2])
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# for pos in skeleton_3d:
#     xs = pos[0]
#     ys = pos[1]
#     zs = pos[2]
#     ax.scatter(xs, ys, zs, marker='*')

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()
###############################
###############################

line_x_all = []
line_y_all = []
line_z_all = []
# total_frames = 1000
for t in range(total_frames):
    print(t)
    skeleton = skeleton_2d[t]
    u_0, v_0, d_0 = skeleton[5]
    u_1, v_1, d_1 = skeleton[6]
    u_2, v_2, d_2 = skeleton[11]
    u_3, v_3, d_3 = skeleton[12]
    u_4, v_4, d_4 = skeleton[7]
    u_5, v_5, d_5 = skeleton[9]
    u_6, v_6, d_6 = skeleton[8]
    u_7, v_7, d_7 = skeleton[10]
    u_8, v_8, d_8 = skeleton[13]
    u_9, v_9, d_9 = skeleton[15]
    u_10, v_10, d_10 = skeleton[14]
    u_11, v_11, d_11 = skeleton[16]
    c_0, c_1, c_2 = 0, 0, 0
    z = np.array([u_0, v_0,
                  u_1, v_1,
                  u_2, v_2,
                  u_3, v_3,
                  u_4, v_4,
                  u_5, v_5,
                  u_6, v_6,
                  u_7, v_7,
                  u_8, v_8,
                  u_9, v_9,
                  u_10, v_10,
                  u_11, v_11,
                  c_0, c_1, c_2,
                  d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9, d_10, d_11
                  ])
    # print(z)
    x_hat = F @ x + G @ V_NOISE    # TODO: Add noise
    P = F @ P @ F.T + G @ Qd @ G.T   # TODO: Add noise
    uv_hat = cam_matrix @ state2pos(x_hat).T
    uv_hat = uv_hat[:2].T.flatten()
    # print(uv_hat)
    # print(x_hat)
    ############################
    # observation matrix
    ############################
    H = np.zeros((num_z, num_states))
    # u_0, v_0
    du0_dx0 = fx/z_0
    du0_dy0 = s/z_0
    du0_dz0 = -(fx*x_0+s*y_0)/z_0**2
    dv0_dx0 = 0
    dv0_dy0 = fy/z_0
    dv0_dz0 = -fy*y_0/z_0**2
    ############################
    H[0,0] = du0_dx0
    H[0,2] = du0_dy0
    H[0,4] = du0_dz0
    H[1,0] = dv0_dx0
    H[1,2] = dv0_dy0
    H[1,4] = dv0_dz0

    # u_1, v_1
    du1_dx1 = fx/z_1
    du1_dy1 = s/z_1
    du1_dz1 = -(fx*x_1+s*y_1)/z_1**2
    dv1_dx1 = 0
    dv1_dy1 = fy/z_1
    dv1_dz1 = -fy*y_1/z_1**2
    dx1_dx0 = 1
    dx1_dtheta1 = r_1*cos(theta_1)*cos(phi_1)
    dx1_dphi1 = -r_1*sin(theta_1)*sin(phi_1)
    dy1_dy0 = 1
    dy1_dtheta1 = r_1*cos(theta_1)*sin(phi_1)
    dy1_dphi1 = r_1*sin(theta_1)*cos(phi_1)
    dz1_dz0 = 1
    dz1_dtheta1 = -r_1*sin(theta_1)
    dz1_dphi1 = 0
    ############################
    du1_dx0 = du1_dx1*dx1_dx0
    du1_dy0 = du1_dy1*dy1_dy0
    du1_dz0 = du1_dz1*dz1_dz0
    du1_dtheta1 = du1_dx1*dx1_dtheta1 + du1_dy1*dy1_dtheta1 + du1_dz1*dz1_dtheta1
    du1_dphi1 = du1_dx1*dx1_dphi1 + du1_dy1*dy1_dphi1 + du1_dz1*dz1_dphi1
    dv1_dx0 = dv1_dx1*dx1_dx0
    dv1_dy0 = dv1_dy1*dy1_dy0
    dv1_dz0 = dv1_dz1*dz1_dz0
    dv1_dtheta1 = dv1_dx1*dx1_dtheta1 + dv1_dy1*dy1_dtheta1 + dv1_dz1*dz1_dtheta1
    dv1_dphi1 = dv1_dx1*dx1_dphi1 + dv1_dy1*dy1_dphi1 + dv1_dz1*dz1_dphi1
    ############################
    H[2,0] = du1_dx0
    H[2,2] = du1_dy0
    H[2,4] = du1_dz0
    H[2,6] = du1_dtheta1
    H[2,8] = du1_dphi1
    H[3,0] = dv1_dx0
    H[3,2] = dv1_dy0
    H[3,4] = dv1_dz0
    H[3,6] = dv1_dtheta1
    H[3,8] = dv1_dphi1

    # u_2, v_2
    du2_dx2 = fx/z_2
    du2_dy2 = s/z_2
    du2_dz2 = -(fx*x_2+s*y_2)/z_2**2
    dv2_dx2 = 0
    dv2_dy2 = fy/z_2
    dv2_dz2 = -fy*y_2/z_2**2
    dx2_dx0 = 1/2*(1+dx1_dx0)
    dx2_dtheta1 = 1/2*dx1_dtheta1
    dx2_dphi1 = 1/2*dx1_dphi1
    dx2_dtheta2 = r_2*cos(theta_2)*cos(phi_2)
    dx2_dphi2 = -r_2*sin(theta_2)*sin(phi_2)
    dy2_dy0 = 1/2*(1+dy1_dy0)
    dy2_dtheta1 = 1/2*dy1_dtheta1
    dy2_dphi1 = 1/2*dy1_dphi1
    dy2_dtheta2 = r_2*cos(theta_2)*sin(phi_2)
    dy2_dphi2 = r_2*sin(theta_2)*cos(phi_2)
    dz2_dz0 = 1/2*(1+dz1_dz0)
    dz2_dtheta1 = 1/2*dz1_dtheta1
    dz2_dphi1 = 1/2*dz1_dphi1
    dz2_dtheta2 = -r_2*sin(theta_2)
    dz2_dphi2 = 0
    ############################
    du2_dx0 = du2_dx2*dx2_dx0
    du2_dy0 = du2_dy2*dy2_dy0
    du2_dz0 = du2_dz2*dz2_dz0
    du2_dtheta1 = du2_dx2*dx2_dtheta1 + du2_dy2*dy2_dtheta1 + du2_dz2*dz2_dtheta1
    du2_dphi1 = du2_dx2*dx2_dphi1 + du2_dy2*dy2_dphi1 + du2_dz2*dz2_dphi1
    du2_dtheta2 = du2_dx2*dx2_dtheta2 + du2_dy2*dy2_dtheta2 + du2_dz2*dz2_dtheta2
    du2_dphi2 = du2_dx2*dx2_dphi2 + du2_dy2*dy2_dphi2 + du2_dz2*dz2_dphi2
    dv2_dx0 = dv2_dx2*dx2_dx0
    dv2_dy0 = dv2_dy2*dy2_dy0
    dv2_dz0 = dv2_dz2*dz2_dz0
    dv2_dtheta1 = dv2_dx2*dx2_dtheta1 + dv2_dy2*dy2_dtheta1 + dv2_dz2*dz2_dtheta1
    dv2_dphi1 = dv2_dx2*dx2_dphi1 + dv2_dy2*dy2_dphi1 + dv2_dz2*dz2_dphi1
    dv2_dtheta2 = dv2_dx2*dx2_dtheta2 + dv2_dy2*dy2_dtheta2 + dv2_dz2*dz2_dtheta2
    dv2_dphi2 = dv2_dx2*dx2_dphi2 + dv2_dy2*dy2_dphi2 + dv2_dz2*dz2_dphi2
    ############################
    H[4,0] = du2_dx0
    H[4,2] = du2_dy0
    H[4,4] = du2_dz0
    H[4,6] = du2_dtheta1
    H[4,8] = du2_dphi1
    H[4,10] = du2_dtheta2
    H[4,12] = du2_dphi2
    H[5,0] = dv2_dx0
    H[5,2] = dv2_dy0
    H[5,4] = dv2_dz0
    H[5,6] = dv2_dtheta1
    H[5,8] = dv2_dphi1
    H[5,10] = dv2_dtheta2
    H[5,12] = dv2_dphi2

    # u_3, v_3
    du3_dx3 = fx/z_3
    du3_dy3 = s/z_3
    du3_dz3 = -(fx*x_3+s*y_3)/z_3**2
    dv3_dx3 = 0
    dv3_dy3 = fy/z_3
    dv3_dz3 = -fy*y_3/z_3**2
    dx3_dx0 = dx2_dx0
    dx3_dtheta1 = dx2_dtheta1
    dx3_dphi1 = dx2_dphi1
    dx3_dtheta2 = dx2_dtheta2
    dx3_dphi2 = dx2_dphi2
    dx3_dtheta3 = r_3*cos(theta_3)*cos(phi_3)
    dx3_dphi3 = -r_3*sin(theta_3)*sin(phi_3)
    dy3_dy0 = dy2_dy0
    dy3_dtheta1 = dy2_dtheta1
    dy3_dphi1 = dy2_dphi1
    dy3_dtheta2 = dy2_dtheta2
    dy3_dphi2 = dy2_dphi2
    dy3_dtheta3 = r_3*cos(theta_3)*sin(phi_3)
    dy3_dphi3 = r_3*sin(theta_3)*cos(phi_3)
    dz3_dz0 = dz2_dz0
    dz3_dtheta1 = dz2_dtheta1
    dz3_dphi1 = dz2_dphi1
    dz3_dtheta2 = dz2_dtheta2
    dz3_dphi2 = dz2_dphi2
    dz3_dtheta3 = -r_3*sin(theta_3)
    dz3_dphi3 = 0
    ############################
    du3_dx0 = du3_dx3*dx3_dx0
    du3_dy0 = du3_dy3*dy3_dy0
    du3_dz0 = du3_dz3*dz3_dz0
    du3_dtheta1 = du3_dx3*dx3_dtheta1 + du3_dy3*dy3_dtheta1 + du3_dz3*dz3_dtheta1
    du3_dphi1 = du3_dx3*dx3_dphi1 + du3_dy3*dy3_dphi1 + du3_dz3*dz3_dphi1
    du3_dtheta2 = du3_dx3*dx3_dtheta2 + du3_dy3*dy3_dtheta2 + du3_dz3*dz3_dtheta2
    du3_dphi2 = du3_dx3*dx3_dphi2 + du3_dy3*dy3_dphi2 + du3_dz3*dz3_dphi2
    du3_dtheta3 = du3_dx3*dx3_dtheta3 + du3_dy3*dy3_dtheta3 + du3_dz3*dz3_dtheta3
    du3_dphi3 = du3_dx3*dx3_dphi3 + du3_dy3*dy3_dphi3 + du3_dz3*dz3_dphi3
    dv3_dx0 = dv3_dx3*dx3_dx0
    dv3_dy0 = dv3_dy3*dy3_dy0
    dv3_dz0 = dv3_dz3*dz3_dz0
    dv3_dtheta1 = dv3_dx3*dx3_dtheta1 + dv3_dy3*dy3_dtheta1 + dv3_dz3*dz3_dtheta1
    dv3_dphi1 = dv3_dx3*dx3_dphi1 + dv3_dy3*dy3_dphi1 + dv3_dz3*dz3_dphi1
    dv3_dtheta2 = dv3_dx3*dx3_dtheta2 + dv3_dy3*dy3_dtheta2 + dv3_dz3*dz3_dtheta2
    dv3_dphi2 = dv3_dx3*dx3_dphi2 + dv3_dy3*dy3_dphi2 + dv3_dz3*dz3_dphi2
    dv3_dtheta3 = dv3_dx3*dx3_dtheta3 + dv3_dy3*dy3_dtheta3 + dv3_dz3*dz3_dtheta3
    dv3_dphi3 = dv3_dx3*dx3_dphi3 + dv3_dy3*dy3_dphi3 + dv3_dz3*dz3_dphi3
    ############################
    H[6,0] = du3_dx0
    H[6,2] = du3_dy0
    H[6,4] = du3_dz0
    H[6,6] = du3_dtheta1
    H[6,8] = du3_dphi1
    H[6,10] = du3_dtheta2
    H[6,12] = du3_dphi2
    H[6,14] = du3_dtheta3
    H[7,0] = dv3_dx0
    H[7,2] = dv3_dy0
    H[7,4] = dv3_dz0
    H[7,6] = dv3_dtheta1
    H[7,8] = dv3_dphi1
    H[7,10] = dv3_dtheta2
    H[7,12] = dv3_dphi2
    H[7,14] = dv3_dtheta3

    # u_4, v_4
    du4_dx4 = fx/z_4
    du4_dy4 = s/z_4
    du4_dz4 = -(fx*x_4+s*y_4)/z_4**2
    dv4_dx4 = 0
    dv4_dy4 = fy/z_4
    dv4_dz4 = -fy*y_4/z_4**2
    dx4_dx0 = 1
    dx4_dtheta4 = r_4*cos(theta_4)*cos(phi_4)
    dx4_dphi4 = -r_4*sin(theta_4)*sin(phi_4)
    dy4_dy0 = 1
    dy4_dtheta4 = r_4*cos(theta_4)*sin(phi_4)
    dy4_dphi4 = r_4*sin(theta_4)*cos(phi_4)
    dz4_dz0 = 1
    dz4_dtheta4 = -r_4*sin(theta_4)
    dz4_dphi4 = 0
    ############################
    du4_dx0 = du4_dx4*dx4_dx0
    du4_dy0 = du4_dy4*dy4_dy0
    du4_dz0 = du4_dz4*dz4_dz0
    du4_dtheta4 = du4_dx4*dx4_dtheta4 + du4_dy4*dy4_dtheta4 + du4_dz4*dz4_dtheta4
    du4_dphi4 = du4_dx4*dx4_dphi4 + du4_dy4*dy4_dphi4 + du4_dz4*dz4_dphi4
    dv4_dx0 = dv4_dx4*dx4_dx0
    dv4_dy0 = dv4_dy4*dy4_dy0
    dv4_dz0 = dv4_dz4*dz4_dz0
    dv4_dtheta4 = dv4_dx4*dx4_dtheta4 + dv4_dy4*dy4_dtheta4 + dv4_dz4*dz4_dtheta4
    dv4_dphi4 = dv4_dx4*dx4_dphi4 + dv4_dy4*dy4_dphi4 + dv4_dz4*dz4_dphi4
    ############################
    H[8,0] = du4_dx0
    H[8,2] = du4_dy0
    H[8,4] = du4_dz0
    H[8,16] = du4_dtheta4
    H[8,18] = du4_dphi4
    H[9,0] = dv4_dx0
    H[9,2] = dv4_dy0
    H[9,4] = dv4_dz0
    H[9,16] = dv4_dtheta4
    H[9,18] = dv4_dphi4

    # u_5, v_5
    du5_dx5 = fx/z_5
    du5_dy5 = s/z_5
    du5_dz5 = -(fx*x_5+s*y_5)/z_5**2
    dv5_dx5 = 0
    dv5_dy5 = fy/z_5
    dv5_dz5 = -fy*y_5/z_5**2
    dx5_dx0 = dx4_dx0
    dx5_dtheta4 = dx4_dtheta4
    dx5_dphi4 = dx4_dphi4
    dx5_dtheta5 = r_5*cos(theta_5)*cos(phi_5)
    dx5_dphi5 = -r_5*sin(theta_5)*sin(phi_5)
    dy5_dy0 = dy4_dy0
    dy5_dtheta4 = dy4_dtheta4
    dy5_dphi4 = dy4_dphi4
    dy5_dtheta5 = r_5*cos(theta_5)*sin(phi_5)
    dy5_dphi5 = r_5*sin(theta_5)*cos(phi_5)
    dz5_dz0 = dz4_dz0
    dz5_dtheta4 = dz4_dtheta4
    dz5_dphi4 = dz4_dphi4
    dz5_dtheta5 = -r_5*sin(theta_5)
    dz5_dphi5 = 0
    ############################
    du5_dx0 = du5_dx5*dx5_dx0
    du5_dy0 = du5_dy5*dy5_dy0
    du5_dz0 = du5_dz5*dz5_dz0
    du5_dtheta4 = du5_dx5*dx5_dtheta4 + du5_dy5*dy5_dtheta4 + du5_dz5*dz5_dtheta4
    du5_dphi4 = du5_dx5*dx5_dphi4 + du5_dy5*dy5_dphi4 + du5_dz5*dz5_dphi4
    du5_dtheta5 = du5_dx5*dx5_dtheta5 + du5_dy5*dy5_dtheta5 + du5_dz5*dz5_dtheta5
    du5_dphi5 = du5_dx5*dx5_dphi5 + du5_dy5*dy5_dphi5 + du5_dz5*dz5_dphi5
    dv5_dx0 = dv5_dx5*dx5_dx0
    dv5_dy0 = dv5_dy5*dy5_dy0
    dv5_dz0 = dv5_dz5*dz5_dz0
    dv5_dtheta4 = dv5_dx5*dx5_dtheta4 + dv5_dy5*dy5_dtheta4 + dv5_dz5*dz5_dtheta4
    dv5_dphi4 = dv5_dx5*dx5_dphi4 + dv5_dy5*dy5_dphi4 + dv5_dz5*dz5_dphi4
    dv5_dtheta5 = dv5_dx5*dx5_dtheta5 + dv5_dy5*dy5_dtheta5 + dv5_dz5*dz5_dtheta5
    dv5_dphi5 = dv5_dx5*dx5_dphi5 + dv5_dy5*dy5_dphi5 + dv5_dz5*dz5_dphi5
    ############################
    H[10,0] = du5_dx0
    H[10,2] = du5_dy0
    H[10,4] = du5_dz0
    H[10,16] = du5_dtheta4
    H[10,18] = du5_dphi4
    H[10,20] = du5_dtheta5
    H[10,22] = du5_dphi5
    H[11,0] = dv5_dx0
    H[11,2] = dv5_dy0
    H[11,4] = dv5_dz0
    H[11,16] = dv5_dtheta4
    H[11,18] = dv5_dphi4
    H[11,20] = dv5_dtheta5
    H[11,22] = dv5_dphi5

    # u_6, v_6
    du6_dx6 = fx/z_6
    du6_dy6 = s/z_6
    du6_dz6 = -(fx*x_6+s*y_6)/z_6**2
    dv6_dx6 = 0
    dv6_dy6 = fy/z_6
    dv6_dz6 = -fy*y_6/z_6**2
    dx6_dx0 = dx1_dx0
    dx6_dtheta1 = dx1_dtheta1
    dx6_dphi1 = dx1_dphi1
    dx6_dtheta6 = r_6*cos(theta_6)*cos(phi_6)
    dx6_dphi6 = -r_6*sin(theta_6)*sin(phi_6)
    dy6_dy0 = dy1_dy0
    dy6_dtheta1 = dy1_dtheta1
    dy6_dphi1 = dy1_dphi1
    dy6_dtheta6 = r_6*cos(theta_6)*sin(phi_6)
    dy6_dphi6 = r_6*sin(theta_6)*cos(phi_6)
    dz6_dz0 = dz1_dz0
    dz6_dtheta1 = dz1_dtheta1
    dz6_dphi1 = dz1_dphi1
    dz6_dtheta6 = -r_6*sin(theta_6)
    dz6_dphi6 = 0
    ############################
    du6_dx0 = du6_dx6*dx6_dx0
    du6_dy0 = du6_dy6*dy6_dy0
    du6_dz0 = du6_dz6*dz6_dz0
    du6_dtheta1 = du6_dx6*dx6_dtheta1 + du6_dy6*dy6_dtheta1 + du6_dz6*dz6_dtheta1
    du6_dphi1 = du6_dx6*dx6_dphi1 + du6_dy6*dy6_dphi1 + du6_dz6*dz6_dphi1
    du6_dtheta6 = du6_dx6*dx6_dtheta6 + du6_dy6*dy6_dtheta6 + du6_dz6*dz6_dtheta6
    du6_dphi6 = du6_dx6*dx6_dphi6 + du6_dy6*dy6_dphi6 + du6_dz6*dz6_dphi6
    dv6_dx0 = dv6_dx6*dx6_dx0
    dv6_dy0 = dv6_dy6*dy6_dy0
    dv6_dz0 = dv6_dz6*dz6_dz0
    dv6_dtheta1 = dv6_dx6*dx6_dtheta1 + dv6_dy6*dy6_dtheta1 + dv6_dz6*dz6_dtheta1
    dv6_dphi1 = dv6_dx6*dx6_dphi1 + dv6_dy6*dy6_dphi1 + dv6_dz6*dz6_dphi1
    dv6_dtheta6 = dv6_dx6*dx6_dtheta6 + dv6_dy6*dy6_dtheta6 + dv6_dz6*dz6_dtheta6
    dv6_dphi6 = dv6_dx6*dx6_dphi6 + dv6_dy6*dy6_dphi6 + dv6_dz6*dz6_dphi6
    ############################
    H[12,0] = du6_dx0
    H[12,2] = du6_dy0
    H[12,4] = du6_dz0
    H[12,6] = du6_dtheta1
    H[12,8] = du6_dphi1
    H[12,24] = du6_dtheta6
    H[12,26] = du6_dphi6
    H[13,0] = dv6_dx0
    H[13,2] = dv6_dy0
    H[13,4] = dv6_dz0
    H[13,6] = dv6_dtheta1
    H[13,8] = dv6_dphi1
    H[13,24] = dv6_dtheta6
    H[13,26] = dv6_dphi6

    # u_7, v_7
    du7_dx7 = fx/z_7
    du7_dy7 = s/z_7
    du7_dz7 = -(fx*x_7+s*y_7)/z_7**2
    dv7_dx7 = 0
    dv7_dy7 = fy/z_7
    dv7_dz7 = -fy*y_7/z_7**2
    dx7_dx0 = dx6_dx0
    dx7_dtheta1 = dx6_dtheta1
    dx7_dphi1 = dx6_dphi1
    dx7_dtheta6 = dx6_dtheta6
    dx7_dphi6 = dx6_dphi6
    dx7_dtheta7 = r_7*cos(theta_7)*cos(phi_7)
    dx7_dphi7 = -r_7*sin(theta_7)*sin(phi_7)
    dy7_dy0 = dy6_dy0
    dy7_dtheta1 = dy6_dtheta1
    dy7_dphi1 = dy6_dphi1
    dy7_dtheta6 = dy6_dtheta6
    dy7_dphi6 = dy6_dphi6
    dy7_dtheta7 = r_7*cos(theta_7)*sin(phi_7)
    dy7_dphi7 = r_7*sin(theta_7)*cos(phi_7)
    dz7_dz0 = dz6_dz0
    dz7_dtheta1 = dz6_dtheta1
    dz7_dphi1 = dz6_dphi1
    dz7_dtheta6 = dz6_dtheta6
    dz7_dphi6 = dz6_dphi6
    dz7_dtheta7 = -r_7*sin(theta_7)
    dz7_dphi7 = 0
    ############################
    du7_dx0 = du7_dx7*dx7_dx0
    du7_dy0 = du7_dy7*dy7_dy0
    du7_dz0 = du7_dz7*dz7_dz0
    du7_dtheta1 = du7_dx7*dx7_dtheta1 + du7_dy7*dy7_dtheta1 + du7_dz7*dz7_dtheta1
    du7_dphi1 = du7_dx7*dx7_dphi1 + du7_dy7*dy7_dphi1 + du7_dz7*dz7_dphi1
    du7_dtheta6 = du7_dx7*dx7_dtheta6 + du7_dy7*dy7_dtheta6 + du7_dz7*dz7_dtheta6
    du7_dphi6 = du7_dx7*dx7_dphi6 + du7_dy7*dy7_dphi6 + du7_dz7*dz7_dphi6
    du7_dtheta7 = du7_dx7*dx7_dtheta7 + du7_dy7*dy7_dtheta7 + du7_dz7*dz7_dtheta7
    du7_dphi7 = du7_dx7*dx7_dphi7 + du7_dy7*dy7_dphi7 + du7_dz7*dz7_dphi7
    dv7_dx0 = dv7_dx7*dx7_dx0
    dv7_dy0 = dv7_dy7*dy7_dy0
    dv7_dz0 = dv7_dz7*dz7_dz0
    dv7_dtheta1 = dv7_dx7*dx7_dtheta1 + dv7_dy7*dy7_dtheta1 + dv7_dz7*dz7_dtheta1
    dv7_dphi1 = dv7_dx7*dx7_dphi1 + dv7_dy7*dy7_dphi1 + dv7_dz7*dz7_dphi1
    dv7_dtheta6 = dv7_dx7*dx7_dtheta6 + dv7_dy7*dy7_dtheta6 + dv7_dz7*dz7_dtheta6
    dv7_dphi6 = dv7_dx7*dx7_dphi6 + dv7_dy7*dy7_dphi6 + dv7_dz7*dz7_dphi6
    dv7_dtheta7 = dv7_dx7*dx7_dtheta7 + dv7_dy7*dy7_dtheta7 + dv7_dz7*dz7_dtheta7
    dv7_dphi7 = dv7_dx7*dx7_dphi7 + dv7_dy7*dy7_dphi7 + dv7_dz7*dz7_dphi7
    ############################
    H[14,0] = du7_dx0
    H[14,2] = du7_dy0
    H[14,4] = du7_dz0
    H[14,6] = du7_dtheta1
    H[14,8] = du7_dphi1
    H[14,24] = du7_dtheta6
    H[14,26] = du7_dphi6
    H[14,28] = du7_dtheta7
    H[14,30] = du7_dphi7
    H[15,0] = dv7_dx0
    H[15,2] = dv7_dy0
    H[15,4] = dv7_dz0
    H[15,6] = dv7_dtheta1
    H[15,8] = dv7_dphi1
    H[15,24] = dv7_dtheta6
    H[15,26] = dv7_dphi6
    H[15,28] = dv7_dtheta7
    H[15,30] = dv7_dphi7

    # u_8, v_8
    du8_dx8 = fx/z_8
    du8_dy8 = s/z_8
    du8_dz8 = -(fx*x_8+s*y_8)/z_8**2
    dv8_dx8 = 0
    dv8_dy8 = fy/z_8
    dv8_dz8 = -fy*y_8/z_8**2
    dx8_dx0 = dx2_dx0
    dx8_dtheta1 = dx2_dtheta1
    dx8_dphi1 = dx2_dphi1
    dx8_dtheta2 = dx2_dtheta2
    dx8_dphi2 = dx2_dphi2
    dx8_dtheta8 = r_8*cos(theta_8)*cos(phi_8)
    dx8_dphi8 = -r_8*sin(theta_8)*sin(phi_8)
    dy8_dy0 = dy2_dy0
    dy8_dtheta1 = dy2_dtheta1
    dy8_dphi1 = dy2_dphi1
    dy8_dtheta2 = dy2_dtheta2
    dy8_dphi2 = dy2_dphi2
    dy8_dtheta8 = r_8*cos(theta_8)*sin(phi_8)
    dy8_dphi8 = r_8*sin(theta_8)*cos(phi_8)
    dz8_dz0 = dz2_dz0
    dz8_dtheta1 = dz2_dtheta1
    dz8_dphi1 = dz2_dphi1
    dz8_dtheta2 = dz2_dtheta2
    dz8_dphi2 = dz2_dphi2
    dz8_dtheta8 = -r_8*sin(theta_8)
    dz8_dphi8 = 0
    ############################
    du8_dx0 = du8_dx8*dx8_dx0
    du8_dy0 = du8_dy8*dy8_dy0
    du8_dz0 = du8_dz8*dz8_dz0
    du8_dtheta1 = du8_dx8*dx8_dtheta1 + du8_dy8*dy8_dtheta1 + du8_dz8*dz8_dtheta1
    du8_dphi1 = du8_dx8*dx8_dphi1 + du8_dy8*dy8_dphi1 + du8_dz8*dz8_dphi1
    du8_dtheta2 = du8_dx8*dx8_dtheta2 + du8_dy8*dy8_dtheta2 + du8_dz8*dz8_dtheta2
    du8_dphi2 = du8_dx8*dx8_dphi2 + du8_dy8*dy8_dphi2 + du8_dz8*dz8_dphi2
    du8_dtheta8 = du8_dx8*dx8_dtheta8 + du8_dy8*dy8_dtheta8 + du8_dz8*dz8_dtheta8
    du8_dphi8 = du8_dx8*dx8_dphi8 + du8_dy8*dy8_dphi8 + du8_dz8*dz8_dphi8
    dv8_dx0 = dv8_dx8*dx8_dx0
    dv8_dy0 = dv8_dy8*dy8_dy0
    dv8_dz0 = dv8_dz8*dz8_dz0
    dv8_dtheta1 = dv8_dx8*dx8_dtheta1 + dv8_dy8*dy8_dtheta1 + dv8_dz8*dz8_dtheta1
    dv8_dphi1 = dv8_dx8*dx8_dphi1 + dv8_dy8*dy8_dphi1 + dv8_dz8*dz8_dphi1
    dv8_dtheta2 = dv8_dx8*dx8_dtheta2 + dv8_dy8*dy8_dtheta2 + dv8_dz8*dz8_dtheta2
    dv8_dphi2 = dv8_dx8*dx8_dphi2 + dv8_dy8*dy8_dphi2 + dv8_dz8*dz8_dphi2
    dv8_dtheta8 = dv8_dx8*dx8_dtheta8 + dv8_dy8*dy8_dtheta8 + dv8_dz8*dz8_dtheta8
    dv8_dphi8 = dv8_dx8*dx8_dphi8 + dv8_dy8*dy8_dphi8 + dv8_dz8*dz8_dphi8
    ############################
    H[16,0] = du8_dx0
    H[16,2] = du8_dy0
    H[16,4] = du8_dz0
    H[16,6] = du8_dtheta1
    H[16,8] = du8_dphi1
    H[16,10] = du8_dtheta2
    H[16,12] = du8_dphi2
    H[16,32] = du8_dtheta8
    H[16,34] = du8_dphi8
    H[17,0] = dv8_dx0
    H[17,2] = dv8_dy0
    H[17,4] = dv8_dz0
    H[17,6] = dv8_dtheta1
    H[17,8] = dv8_dphi1
    H[17,10] = dv8_dtheta2
    H[17,12] = dv8_dphi2
    H[17,32] = dv8_dtheta8
    H[17,34] = dv8_dphi8

    # u_9, v_9
    du9_dx9 = fx/z_9
    du9_dy9 = s/z_9
    du9_dz9 = -(fx*x_9+s*y_9)/z_9**2
    dv9_dx9 = 0
    dv9_dy9 = fy/z_9
    dv9_dz9 = -fy*y_9/z_9**2
    dx9_dx0 = dx8_dx0
    dx9_dtheta1 = dx8_dtheta1
    dx9_dphi1 = dx8_dphi1
    dx9_dtheta2 = dx8_dtheta2
    dx9_dphi2 = dx8_dphi2
    dx9_dtheta8 = dx8_dtheta8
    dx9_dphi8 = dx8_dphi8
    dx9_dtheta9 = r_9*cos(theta_9)*cos(phi_9)
    dx9_dphi9 = -r_9*sin(theta_9)*sin(phi_9)
    dy9_dy0 = dy8_dy0
    dy9_dtheta1 = dy8_dtheta1
    dy9_dphi1 = dy8_dphi1
    dy9_dtheta2 = dy8_dtheta2
    dy9_dphi2 = dy8_dphi2
    dy9_dtheta8 = dy8_dtheta8
    dy9_dphi8 = dy8_dphi8
    dy9_dtheta9 = r_9*cos(theta_9)*sin(phi_9)
    dy9_dphi9 = r_9*sin(theta_9)*cos(phi_9)
    dz9_dz0 = dz8_dz0
    dz9_dtheta1 = dz8_dtheta1
    dz9_dphi1 = dz8_dphi1
    dz9_dtheta2 = dz8_dtheta2
    dz9_dphi2 = dz8_dphi2
    dz9_dtheta8 = dz8_dtheta8
    dz9_dphi8 = dz8_dphi8
    dz9_dtheta9 = -r_9*sin(theta_9)
    dz9_dphi9 = 0
    ############################
    du9_dx0 = du9_dx9*dx9_dx0
    du9_dy0 = du9_dy9*dy9_dy0
    du9_dz0 = du9_dz9*dz9_dz0
    du9_dtheta1 = du9_dx9*dx9_dtheta1 + du9_dy9*dy9_dtheta1 + du9_dz9*dz9_dtheta1
    du9_dphi1 = du9_dx9*dx9_dphi1 + du9_dy9*dy9_dphi1 + du9_dz9*dz9_dphi1
    du9_dtheta2 = du9_dx9*dx9_dtheta2 + du9_dy9*dy9_dtheta2 + du9_dz9*dz9_dtheta2
    du9_dphi2 = du9_dx9*dx9_dphi2 + du9_dy9*dy9_dphi2 + du9_dz9*dz9_dphi2
    du9_dtheta8 = du9_dx9*dx9_dtheta8 + du9_dy9*dy9_dtheta8 + du9_dz9*dz9_dtheta8
    du9_dphi8 = du9_dx9*dx9_dphi8 + du9_dy9*dy9_dphi8 + du9_dz9*dz9_dphi8
    du9_dtheta9 = du9_dx9*dx9_dtheta9 + du9_dy9*dy9_dtheta9 + du9_dz9*dz9_dtheta9
    du9_dphi9 = du9_dx9*dx9_dphi9 + du9_dy9*dy9_dphi9 + du9_dz9*dz9_dphi9
    dv9_dx0 = dv9_dx9*dx9_dx0
    dv9_dy0 = dv9_dy9*dy9_dy0
    dv9_dz0 = dv9_dz9*dz9_dz0
    dv9_dtheta1 = dv9_dx9*dx9_dtheta1 + dv9_dy9*dy9_dtheta1 + dv9_dz9*dz9_dtheta1
    dv9_dphi1 = dv9_dx9*dx9_dphi1 + dv9_dy9*dy9_dphi1 + dv9_dz9*dz9_dphi1
    dv9_dtheta2 = dv9_dx9*dx9_dtheta2 + dv9_dy9*dy9_dtheta2 + dv9_dz9*dz9_dtheta2
    dv9_dphi2 = dv9_dx9*dx9_dphi2 + dv9_dy9*dy9_dphi2 + dv9_dz9*dz9_dphi2
    dv9_dtheta8 = dv9_dx9*dx9_dtheta8 + dv9_dy9*dy9_dtheta8 + dv9_dz9*dz9_dtheta8
    dv9_dphi8 = dv9_dx9*dx9_dphi8 + dv9_dy9*dy9_dphi8 + dv9_dz9*dz9_dphi8
    dv9_dtheta9 = dv9_dx9*dx9_dtheta9 + dv9_dy9*dy9_dtheta9 + dv9_dz9*dz9_dtheta9
    dv9_dphi9 = dv9_dx9*dx9_dphi9 + dv9_dy9*dy9_dphi9 + dv9_dz9*dz9_dphi9
    ############################
    H[18,0] = du9_dx0
    H[18,2] = du9_dy0
    H[18,4] = du9_dz0
    H[18,6] = du9_dtheta1
    H[18,8] = du9_dphi1
    H[18,10] = du9_dtheta2
    H[18,12] = du9_dphi2
    H[18,32] = du9_dtheta8
    H[18,34] = du9_dphi8
    H[18,36] = du9_dtheta9
    H[18,38] = du9_dphi9
    H[19,0] = dv9_dx0
    H[19,2] = dv9_dy0
    H[19,4] = dv9_dz0
    H[19,6] = dv9_dtheta1
    H[19,8] = dv9_dphi1
    H[19,10] = dv9_dtheta2
    H[19,12] = dv9_dphi2
    H[19,32] = dv9_dtheta8
    H[19,34] = dv9_dphi8
    H[19,36] = dv9_dtheta9
    H[19,38] = dv9_dphi9

    # u_10, v_10
    du10_dx10 = fx/z_10
    du10_dy10 = s/z_10
    du10_dz10 = -(fx*x_10+s*y_10)/z_10**2
    dv10_dx10 = 0
    dv10_dy10 = fy/z_10
    dv10_dz10 = -fy*y_10/z_10**2
    dx10_dx0 = dx3_dx0
    dx10_dtheta1 = dx3_dtheta1
    dx10_dphi1 = dx3_dphi1
    dx10_dtheta2 = dx3_dtheta2
    dx10_dphi2 = dx3_dphi2
    dx10_dtheta3 = dx3_dtheta3
    dx10_dphi3 = dx3_dphi3
    dx10_dtheta10 = r_10*cos(theta_10)*cos(phi_10)
    dx10_dphi10 = -r_10*sin(theta_10)*sin(phi_10)
    dy10_dy0 = dy3_dy0
    dy10_dtheta1 = dy3_dtheta1
    dy10_dphi1 = dy3_dphi1
    dy10_dtheta2 = dy3_dtheta2
    dy10_dphi2 = dy3_dphi2
    dy10_dtheta3 = dy3_dtheta3
    dy10_dphi3 = dy3_dphi3
    dy10_dtheta10 = r_10*cos(theta_10)*sin(phi_10)
    dy10_dphi10 = r_10*sin(theta_10)*cos(phi_10)
    dz10_dz0 = dz3_dz0
    dz10_dtheta1 = dz3_dtheta1
    dz10_dphi1 = dz3_dphi1
    dz10_dtheta2 = dz3_dtheta2
    dz10_dphi2 = dz3_dphi2
    dz10_dtheta3 = dz3_dtheta3
    dz10_dphi3 = dz3_dphi3
    dz10_dtheta10 = -r_10*sin(theta_10)
    dz10_dphi10 = 0
    ############################
    du10_dx0 = du10_dx10*dx10_dx0
    du10_dy0 = du10_dy10*dy10_dy0
    du10_dz0 = du10_dz10*dz10_dz0
    du10_dtheta1 = du10_dx10*dx10_dtheta1 + du10_dy10*dy10_dtheta1 + du10_dz10*dz10_dtheta1
    du10_dphi1 = du10_dx10*dx10_dphi1 + du10_dy10*dy10_dphi1 + du10_dz10*dz10_dphi1
    du10_dtheta2 = du10_dx10*dx10_dtheta2 + du10_dy10*dy10_dtheta2 + du10_dz10*dz10_dtheta2
    du10_dphi2 = du10_dx10*dx10_dphi2 + du10_dy10*dy10_dphi2 + du10_dz10*dz10_dphi2
    du10_dtheta3 = du10_dx10*dx10_dtheta3 + du10_dy10*dy10_dtheta3 + du10_dz10*dz10_dtheta3
    du10_dphi3 = du10_dx10*dx10_dphi3 + du10_dy10*dy10_dphi3 + du10_dz10*dz10_dphi3
    du10_dtheta10 = du10_dx10*dx10_dtheta10 + du10_dy10*dy10_dtheta10 + du10_dz10*dz10_dtheta10
    du10_dphi10 = du10_dx10*dx10_dphi10 + du10_dy10*dy10_dphi10 + du10_dz10*dz10_dphi10
    dv10_dx0 = dv10_dx10*dx10_dx0
    dv10_dy0 = dv10_dy10*dy10_dy0
    dv10_dz0 = dv10_dz10*dz10_dz0
    dv10_dtheta1 = dv10_dx10*dx10_dtheta1 + dv10_dy10*dy10_dtheta1 + dv10_dz10*dz10_dtheta1
    dv10_dphi1 = dv10_dx10*dx10_dphi1 + dv10_dy10*dy10_dphi1 + dv10_dz10*dz10_dphi1
    dv10_dtheta2 = dv10_dx10*dx10_dtheta2 + dv10_dy10*dy10_dtheta2 + dv10_dz10*dz10_dtheta2
    dv10_dphi2 = dv10_dx10*dx10_dphi2 + dv10_dy10*dy10_dphi2 + dv10_dz10*dz10_dphi2
    dv10_dtheta3 = dv10_dx10*dx10_dtheta3 + dv10_dy10*dy10_dtheta3 + dv10_dz10*dz10_dtheta3
    dv10_dphi3 = dv10_dx10*dx10_dphi3 + dv10_dy10*dy10_dphi3 + dv10_dz10*dz10_dphi3
    dv10_dtheta10 = dv10_dx10*dx10_dtheta10 + dv10_dy10*dy10_dtheta10 + dv10_dz10*dz10_dtheta10
    dv10_dphi10 = dv10_dx10*dx10_dphi10 + dv10_dy10*dy10_dphi10 + dv10_dz10*dz10_dphi10
    ############################
    H[20,0] = du10_dx0
    H[20,2] = du10_dy0
    H[20,4] = du10_dz0
    H[20,6] = du10_dtheta1
    H[20,8] = du10_dphi1
    H[20,10] = du10_dtheta2
    H[20,12] = du10_dphi2
    H[20,14] = du10_dtheta3
    H[20,40] = du10_dtheta10
    H[20,42] = du10_dphi10
    H[21,0] = dv10_dx0
    H[21,2] = dv10_dy0
    H[21,4] = dv10_dz0
    H[21,6] = dv10_dtheta1
    H[21,8] = dv10_dphi1
    H[21,10] = dv10_dtheta2
    H[21,12] = dv10_dphi2
    H[21,14] = dv10_dtheta3
    H[21,40] = dv10_dtheta10
    H[21,42] = dv10_dphi10

    # u_11, v_11
    du11_dx11 = fx/z_11
    du11_dy11 = s/z_11
    du11_dz11 = -(fx*x_11+s*y_11)/z_11**2
    dv11_dx11 = 0
    dv11_dy11 = fy/z_11
    dv11_dz11 = -fy*y_11/z_11**2
    dx11_dx0 = dx10_dx0
    dx11_dtheta1 = dx10_dtheta1
    dx11_dphi1 = dx10_dphi1
    dx11_dtheta2 = dx10_dtheta2
    dx11_dphi2 = dx10_dphi2
    dx11_dtheta3 = dx10_dtheta3
    dx11_dphi3 = dx10_dphi3
    dx11_dtheta10 = dx10_dtheta10
    dx11_dphi10 = dx10_dphi10
    dx11_dtheta11 = r_11*cos(theta_11)*cos(phi_11)
    dx11_dphi11 = -r_11*sin(theta_11)*sin(phi_11)
    dy11_dy0 = dy10_dy0
    dy11_dtheta1 = dy10_dtheta1
    dy11_dphi1 = dy10_dphi1
    dy11_dtheta2 = dy10_dtheta2
    dy11_dphi2 = dy10_dphi2
    dy11_dtheta3 = dy10_dtheta3
    dy11_dphi3 = dy10_dphi3
    dy11_dtheta10 = dy10_dtheta10
    dy11_dphi10 = dy10_dphi10
    dy11_dtheta11 = r_11*cos(theta_11)*sin(phi_11)
    dy11_dphi11 = r_11*sin(theta_11)*cos(phi_11)
    dz11_dz0 = dz10_dz0
    dz11_dtheta1 = dz10_dtheta1
    dz11_dphi1 = dz10_dphi1
    dz11_dtheta2 = dz10_dtheta2
    dz11_dphi2 = dz10_dphi2
    dz11_dtheta3 = dz10_dtheta3
    dz11_dphi3 = dz10_dphi3
    dz11_dtheta10 = dz10_dtheta10
    dz11_dphi10 = dz10_dphi10
    dz11_dtheta11 = -r_11*sin(theta_11)
    dz11_dphi11 = 0
    ############################
    du11_dx0 = du11_dx11*dx11_dx0
    du11_dy0 = du11_dy11*dy11_dy0
    du11_dz0 = du11_dz11*dz11_dz0
    du11_dtheta1 = du11_dx11*dx11_dtheta1 + du11_dy11*dy11_dtheta1 + du11_dz11*dz11_dtheta1
    du11_dphi1 = du11_dx11*dx11_dphi1 + du11_dy11*dy11_dphi1 + du11_dz11*dz11_dphi1
    du11_dtheta2 = du11_dx11*dx11_dtheta2 + du11_dy11*dy11_dtheta2 + du11_dz11*dz11_dtheta2
    du11_dphi2 = du11_dx11*dx11_dphi2 + du11_dy11*dy11_dphi2 + du11_dz11*dz11_dphi2
    du11_dtheta3 = du11_dx11*dx11_dtheta3 + du11_dy11*dy11_dtheta3 + du11_dz11*dz11_dtheta3
    du11_dphi3 = du11_dx11*dx11_dphi3 + du11_dy11*dy11_dphi3 + du11_dz11*dz11_dphi3
    du11_dtheta10 = du11_dx11*dx11_dtheta10 + du11_dy11*dy11_dtheta10 + du11_dz11*dz11_dtheta10
    du11_dphi10 = du11_dx11*dx11_dphi10 + du11_dy11*dy11_dphi10 + du11_dz11*dz11_dphi10
    du11_dtheta11 = du11_dx11*dx11_dtheta11 + du11_dy11*dy11_dtheta11 + du11_dz11*dz11_dtheta11
    du11_dphi11 = du11_dx11*dx11_dphi11 + du11_dy11*dy11_dphi11 + du11_dz11*dz11_dphi11
    dv11_dx0 = dv11_dx11*dx11_dx0
    dv11_dy0 = dv11_dy11*dy11_dy0
    dv11_dz0 = dv11_dz11*dz11_dz0
    dv11_dtheta1 = dv11_dx11*dx11_dtheta1 + dv11_dy11*dy11_dtheta1 + dv11_dz11*dz11_dtheta1
    dv11_dphi1 = dv11_dx11*dx11_dphi1 + dv11_dy11*dy11_dphi1 + dv11_dz11*dz11_dphi1
    dv11_dtheta2 = dv11_dx11*dx11_dtheta2 + dv11_dy11*dy11_dtheta2 + dv11_dz11*dz11_dtheta2
    dv11_dphi2 = dv11_dx11*dx11_dphi2 + dv11_dy11*dy11_dphi2 + dv11_dz11*dz11_dphi2
    dv11_dtheta3 = dv11_dx11*dx11_dtheta3 + dv11_dy11*dy11_dtheta3 + dv11_dz11*dz11_dtheta3
    dv11_dphi3 = dv11_dx11*dx11_dphi3 + dv11_dy11*dy11_dphi3 + dv11_dz11*dz11_dphi3
    dv11_dtheta10 = dv11_dx11*dx11_dtheta10 + dv11_dy11*dy11_dtheta10 + dv11_dz11*dz11_dtheta10
    dv11_dphi10 = dv11_dx11*dx11_dphi10 + dv11_dy11*dy11_dphi10 + dv11_dz11*dz11_dphi10
    dv11_dtheta11 = dv11_dx11*dx11_dtheta11 + dv11_dy11*dy11_dtheta11 + dv11_dz11*dz11_dtheta11
    dv11_dphi11 = dv11_dx11*dx11_dphi11 + dv11_dy11*dy11_dphi11 + dv11_dz11*dz11_dphi11
    ############################
    H[22,0] = du11_dx0
    H[22,2] = du11_dy0
    H[22,4] = du11_dz0
    H[22,6] = du11_dtheta1
    H[22,8] = du11_dphi1
    H[22,10] = du11_dtheta2
    H[22,12] = du11_dphi2
    H[22,14] = du11_dtheta3
    H[22,40] = du11_dtheta10
    H[22,42] = du11_dphi10
    H[22,44] = du11_dtheta11
    H[22,46] = du11_dphi11
    H[23,0] = dv11_dx0
    H[23,2] = dv11_dy0
    H[23,4] = dv11_dz0
    H[23,6] = dv11_dtheta1
    H[23,8] = dv11_dphi1
    H[23,10] = dv11_dtheta2
    H[23,12] = dv11_dphi2
    H[23,14] = dv11_dtheta3
    H[23,40] = dv11_dtheta10
    H[23,42] = dv11_dphi10
    H[23,44] = dv11_dtheta11
    H[23,46] = dv11_dphi11

    # c_0, c_1, c_2
    c_0_hat = sin(theta_1)*sin(phi_1)*cos(theta_3) - cos(theta_1)*sin(theta_3)*sin(phi_3)
    c_1_hat = cos(theta_1)*sin(theta_3)*cos(phi_3) - sin(theta_1)*cos(phi_1)*cos(theta_3)
    c_2_hat = sin(theta_1)*sin(theta_3)*(cos(phi_1)*sin(phi_3) - sin(phi_1)*cos(phi_3))
    ############################
    dc0_dtheta1 = cos(theta_1)*sin(phi_1)*cos(theta_3) + sin(theta_1)*sin(theta_3)*sin(phi_3)
    dc0_dphi1 = sin(theta_1)*cos(phi_1)*cos(theta_3)
    dc0_dtheta3 = - sin(theta_1)*sin(phi_1)*sin(theta_3) - cos(theta_1)*cos(theta_3)*sin(phi_3)
    dc0_dphi3 = - cos(theta_1)*sin(theta_3)*cos(phi_3)
    dc1_dtheta1 = - sin(theta_1)*sin(theta_3)*cos(phi_3) - cos(theta_1)*cos(phi_1)*cos(theta_3)
    dc1_dphi1 = sin(theta_1)*sin(phi_1)*cos(theta_3)
    dc1_dtheta3 = cos(theta_1)*cos(theta_3)*cos(phi_3) + sin(theta_1)*cos(phi_1)*sin(theta_3)
    dc1_dphi3 = - cos(theta_1)*sin(theta_3)*sin(phi_3)
    dc2_dtheta1 = cos(theta_1)*sin(theta_3)*(cos(phi_1)*sin(phi_3) - sin(phi_1)*cos(phi_3))
    dc2_dphi1 = sin(theta_1)*sin(theta_3)*(- sin(phi_1)*sin(phi_3) - cos(phi_1)*cos(phi_3))
    dc2_dtheta3 = sin(theta_1)*cos(theta_3)*(cos(phi_1)*sin(phi_3) - sin(phi_1)*cos(phi_3))
    dc2_dphi3 = sin(theta_1)*sin(theta_3)*(cos(phi_1)*cos(phi_3) + sin(phi_1)*sin(phi_3))
    ############################
    H[24,6] = dc0_dtheta1
    H[24,8] = dc0_dphi1
    H[24,14] = dc0_dtheta3
    H[25,6] = dc1_dtheta1
    H[25,8] = dc1_dphi1
    H[25,14] = dc1_dtheta3
    H[26,6] = dc2_dtheta1
    H[26,8] = dc2_dphi1
    H[26,14] = dc2_dtheta3

    # d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9, d_10, d_11
    baseline = 50.0*1e-3 # 50.0mm
    focal = 435 # 1.93mm
    d_0_hat = baseline * focal / z_0
    d_1_hat = baseline * focal / z_1
    d_2_hat = baseline * focal / z_2
    d_3_hat = baseline * focal / z_3
    d_4_hat = baseline * focal / z_4
    d_5_hat = baseline * focal / z_5
    d_6_hat = baseline * focal / z_6
    d_7_hat = baseline * focal / z_7
    d_8_hat = baseline * focal / z_8
    d_9_hat = baseline * focal / z_9
    d_10_hat = baseline * focal / z_10
    d_11_hat = baseline * focal / z_11
    ############################
    dd0_dz0 = - baseline * focal / z_0**2
    dd1_dz1 = - baseline * focal / z_1**2
    dd2_dz2 = - baseline * focal / z_2**2
    dd3_dz3 = - baseline * focal / z_3**2
    dd4_dz4 = - baseline * focal / z_4**2
    dd5_dz5 = - baseline * focal / z_5**2
    dd6_dz6 = - baseline * focal / z_6**2
    dd7_dz7 = - baseline * focal / z_7**2
    dd8_dz8 = - baseline * focal / z_8**2
    dd9_dz9 = - baseline * focal / z_9**2
    dd10_dz10 = - baseline * focal / z_10**2
    dd11_dz11 = - baseline * focal / z_11**2
    ############################
    H[27,4] = dd0_dz0*1 # dz0_dz0
    H[28,4] = dd1_dz1*dz1_dz0
    H[28,6] = dd1_dz1*dz1_dtheta1
    H[28,8] = dd1_dz1*dz1_dphi1
    H[29,4] = dd2_dz2*dz2_dz0
    H[29,6] = dd2_dz2*dz2_dtheta1
    H[29,8] = dd2_dz2*dz2_dphi1
    H[29,10] = dd2_dz2*dz2_dtheta2
    H[29,12] = dd2_dz2*dz2_dphi2
    H[30,4] = dd3_dz3*dz3_dz0
    H[30,6] = dd3_dz3*dz3_dtheta1
    H[30,8] = dd3_dz3*dz3_dphi1
    H[30,10] = dd3_dz3*dz3_dtheta2
    H[30,12] = dd3_dz3*dz3_dphi2
    H[30,14] = dd3_dz3*dz3_dtheta3
    H[31,4] = dd4_dz4*dz4_dz0
    H[31,16] = dd4_dz4*dz4_dtheta4
    H[31,18] = dd4_dz4*dz4_dphi4
    H[32,4] = dd5_dz5*dz5_dz0
    H[32,16] = dd5_dz5*dz5_dtheta4
    H[32,18] = dd5_dz5*dz5_dphi4
    H[32,20] = dd5_dz5*dz5_dtheta5
    H[32,22] = dd5_dz5*dz5_dphi5
    H[33,4] = dd6_dz6*dz6_dz0
    H[33,6] = dd6_dz6*dz6_dtheta1
    H[33,8] = dd6_dz6*dz6_dphi1
    H[33,24] = dd6_dz6*dz6_dtheta6
    H[33,26] = dd6_dz6*dz6_dphi6
    H[34,4] = dd7_dz7*dz7_dz0
    H[34,6] = dd7_dz7*dz7_dtheta1
    H[34,8] = dd7_dz7*dz7_dphi1
    H[34,24] = dd7_dz7*dz7_dtheta6
    H[34,26] = dd7_dz7*dz7_dphi6
    H[34,28] = dd7_dz7*dz7_dtheta7
    H[34,30] = dd7_dz7*dz7_dphi7
    H[35,4] = dd8_dz8*dz8_dz0
    H[35,6] = dd8_dz8*dz8_dtheta1
    H[35,8] = dd8_dz8*dz8_dphi1
    H[35,10] = dd8_dz8*dz8_dtheta2
    H[35,12] = dd8_dz8*dz8_dphi2
    H[35,32] = dd8_dz8*dz8_dtheta8
    H[35,34] = dd8_dz8*dz8_dphi8
    H[36,4] = dd9_dz9*dz9_dz0
    H[36,6] = dd9_dz9*dz9_dtheta1
    H[36,8] = dd9_dz9*dz9_dphi1
    H[36,10] = dd9_dz9*dz9_dtheta2
    H[36,12] = dd9_dz9*dz9_dphi2
    H[36,32] = dd9_dz9*dz9_dtheta8
    H[36,34] = dd9_dz9*dz9_dphi8
    H[36,36] = dd9_dz9*dz9_dtheta9
    H[36,38] = dd9_dz9*dz9_dphi9
    H[37,4] = dd10_dz10*dz10_dz0
    H[37,6] = dd10_dz10*dz10_dtheta1
    H[37,8] = dd10_dz10*dz10_dphi1
    H[37,10] = dd10_dz10*dz10_dtheta2
    H[37,12] = dd10_dz10*dz10_dphi2
    H[37,14] = dd10_dz10*dz10_dtheta3
    H[37,40] = dd10_dz10*dz10_dtheta10
    H[37,42] = dd10_dz10*dz10_dphi10
    H[38,4] = dd11_dz11*dz11_dz0
    H[38,6] = dd11_dz11*dz11_dtheta1
    H[38,8] = dd11_dz11*dz11_dphi1
    H[38,10] = dd11_dz11*dz11_dtheta2
    H[38,12] = dd11_dz11*dz11_dphi2
    H[38,14] = dd11_dz11*dz11_dtheta3
    H[38,40] = dd11_dz11*dz11_dtheta10
    H[38,42] = dd11_dz11*dz11_dphi10
    H[38,44] = dd11_dz11*dz11_dtheta11
    H[38,46] = dd11_dz11*dz11_dphi11

    z_hat = np.concatenate((uv_hat, np.array([c_0_hat, c_1_hat, c_2_hat]),
                                    np.array([d_0_hat, d_1_hat, d_2_hat, d_3_hat, d_4_hat, d_5_hat, d_6_hat, d_7_hat, d_8_hat, d_9_hat, d_10_hat, d_11_hat])))
    # print(c_0_hat, c_1_hat, c_2_hat)

    # kalman gain
    # print((H @ P @ H.T))
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    x = x_hat + K @ (z - z_hat)
    P = P - K @ H @ P
    # update state
    # x[14] += 0.1
    x_0, x_dot_0, y_0, y_dot_0, z_0, z_dot_0, theta_1, theta_dot_1, phi_1, phi_dot_1, theta_2, theta_dot_2, phi_2, phi_dot_2, theta_3, theta_dot_3, theta_4, theta_dot_4, phi_4, phi_dot_4, theta_5, theta_dot_5, phi_5, phi_dot_5, theta_6, theta_dot_6, phi_6, phi_dot_6, theta_7, theta_dot_7, phi_7, phi_dot_7, theta_8, theta_dot_8, phi_8, phi_dot_8, theta_9, theta_dot_9, phi_9, phi_dot_9, theta_10, theta_dot_10, phi_10, phi_dot_10, theta_11, theta_dot_11, phi_11, phi_dot_11 = x
    x_1, y_1, z_1 = state2pos(x, normalize=False)[1]
    x_2, y_2, z_2 = state2pos(x, normalize=False)[2]
    x_3, y_3, z_3 = state2pos(x, normalize=False)[3]
    x_4, y_4, z_4 = state2pos(x, normalize=False)[4]
    x_5, y_5, z_5 = state2pos(x, normalize=False)[5]
    x_6, y_6, z_6 = state2pos(x, normalize=False)[6]
    x_7, y_7, z_7 = state2pos(x, normalize=False)[7]
    x_8, y_8, z_8 = state2pos(x, normalize=False)[8]
    x_9, y_9, z_9 = state2pos(x, normalize=False)[9]
    x_10, y_10, z_10 = state2pos(x, normalize=False)[10]
    x_11, y_11, z_11 = state2pos(x, normalize=False)[11]
    line_x_all.append([[x_0, x_1], [(x_0+x_1)/2, x_2], [(x_0+x_1)/2, x_3], [x_0, x_4], [x_4, x_5], [x_1, x_6], [x_6, x_7], [x_2, x_8], [x_8, x_9], [x_3, x_10], [x_10, x_11]])
    line_y_all.append([[y_0, y_1], [(y_0+y_1)/2, y_2], [(y_0+y_1)/2, y_3], [y_0, y_4], [y_4, y_5], [y_1, y_6], [y_6, y_7], [y_2, y_8], [y_8, y_9], [y_3, y_10], [y_10, y_11]])
    line_z_all.append([[z_0, z_1], [(z_0+z_1)/2, z_2], [(z_0+z_1)/2, z_3], [z_0, z_4], [z_4, z_5], [z_1, z_6], [z_6, z_7], [z_2, z_8], [z_8, z_9], [z_3, z_10], [z_10, z_11]])
    print(x_0, y_0, z_0)
    # estimated u, v
    new_uv = cam_matrix @ state2pos(x).T
    # print(new_uv.T[0], u_0, v_0)
    # break

def run(t):
    # print(t)
    line_x_t = line_x_all[t]
    line_y_t = line_y_all[t]
    line_z_t = line_z_all[t]
    for line, line_x, line_y, line_z in zip(lines, line_x_t, line_y_t, line_z_t):
        line.set_data(np.array([line_x, line_y]))
        line.set_3d_properties(np.array(line_z))
    # time.sleep(0.1)
    return lines
# attach 3D axis to figure
fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(elev=-75, azim=-90)
# set axis limits & labels
ax.set_xlim3d([-1, 1])
ax.set_xlabel('X')
ax.set_ylim3d([-1, 1])
ax.set_ylabel('Y')
ax.set_zlim3d([1, 5])
ax.set_zlabel('Z')
# create animation
lines = [ax.plot([], [], [], 'royalblue', marker='o')[0] for i in range(num_edges)]
ani = animation.FuncAnimation(fig, run, np.arange(total_frames), interval=30)
FFwriter = animation.FFMpegWriter(fps=30)
ani.save('skeleton.mp4', writer=FFwriter)
plt.show()