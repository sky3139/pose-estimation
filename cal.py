
import numpy as np
import cdflib
# file = open("a.txt")
a=np.zeros((32,11))
np.set_printoptions(suppress=True)
def cal_k_p(Points_cali):

    ii=0
    for i in range(32):
        if (i % 2 == 0):
            a[i][0] = Points_cali[ii][0];
            a[i][1] = Points_cali[ii][1];
            a[i][2] = Points_cali[ii][2];
            a[i][3] = 1;
            a[i][8] = -Points_cali[ii][0] * Points_cali[ii][3];
            a[i][9] = -Points_cali[ii][1] * Points_cali[ii][3];
            a[i][10] = -Points_cali[ii][2] * Points_cali[ii][3];
        else:
            a[i][4] = Points_cali[ii][0];
            a[i][5] = Points_cali[ii][1];
            a[i][6] = Points_cali[ii][2];
            a[i][7] = 1;
            a[i][8] = -Points_cali[ii][0] * Points_cali[ii][4];
            a[i][9] = -Points_cali[ii][1] * Points_cali[ii][4];
            a[i][10] = -Points_cali[ii][2] * Points_cali[ii][4];
            ii+=1
    # print(a)
    u=np.zeros((32,1))
    ii = 0;
    for i in range(16):
        u[i*2] = Points_cali[ii][3];
        u[i*2 + 1] = Points_cali[ii][4]
        ii+=1;
    L =np.linalg.inv(np.dot(a.T, a))
    L=np.dot( np.dot(L,a.T),u)
    # print(L)
    _tp=pow(L[8][0], 2) + pow(L[9][0], 2) + pow(L[10][0], 2)
    u0 = (L[0][0] * L[8][0] + L[1][0]* L[9][0] + L[2][0] * L[10][0]) / _tp;
    v0 = (L[4][0] *L[8][0] +L[5][0] *L[9][0]+L[6][0] *L[10][0] )/ _tp;

    fu =np.sqrt((pow(u0 *  L[8][0] -L[0][0], 2) + pow(u0 * L[9][0] -L[1][0], 2) + pow(u0 * L[10][0] - L[2][0], 2)) / _tp);
    fv =np.sqrt((pow(v0 *  L[8][0] -L[4][0], 2) + pow(v0 * L[9][0] -L[5][0], 2) + pow(v0 * L[10][0] - L[6][0], 2)) / _tp);
    K=np.array([fu,0,u0,0,fv,v0,0,0,1]).reshape(3,3)
    # print(K)
    K_34=np.append(L,1.0).reshape(3,4)
    # print(K_34)
    pose=np.dot( np.linalg.inv(K) , K_34)/np.sqrt(_tp)
    return pose,K
    # print();


# Points_cali=[]
# for i in range(32):
#     line = file.readline().split(' ')
#     p_param = [float(i) for i in line]
#     print(p_param)
#     Points_cali.append(p_param)
# cal_k_p(Points_cali)

import pickle as pkl

if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    cdf = cdflib.CDF("/home/u20/Downloads/D3_Positions/SittingDown.cdf")
    cdf_2d=cdflib.CDF("data/D2_Positions/SittingDown.54138969.cdf")
    info = cdf.varget("Pose")
    info_2D=cdf_2d.varget("Pose")
    pose=[]
    data=dict()
    data['cam_poses']=[]
    data['jointPositions']=[]
    data['poses2d']=[]
    data['CAM_P3D']=[]
    for i, v in enumerate(info[0]):
        points_2d = info_2D[0][i].reshape(-1, 2)  # 世界坐标系的点
        # data['jointPositions'].append(v)
        points = v.reshape(-1, 3)*0.001  # 世界坐标系的点
        if len(pose)==0:
            ___a=np.hstack((points, points_2d))
            pose,K=cal_k_p(___a)
            data['cam_intrinsics']=K
        p3d=np.insert(points, 3,values=1,axis=1)
        CAM_P3D=np.dot(pose,p3d.T).T
        data['jointPositions'].append(points)
        data['poses2d'].append(points_2d)
        data['cam_poses'].append(pose)
        data['CAM_P3D'].append(CAM_P3D)
        # print(CAM_P3D)

    # print(data['jointPositions'])
    with open("data/human/test/medicine.pkl", "wb") as f:
        pkl.dump(data, f)
    with open("data/human/train/medicine.pkl", "wb") as f:
        pkl.dump(data, f)
    with open("data/human/validation/medicine.pkl", "wb") as f:
        pkl.dump(data, f)