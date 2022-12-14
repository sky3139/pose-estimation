import cv2
import cdflib
import open3d as o3d
import numpy as np
import cal
lines_link = [[13, 12], [12, 11], [11, 0], [1, 2], [2, 3], [3, 4], [4, 5], [7, 8], [8, 9], [9, 10], [1, 0], [0, 6], [6, 7], [
    15, 14], [18, 17], [17, 13], [14, 13], [13, 25], [25, 26], [26, 27], [27, 29], [30, 27], [21, 19], [19, 18], [19, 21], [19, 22]]
if __name__ == "__main__":
    vc = cv2.VideoCapture("/home/u20/Downloads/SittingDown.54138969.mp4")  # 读入视频文件
    # 获取视频文件总帧数，并产生在总帧数范围内的随机数
    NumFrames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频的总帧数
    np.set_printoptions(suppress=True)
    cdf = cdflib.CDF("/home/u20/Downloads/D3_Positions/SittingDown.cdf")
    cdf_2d=cdflib.CDF("data/D2_Positions/SittingDown.54138969.cdf")
    # print(cdf.cdf_info())
    info = cdf.varget("Pose")
    info_2D=cdf_2d.varget("Pose")
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[
                                                                 0, 0, 0])
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='可视化', width=800, height=600)
    # vis.toggle_full_screen() #全屏
    # 设置
    point_color = (0, 0, 255)  # BGR
    opt = vis.get_render_option()
    # opt.background_color = np.asarray([0, 0, 0]) #背景
    opt.point_size = 5  # 点云大小
    lines_pcd = o3d.geometry.LineSet()
    lines_pcd.lines = o3d.utility.Vector2iVector(lines_link)
    vis.add_geometry(lines_pcd)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, origin=[0, 0, 0])
    geometry = o3d.geometry.PointCloud()
    vis.add_geometry(geometry)  # 添加源点云到显示窗口
    vis.add_geometry(mesh_frame)  # 添加源点云到显示窗口
    for i, v in enumerate(info[0]):
        # if i!=20:
        #     continue
        points_2d = info_2D[0][i].reshape(-1, 2)  # 世界坐标系的点
        # print(points_2d)
        
        vc.set(cv2.CAP_PROP_POS_FRAMES, i)  # 截取指定帧数
        rval, frame = vc.read()      # 分帧读取视频
        points = v.reshape(-1, 3)*0.001  # 世界坐标系的点
        geometry.points = o3d.utility.Vector3dVector(points)
        geometry.paint_uniform_color([1, 0, 0])  # 自定义源点云为红色
        lines_pcd.points = o3d.utility.Vector3dVector(points)
        vis.update_geometry(geometry)
        vis.update_geometry(mesh_frame)
        vis.update_geometry(lines_pcd)
        vis.poll_events()
        vis.update_renderer()
        points_2di=points_2d.astype("int")
        cv2.circle(frame, points_2di[18], 5, point_color, 2)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # for i,p2 in enumerate(points_2d):
        #     print(points[i][0],points[i][1],points[i][2],p2[0],p2[1])
        # print(points,points_2d)
        ___a=np.hstack((points, points_2d))
        # print(___a)
        pose=cal.cal_k_p(___a)
        p3d=np.insert(points, 3,values=1,axis=1)
        # print(p3d)
        CAM_P3D=np.dot(pose,p3d.T).T
        print(CAM_P3D)
        cv2.imshow("a", frame)
        cv2.waitKey(1)
    vis.run()
    vis.destroy_window()
