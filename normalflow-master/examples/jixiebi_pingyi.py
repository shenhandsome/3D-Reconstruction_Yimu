import os
import numpy as np
import open3d as o3d
import cv2
import yaml
from gs_sdk.gs_reconstruct import Reconstructor
from normalflow.utils import erode_contact_mask, height2pointcloud

def test_tracking():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    model_path = os.path.join(os.path.dirname(__file__), "models", "nnmodel.pth")
    config_path = os.path.join(os.path.dirname(__file__), "configs", "gsmini.yaml")
    image_files = ["7211.png", "7212.png", "7213.png", "7214.png", "7215.png"]  # 按顺序
    #image_files = ["7211.png", "7214.png","7215.png"]  # 按顺序

#     [599.5601, 213.1957, -225.5, 180.0, -0.0, 0.0]
# [599.5601, 210.1957, -225.5, 180.0, -0.0, 0.0]
# [599.5601, 216.1957, -225.5, 180.0, -0.0, 0.0]
# [602.5601, 213.1957, -225.5, 180.0, -0.0, 0.0]
# [596.5601, 213.1957, -225.5, 180.0, -0.0, 0.0]


    # 机械臂五帧的平移位姿（单位mm），顺序与image_files一致
    poses = np.array([
        # [599.5601, 213.1957, -225.5],  # 第一帧
        # [599.5601, 210.1957, -225.5],  # 第二帧
        # [599.5601, 216.1957, -225.5],  # 第三帧
        # [602.5601, 213.1957, -225.5],  # 第四帧
        # [596.5601, 213.1957, -225.5],  # 第五帧
        [213.1957, 599.5601, -225.5],  # 第一帧
        [210.1957, 599.5601, -225.5],  # 第二帧
        [216.1957, 599.5601, -225.5],  # 第三帧
        [213.1957, 602.5601, -225.5],  # 第四帧
        [213.1957, 596.5601, -225.5],  # 第五帧
    ])





    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        ppmm = config["ppmm"]

    recon = Reconstructor(model_path, device="cpu")
    bg_image = cv2.imread(os.path.join(data_dir, "background721.png"))
    recon.load_bg(bg_image)

    # 第一帧的平移（参考系，单位m）
    t_ref = poses[0] / 1000.0

    global_points = []
    for img_file, t in zip(image_files, poses):
        img_path = os.path.join(data_dir, img_file)
        img = cv2.imread(img_path)
        G, H, C = recon.get_surface_info(img, ppmm)
        C = erode_contact_mask(C)
        pc = height2pointcloud(H, ppmm)
        pc = pc[C.reshape(-1)]
        # t_gelsight= t /1000.0+hand2cam_offset
        # # 只做平移，变换到第一帧坐标系
        # t_delta=t_gelsight-t_ref
        # pc_trans = pc + t_delta
        t_delta = (t - poses[0]) / 1000.0  
        pc_trans = pc + t_delta  
        #global_points.append(pc_trans)
        global_points.append(pc_trans)

    # 合并点云并保存
    all_points = np.concatenate(global_points, axis=0)
    print("原始点数：", all_points.shape[0])
    rounded = np.round(all_points, 4)
    unique_points = np.unique(rounded, axis=0)
    print("去重后点数：", unique_points.shape[0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(unique_points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=20))
    pcd.orient_normals_consistent_tangent_plane(100)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    inlier_cloud = pcd.select_by_index(ind)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(inlier_cloud, depth=10)
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    o3d.io.write_point_cloud(os.path.join(data_dir, "relative_pose_recon.ply"), inlier_cloud)
    o3d.io.write_triangle_mesh(os.path.join(data_dir, "relative_pose_mesh.ply"), mesh)
    print("[完成] 已保存五帧平移拼接的点云与网格。")

if __name__ == "__main__":
    test_tracking()
