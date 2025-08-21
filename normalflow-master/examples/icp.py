import argparse
import os
import cv2
import numpy as np
import open3d as o3d
import yaml
from gs_sdk.gs_reconstruct import Reconstructor
from normalflow.utils import erode_contact_mask, gxy2normal, height2pointcloud
from normalflow.viz_utils import annotate_coordinate_system

def test_tracking():
    parser = argparse.ArgumentParser(description="Track the object in the tactile video.")
    parser.add_argument("-d", "--device", type=str, choices=["cuda", "cpu"], default="cpu", help="The device to load and run the neural network model.")
    args = parser.parse_args()

    model_path = os.path.join(os.path.dirname(__file__), "models", "nnmodel.pth")
    config_path = os.path.join(os.path.dirname(__file__), "configs", "gsmini.yaml")
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        ppmm = config["ppmm"]
        imgh = config["imgh"]
        imgw = config["imgw"]

    recon = Reconstructor(model_path, device=args.device)
    bg_image = cv2.imread(os.path.join(data_dir, "background.png"))
    recon.load_bg(bg_image)

    video_path = os.path.join(data_dir, "tactile_video.avi")
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    tactile_images = []
    while True:
        ret, tactile_image = video.read()
        if not ret:
            break
        tactile_images.append(tactile_image)
    video.release()

    # 第一帧点云与法线
    G_ref, H_ref, C_ref = recon.get_surface_info(tactile_images[0], ppmm)
    C_ref = erode_contact_mask(C_ref)
    pc_ref = height2pointcloud(H_ref, ppmm)
    pc_ref = pc_ref[C_ref.reshape(-1)]
    ref_pcd = o3d.geometry.PointCloud()
    ref_pcd.points = o3d.utility.Vector3dVector(pc_ref)
    ref_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    ref_pcd.orient_normals_consistent_tangent_plane(100)

    # 全局点云与位姿
    global_pointcloud = [pc_ref]
    T_list = [np.eye(4)]
    pc_list = [pc_ref]
    prev_pcd = ref_pcd  # 新增：上一帧点云
    prev_T = np.eye(4)  # 新增：上一帧累计位姿

    for idx, tactile_image in enumerate(tactile_images[1:], 1):
        G_curr, H_curr, C_curr = recon.get_surface_info(tactile_image, ppmm)
        C_curr = erode_contact_mask(C_curr)
        pc_curr = height2pointcloud(H_curr, ppmm)
        pc_curr = pc_curr[C_curr.reshape(-1)]
        curr_pcd = o3d.geometry.PointCloud()
        curr_pcd.points = o3d.utility.Vector3dVector(pc_curr)
        curr_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        curr_pcd.orient_normals_consistent_tangent_plane(100)

        # 点到面ICP配准（与上一帧）
        reg_p2l = o3d.pipelines.registration.registration_icp(
            curr_pcd, prev_pcd, 0.03, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
        # 当前帧在上一帧坐标系下的变换
        T_delta = reg_p2l.transformation
        # 当前帧在全局坐标系下的累计变换
        curr_T = prev_T @ T_delta
        T_list.append(curr_T.copy())
        # 变换当前帧点云到全局坐标系
        pc_curr_global = (curr_T[:3, :3] @ pc_curr.T).T + curr_T[:3, 3]
        global_pointcloud.append(pc_curr_global)
        pc_list.append(pc_curr)
        # 更新上一帧
        prev_pcd = curr_pcd
        prev_T = curr_T
    # 点云去重
    global_pointcloud = np.concatenate(global_pointcloud, axis=0)
    decimals = 6
    rounded_points = np.round(global_pointcloud, decimals=decimals)
    unique_points = np.unique(rounded_points, axis=0)
    print(f"去重后点数：{unique_points.shape[0]}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(unique_points)




    # 法线估算
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    # Poisson重建
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    o3d.io.write_triangle_mesh(os.path.join(data_dir, "mesh_new_icp.ply"), mesh)
    o3d.io.write_point_cloud(os.path.join(data_dir, "reconstructed_new_icp.ply"), pcd)

if __name__ == "__main__":
    test_tracking()