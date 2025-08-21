######视频流关键帧点云拼接###
import argparse
import os

import cv2
import numpy as np
import open3d as o3d
import yaml

from gs_sdk.gs_reconstruct import Reconstructor
from normalflow.registration import normalflow
from normalflow.utils import erode_contact_mask, gxy2normal
from normalflow.viz_utils import annotate_coordinate_system
from normalflow.utils import height2pointcloud
from scipy.spatial.transform import Rotation as R

def test_tracking():
    parser = argparse.ArgumentParser(
        description="基于关键帧的NormalFlow点云拼接"
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cpu",
        help="The device to load and run the neural network model.",
    )
    args = parser.parse_args()

    # 加载配置
    model_path = os.path.join(os.path.dirname(__file__), "models", "nnmodel.pth")
    config_path = os.path.join(os.path.dirname(__file__), "configs", "gsmini.yaml")
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        ppmm = config["ppmm"]
        imgh = config["imgh"]
        imgw = config["imgw"]

    # 创建重建器
    recon = Reconstructor(model_path, device=args.device)
    bg_image = cv2.imread(os.path.join(data_dir, "0804background.png"))
    recon.load_bg(bg_image)

    # 读取视频帧
    video_path = os.path.join(data_dir, "bead3.avi")
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    tactile_images = []
    while True:
        ret, tactile_image = video.read()
        if not ret:
            break
        tactile_images.append(tactile_image)
    video.release()
    print("总帧数：", len(tactile_images))

    # 初始化索引
    keyframe_idx = 0  # 关键帧索引
    prev_idx = 0      # 前一帧索引
    curr_idx = 0      # 当前帧索引

    # 初始化关键帧、前一帧、当前帧的表面信息
    G_key, H_key, C_key = recon.get_surface_info(tactile_images[keyframe_idx], ppmm)
    C_key = erode_contact_mask(C_key)
    N_key = gxy2normal(G_key)
    pc_key = height2pointcloud(H_key, ppmm)
    pc_key = pc_key[C_key.reshape(-1)]

    # 关键帧点云与索引
    keyframe_pointclouds = [pc_key]
    keyframe_indices = [keyframe_idx]
    last_keyframe_pose = np.eye(4)  # 关键帧在全局的位姿
    pTk = np.eye(4)  # 前一帧到关键帧的累计变换

    # 阈值
    trans_thresh = 0.003  # 2mm
    rot_thresh = np.deg2rad(3)  # 2度

    # 记录可视化
    tracked_tactile_images = []

    # 关键帧中心坐标
    contours_key, _ = cv2.findContours(
        (C_key * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    M_key = cv2.moments(max(contours_key, key=cv2.contourArea))
    cx_key, cy_key = int(M_key["m10"] / M_key["m00"]), int(M_key["m01"] / M_key["m00"])

    for curr_idx in range(1, len(tactile_images)):
        # 当前帧
        tactile_image = tactile_images[curr_idx]
        G_curr, H_curr, C_curr = recon.get_surface_info(tactile_image, ppmm)
        C_curr = erode_contact_mask(C_curr)
        N_curr = gxy2normal(G_curr)

        # 前一帧
        G_prev, H_prev, C_prev = recon.get_surface_info(tactile_images[prev_idx], ppmm)
        C_prev = erode_contact_mask(C_prev)
        N_prev = gxy2normal(G_prev)

        # 1. 当前帧到关键帧的变换 cTk
        cTk = normalflow(
            N_key, C_key, H_key,
            N_curr, C_curr, H_curr,
            np.eye(4), ppmm
        )

        # 2. 当前帧到前一帧的变换 cTp
        cTp = normalflow(
            N_prev, C_prev, H_prev,
            N_curr, C_curr, H_curr,
            np.eye(4), ppmm
        )

        # 3. 前一帧到关键帧的历史累计变换 pTk
        # pTk = 上一轮的累计变换
        # 4. 误差判断
        # Err = (cTp)^-1 * cTk 与 pTk 的差异
        err_mat = np.linalg.inv(cTp) @ cTk
        err_pose = np.linalg.inv(pTk) @ err_mat
        trans_err = np.linalg.norm(err_pose[:3, 3])
        rot_err = R.from_matrix(err_pose[:3, :3]).magnitude()

        # 判断是否切换关键帧
        if trans_err > trans_thresh or rot_err > rot_thresh:
            # 变换当前帧点云到全局坐标系
            last_keyframe_pose = last_keyframe_pose @ np.linalg.inv(cTk)
            # last_keyframe_pose = last_keyframe_pose @ cTk
            pc_curr = height2pointcloud(H_curr, ppmm)
            pc_curr = pc_curr[C_curr.reshape(-1)]
            # 当前帧点云变换到全局
            pc_curr_obj = (last_keyframe_pose[:3, :3] @ pc_curr.T).T + last_keyframe_pose[:3, 3]

            keyframe_pointclouds.append(pc_curr_obj)
            keyframe_indices.append(curr_idx)
            print(f"切换关键帧: {curr_idx}，平移误差: {trans_err:.4f}m，旋转误差: {np.rad2deg(rot_err):.2f}°")
            # 更新关键帧
            G_key, H_key, C_key = G_curr, H_curr, C_curr
            N_key = N_curr
            pTk = np.eye(4)  # 新关键帧，累计变换重置

            # 更新关键帧中心
            contours_key, _ = cv2.findContours(
                (C_key * 255).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            M_key = cv2.moments(max(contours_key, key=cv2.contourArea))
            cx_key, cy_key = int(M_key["m10"] / M_key["m00"]), int(M_key["m01"] / M_key["m00"])
        else:
            # 累计前一帧到关键帧的变换
            pTk = pTk @ cTp

        # 更新前一帧索引
        prev_idx = curr_idx

        # 可视化
        image_l = tactile_images[keyframe_indices[-1]].copy()
        cv2.putText(
            image_l,
            f"Keyframe {keyframe_indices[-1]}",
            (20, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
        center_key = np.array([cx_key, cy_key]).astype(np.int32)
        unit_vectors_key = np.eye(3)[:, :2]
        annotate_coordinate_system(image_l, center_key, unit_vectors_key)

        image_r = tactile_image.copy()
        cv2.putText(
            image_r,
            f"Current Frame {curr_idx}",
            (20, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
        center_3d_key = (
            np.array([(cx_key - imgw / 2 + 0.5), (cy_key - imgh / 2 + 0.5), 0])
            * ppmm
            / 1000.0
        )
        unit_vectors_3d_key = np.eye(3) * ppmm / 1000.0


        remapped_center_3d_key = (
            np.dot(cTk[:3, :3], center_3d_key) + cTk[:3, 3]
        )
        remapped_cx_key = remapped_center_3d_key[0] * 1000 / ppmm + imgw / 2 - 0.5
        remapped_cy_key = remapped_center_3d_key[1] * 1000 / ppmm + imgh / 2 - 0.5
        remapped_center_key = np.array([remapped_cx_key, remapped_cy_key]).astype(
            np.int32
        )
        remapped_unit_vectors_key = (
            np.dot(cTk[:3, :3], unit_vectors_3d_key.T).T * 1000 / ppmm
        )[:, :2]
        annotate_coordinate_system(
            image_r, remapped_center_key, remapped_unit_vectors_key
        )
        tracked_tactile_images.append(cv2.hconcat([image_l, image_r]))

    # 保存可视化视频
    save_path = os.path.join(data_dir, "tracked_bead.avi")
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    video = cv2.VideoWriter(
        save_path,
        fourcc,
        fps,
        (tracked_tactile_images[0].shape[1], tracked_tactile_images[0].shape[0]),
    )
    for tracked_tactile_image in tracked_tactile_images:
        video.write(tracked_tactile_image)
    video.release()

    # 保存关键帧点云为 .ply 文件，并进行 Poisson 重建
    if len(keyframe_pointclouds) > 0:
        global_pointcloud = np.concatenate(keyframe_pointclouds, axis=0)
        print("关键帧拼接后点数：", global_pointcloud.shape[0])
        # rounded-unique方法：先四舍五入再去重，兼顾效率和效果
        decimals = 4
        rounded_points = np.round(global_pointcloud, decimals=decimals)
        unique_points = np.unique(rounded_points, axis=0)
        print(f"rounded-unique去重后点数（decimals={decimals}）：", unique_points.shape[0])

        # 创建点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(unique_points)

        # 法线估算
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=20))
        pcd.orient_normals_consistent_tangent_plane(100)

        # 移除离群点
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        inlier_cloud = pcd.select_by_index(ind)
        print(f"移除离群点后点数：{len(inlier_cloud.points)}")

        # Poisson重建
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            inlier_cloud, depth=10, width=0, scale=1.1, linear_fit=False
        )
        vertices_to_remove = densities < np.quantile(densities, 0.05)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        print(f"密度过滤后三角形数量：{len(mesh.triangles)}")

        # 保存结果
        o3d.io.write_triangle_mesh(os.path.join(data_dir, "mesh_keyframebead3.ply"), mesh)
        o3d.io.write_point_cloud(os.path.join(data_dir, "reconstructed_keyframebead3.ply"), inlier_cloud)

if __name__ == "__main__":
    test_tracking()