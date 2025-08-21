######五张图（Normalflow）##########
# import argparse
# import os

# import cv2
# import numpy as np
# import open3d as o3d
# import yaml

# from gs_sdk.gs_reconstruct import Reconstructor
# from normalflow.registration import normalflow
# from normalflow.utils import erode_contact_mask, gxy2normal
# from normalflow.viz_utils import annotate_coordinate_system
# from normalflow.utils import height2pointcloud

# """
# This script demonstrates tracking objects using NormalFlow by always referencing the first static tactile image (zhong.png).

# It loads five tactile images, uses zhong.png as the reference for all normal flow estimations, 
# then reconstructs a global point cloud and Poisson mesh.

# Usage:
#     python test_tracking.py --device {cuda, cpu}
# """

# model_path = os.path.join(os.path.dirname(__file__), "models", "nnmodel.pth")
# config_path = os.path.join(os.path.dirname(__file__), "configs", "gsmini.yaml")
# data_dir = os.path.join(os.path.dirname(__file__), "data")


# def test_tracking():
#     # Argument Parser
#     parser = argparse.ArgumentParser(description="Track objects across static tactile images using NormalFlow with fixed reference.")
#     parser.add_argument(
#         "-d",
#         "--device",
#         type=str,
#         choices=["cuda", "cpu"],
#         default="cpu",
#         help="The device to load and run the neural network model.",
#     )
#     args = parser.parse_args()

#     # Load the device configuration
#     with open(config_path, "r") as f:
#         config = yaml.safe_load(f)
#         ppmm = config["ppmm"]
#         imgh = config["imgh"]
#         imgw = config["imgw"]

#     # Create reconstructor
#     recon = Reconstructor(model_path, device=args.device)
#     # bg_image = cv2.imread(os.path.join(data_dir, "background4.png"))
#     bg_image = cv2.imread(os.path.join(data_dir, "819background.png"))
#     recon.load_bg(bg_image)

#     # Load five static images in order
#     # filenames = ["zhong.png", "zuoshang.png"]
#     filenames = ["81901.png", "81902.png", "81903.png"]
#     tactile_images = []
#     for fname in filenames:
#         img_path = os.path.join(data_dir, fname)
#         img = cv2.imread(img_path)
#         if img is None:
#             raise FileNotFoundError(f"无法读取文件: {img_path}")
#         tactile_images.append(img)
#     print(f"已成功读取 {len(tactile_images)} 张图片作为输入。")

#     # Always use zhong.png as the reference frame
#     G_ref, H_ref, C_ref = recon.get_surface_info(tactile_images[0], ppmm)
#     C_ref = erode_contact_mask(C_ref)
#     N_ref = gxy2normal(G_ref)

#     contours_ref, _ = cv2.findContours((C_ref * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     M_ref = cv2.moments(max(contours_ref, key=cv2.contourArea))
#     cx_ref, cy_ref = int(M_ref["m10"] / M_ref["m00"]), int(M_ref["m01"] / M_ref["m00"])

#     tracked_tactile_images = []
#     global_pointcloud = []
#     T_list = [np.eye(4)]
#     pc_list = []

#     # Always compute transformation relative to zhong.png
#     for i, tactile_image in enumerate(tactile_images[1:]):
#         G_curr, H_curr, C_curr = recon.get_surface_info(tactile_image, ppmm)
#         C_curr = erode_contact_mask(C_curr)
#         N_curr = gxy2normal(G_curr)

#         # IMPORTANT: always start from identity for each comparison to zhong.png
#         curr_T_ref = normalflow(N_ref, C_ref, H_ref, N_curr, C_curr, H_curr, np.eye(4), ppmm)

#         # 输出当前帧相对zhong.png的位姿
#         print(f"第{i+2}帧（{filenames[i+1]}）相对zhong.png的位姿：\n{curr_T_ref}")

#         pc_curr = height2pointcloud(H_curr, ppmm)
#         pc_curr = pc_curr[C_curr.reshape(-1)]
#         T_obj = np.linalg.inv(curr_T_ref)
#         pc_curr_obj = (T_obj[:3, :3] @ pc_curr.T).T + T_obj[:3, 3]
#         global_pointcloud.append(pc_curr_obj)
#         pc_list.append(pc_curr_obj)
#         T_list.append(curr_T_ref.copy())

#         # Annotate frames for visualization
#         image_l = tactile_images[0].copy()
#         cv2.putText(image_l, "Reference: zhong.png", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#         center_ref = np.array([cx_ref, cy_ref]).astype(np.int32)
#         unit_vectors_ref = np.eye(3)[:, :2]
#         annotate_coordinate_system(image_l, center_ref, unit_vectors_ref)

#         image_r = tactile_image.copy()
#         cv2.putText(image_r, "Current Frame", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#         center_3d_ref = np.array([(cx_ref - imgw / 2 + 0.5), (cy_ref - imgh / 2 + 0.5), 0]) * ppmm / 1000.0
#         unit_vectors_3d_ref = np.eye(3) * ppmm / 1000.0
#         remapped_center_3d_ref = np.dot(curr_T_ref[:3, :3], center_3d_ref) + curr_T_ref[:3, 3]
#         remapped_cx_ref = remapped_center_3d_ref[0] * 1000 / ppmm + imgw / 2 - 0.5
#         remapped_cy_ref = remapped_center_3d_ref[1] * 1000 / ppmm + imgh / 2 - 0.5
#         remapped_center_ref = np.array([remapped_cx_ref, remapped_cy_ref]).astype(np.int32)
#         remapped_unit_vectors_ref = (np.dot(curr_T_ref[:3, :3], unit_vectors_3d_ref.T).T * 1000 / ppmm)[:, :2]
#         annotate_coordinate_system(image_r, remapped_center_ref, remapped_unit_vectors_ref)

#         tracked_tactile_images.append(cv2.hconcat([image_l, image_r]))

#     # Save tracked visualization video
#     save_path = os.path.join(data_dir, "tracked_fixed_reference.avi")
#     fourcc = cv2.VideoWriter_fourcc(*"FFV1")
#     video = cv2.VideoWriter(
#         save_path,
#         fourcc,
#         5,
#         (tracked_tactile_images[0].shape[1], tracked_tactile_images[0].shape[0]),
#     )
#     for tracked_tactile_image in tracked_tactile_images:
#         video.write(tracked_tactile_image)
#     video.release()

#     # Save point cloud & mesh
#     if len(global_pointcloud) > 0:
#         global_pointcloud = np.concatenate(global_pointcloud, axis=0)
#         print("原始点数：", global_pointcloud.shape[0])

#         decimals = 4
#         rounded_points = np.round(global_pointcloud, decimals=decimals)
#         unique_points = np.unique(rounded_points, axis=0)
#         print(f"rounded-unique去重后点数（decimals={decimals}）：", unique_points.shape[0])

#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(unique_points)
#         pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=20))
#         pcd.orient_normals_consistent_tangent_plane(100)

#         cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
#         inlier_cloud = pcd.select_by_index(ind)
#         print(f"移除离群点后点数：{len(inlier_cloud.points)}")

#         mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(inlier_cloud, depth=10)
#         vertices_to_remove = densities < np.quantile(densities, 0.05)
#         mesh.remove_vertices_by_mask(vertices_to_remove)
#         print(f"密度过滤后三角形数量：{len(mesh.triangles)}")

#         o3d.io.write_triangle_mesh(os.path.join(data_dir, "sjw819.ply"), mesh)
#         o3d.io.write_point_cloud(os.path.join(data_dir, "sjw_mesh819.ply"), inlier_cloud)


# if __name__ == "__main__":
#     test_tracking()



###五张图ICP#######
# import argparse
# import os
# import cv2
# import numpy as np
# import open3d as o3d
# import yaml

# from gs_sdk.gs_reconstruct import Reconstructor
# from normalflow.utils import erode_contact_mask, gxy2normal, height2pointcloud

# def robust_icp_registration(source_pcd, target_pcd, max_distance=0.02):
#     """
#     鲁棒的ICP配准函数，包含粗配准和精细配准
#     """
#     print(f"源点云点数：{len(source_pcd.points)}")
#     print(f"目标点云点数：{len(target_pcd.points)}")
    
#     # 1. 降采样提高效率和鲁棒性
#     voxel_size = 0.002
#     source_down = source_pcd.voxel_down_sample(voxel_size=voxel_size)
#     target_down = target_pcd.voxel_down_sample(voxel_size=voxel_size)
    
#     # 2. 估计法线（如果没有的话）
#     if not source_down.has_normals():
#         source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
#         source_down.orient_normals_consistent_tangent_plane(100)
#     if not target_down.has_normals():
#         target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
#         target_down.orient_normals_consistent_tangent_plane(100)
    
#     # 3. FPFH特征计算
#     radius_feature = voxel_size * 5
#     source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#         source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
#     target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#         target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
#     # 4. RANSAC粗配准
#     distance_threshold = voxel_size * 1.5
#     ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
#         source_down, target_down, source_fpfh, target_fpfh, True,
#         distance_threshold,
#         o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
#         3, [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
#             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
#         o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    
#     print(f"粗配准 fitness: {ransac_result.fitness:.4f}")
#     print(f"粗配准 inlier_rmse: {ransac_result.inlier_rmse:.4f}")
    
#     # 5. 点到面ICP精细配准
#     reg_p2l = o3d.pipelines.registration.registration_icp(
#         source_pcd, target_pcd, max_distance,
#         ransac_result.transformation,  # 用粗配准结果初始化
#         o3d.pipelines.registration.TransformationEstimationPointToPlane(),
#         o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100, relative_fitness=1e-6, relative_rmse=1e-6)
#     )
    
#     print(f"精细配准 fitness: {reg_p2l.fitness:.4f}")
#     print(f"精细配准 inlier_rmse: {reg_p2l.inlier_rmse:.4f}")
    
#     # 6. 配准质量检查
#     if reg_p2l.fitness < 0.2:
#         print("警告：配准质量较差，尝试点到点ICP")
#         # 尝试点到点ICP作为备选
#         reg_p2p = o3d.pipelines.registration.registration_icp(
#             source_pcd, target_pcd, max_distance,
#             ransac_result.transformation,
#             o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#             o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
#         )
#         if reg_p2p.fitness > reg_p2l.fitness:
#             print("使用点到点ICP结果")
#             return reg_p2p
    
#     return reg_p2l

# def test_tracking():
#     parser = argparse.ArgumentParser(description="鲁棒的ICP点云拼接")
#     parser.add_argument("-d", "--device", type=str, choices=["cuda", "cpu"], default="cpu", 
#                        help="The device to load and run the neural network model.")
#     args = parser.parse_args()

#     model_path = os.path.join(os.path.dirname(__file__), "models", "nnmodel.pth")
#     config_path = os.path.join(os.path.dirname(__file__), "configs", "gsmini.yaml")
#     data_dir = os.path.join(os.path.dirname(__file__), "data")

#     with open(config_path, "r") as f:
#         config = yaml.safe_load(f)
#         ppmm = config["ppmm"]
#         imgh = config["imgh"]
#         imgw = config["imgw"]

#     recon = Reconstructor(model_path, device=args.device)
#     bg_image = cv2.imread(os.path.join(data_dir, "819background.png"))
#     recon.load_bg(bg_image)

#     # 图片列表
#     image_names = ["8191.png", "8192.png", "8193.png","8194.png", "8195.png", "8196.png","8197.png", "8198.png"]
#     tactile_images = [cv2.imread(os.path.join(data_dir, name)) for name in image_names]
#     print(f"已成功读取 {len(tactile_images)} 张图片作为输入。")

#     # 检查图片是否成功读取
#     for i, img in enumerate(tactile_images):
#         if img is None:
#             print(f"错误：无法读取图片 {image_names[i]}")
#             return

#     # 以第一张图为参考，提取点云
#     G_ref, H_ref, C_ref = recon.get_surface_info(tactile_images[0], ppmm)
#     C_ref = erode_contact_mask(C_ref)
#     pc_ref = height2pointcloud(H_ref, ppmm)
#     pc_ref = pc_ref[C_ref.reshape(-1)]
    
#     # 检查参考点云是否有效
#     if len(pc_ref) < 100:
#         print("警告：参考帧点云点数过少，可能影响配准效果")
    
#     ref_pcd = o3d.geometry.PointCloud()
#     ref_pcd.points = o3d.utility.Vector3dVector(pc_ref)
#     ref_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
#     ref_pcd.orient_normals_consistent_tangent_plane(100)

#     # 保存调试点云
#     o3d.io.write_point_cloud(os.path.join(data_dir, "debug_ref.ply"), ref_pcd)

#     # 全局点云与位姿
#     global_pointcloud = [pc_ref]
#     pose_list = [np.eye(4)]

#     # 其余帧分别与参考帧做ICP
#     for idx, tactile_image in enumerate(tactile_images[1:], 1):
#         print(f"\n处理第{idx+1}帧（{image_names[idx]}）...")
        
#         G_curr, H_curr, C_curr = recon.get_surface_info(tactile_image, ppmm)
#         C_curr = erode_contact_mask(C_curr)
#         pc_curr = height2pointcloud(H_curr, ppmm)
#         pc_curr = pc_curr[C_curr.reshape(-1)]
        
#         # 检查当前点云是否有效
#         if len(pc_curr) < 100:
#             print(f"警告：第{idx+1}帧点云点数过少({len(pc_curr)})，跳过此帧")
#             continue
            
#         curr_pcd = o3d.geometry.PointCloud()
#         curr_pcd.points = o3d.utility.Vector3dVector(pc_curr)
#         curr_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
#         curr_pcd.orient_normals_consistent_tangent_plane(100)

#         # 保存调试点云
#         o3d.io.write_point_cloud(os.path.join(data_dir, f"debug_curr_{idx}.ply"), curr_pcd)

#         # 鲁棒ICP配准
#         reg_result = robust_icp_registration(curr_pcd, ref_pcd, max_distance=0.02)
        
#         # 根据配准质量决定是否使用结果
#         if reg_result.fitness < 0.1:
#             print(f"第{idx+1}帧配准失败，使用单位矩阵")
#             T_icp = np.eye(4)
#         else:
#             T_icp = reg_result.transformation
            
#         pose_list.append(T_icp.copy())
#         print(f"第{idx+1}帧最终变换矩阵：\n{T_icp}")

#         # 变换当前帧点云到参考坐标系
#         pc_curr_global = (T_icp[:3, :3] @ pc_curr.T).T + T_icp[:3, 3]
#         global_pointcloud.append(pc_curr_global)

#     # 点云去重和后处理
#     if len(global_pointcloud) == 0:
#         print("错误：没有有效的点云数据")
#         return
        
#     global_pointcloud = np.concatenate(global_pointcloud, axis=0)
#     print(f"合并后点数：{global_pointcloud.shape[0]}")
    
#     # 多级去重策略
#     decimals = 4
#     rounded_points = np.round(global_pointcloud, decimals=decimals)
#     unique_points = np.unique(rounded_points, axis=0)
#     print(f"去重后点数：{unique_points.shape[0]}")
    
#     if len(unique_points) < 1000:
#         print("警告：去重后点数过少，降低去重精度")
#         decimals = 3
#         rounded_points = np.round(global_pointcloud, decimals=decimals)
#         unique_points = np.unique(rounded_points, axis=0)
#         print(f"重新去重后点数：{unique_points.shape[0]}")
    
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(unique_points)

#     # 改进的法线估算
#     pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=20))
#     pcd.orient_normals_consistent_tangent_plane(100)
    
#     # 统计离群点去除
#     cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
#     inlier_cloud = pcd.select_by_index(ind)
#     print(f"移除离群点后点数：{len(inlier_cloud.points)}")
    
#     if len(inlier_cloud.points) < 500:
#         print("警告：点数过少，可能影响重建质量")
    
#     # 高质量Poisson重建
#     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
#         inlier_cloud, depth=9, width=0, scale=1.1, linear_fit=False)
    
#     # 密度过滤
#     if len(densities) > 0:
#         vertices_to_remove = densities < np.quantile(densities, 0.05)
#         mesh.remove_vertices_by_mask(vertices_to_remove)
#         print(f"密度过滤后三角形数量：{len(mesh.triangles)}")
    
#     # 保存结果
#     o3d.io.write_triangle_mesh(os.path.join(data_dir, "sjw819.ply"), mesh)
#     o3d.io.write_point_cloud(os.path.join(data_dir, "sjw_mesh819.ply"), inlier_cloud)
#     print("鲁棒ICP重建完成，点云和网格已保存到data目录。")

# if __name__ == "__main__":
#     test_tracking()





#####读取图片#######
# import argparse
# import os
# import cv2
# import numpy as np
# import open3d as o3d
# import yaml
# from gs_sdk.gs_reconstruct import Reconstructor
# from normalflow.registration import normalflow
# from normalflow.utils import erode_contact_mask, gxy2normal, height2pointcloud

# def test_tracking():
#     parser = argparse.ArgumentParser(description="Align and reconstruct from 3 tactile images.")
#     parser.add_argument("-d", "--device", type=str, choices=["cuda", "cpu"], default="cpu", help="The device to load and run the neural network model.")
#     args = parser.parse_args()

#     model_path = os.path.join(os.path.dirname(__file__), "models", "nnmodel.pth")
#     config_path = os.path.join(os.path.dirname(__file__), "configs", "gsmini.yaml")
#     data_dir = os.path.join(os.path.dirname(__file__), "data")

#     with open(config_path, "r") as f:
#         config = yaml.safe_load(f)
#         ppmm = config["ppmm"]
#         imgh = config["imgh"]
#         imgw = config["imgw"]

#     recon = Reconstructor(model_path, device=args.device)
#     bg_image = cv2.imread(os.path.join(data_dir, "819background.png"))
#     recon.load_bg(bg_image)

#     # 读取两张图片
#     image_names = [ "819s01.png","819s02.png","819s03.png","819s04.png","819s05.png","819s06.png","819s07.png"]
#     tactile_images = [cv2.imread(os.path.join(data_dir, name)) for name in image_names]

#     # 第一帧初始化
#     G_ref, H_ref, C_ref = recon.get_surface_info(tactile_images[0], ppmm)
#     C_ref = erode_contact_mask(C_ref)
#     N_ref = gxy2normal(G_ref)
#     pc_ref = height2pointcloud(H_ref, ppmm)
#     pc_ref = pc_ref[C_ref.reshape(-1)]
#     T_list = [np.eye(4)]
#     pc_list = [pc_ref]
#     T_global=np.eye(4)
#     #curr_T = np.eye(4)

#     # 后续帧循环
#     for i in range(1, len(tactile_images)):
#         G_curr, H_curr, C_curr = recon.get_surface_info(tactile_images[i], ppmm)
#         C_curr = erode_contact_mask(C_curr)
#         N_curr = gxy2normal(G_curr)
#         # normalflow配准，得到当前帧相对前一帧的变换
#         #T_delta = normalflow(
#         curr_T_ref = normalflow(
#             N_ref, C_ref, H_ref,
#             N_curr, C_curr, H_curr,
#             np.eye(4), ppmm
#         )
#         # curr_T = curr_T @ T_delta
#         # T_list.append(curr_T.copy())
#         T_global=T_global@curr_T_ref
#         T_list.append(T_global.copy())
#         pc_curr = height2pointcloud(H_curr, ppmm)
#         pc_curr = pc_curr[C_curr.reshape(-1)]
#         # 变换到全局坐标系（用逆变换）
#         # T_obj = np.linalg.inv(curr_T)
#         T_obj = np.linalg.inv(T_global)
#         pc_curr_global = (T_obj[:3, :3] @ pc_curr.T).T + T_obj[:3, 3]
#         pc_list.append(pc_curr_global)
#         print(f"curr_T_ref:\n{curr_T_ref}")
#         print(f"T_global:\n{T_global}")
#         # 更新参考为当前帧
#         N_ref, C_ref, H_ref = N_curr, C_curr, H_curr

#     # 点云拼接
#     all_points = np.concatenate(pc_list, axis=0)
#     #  去重
#     decimals = 4
#     rounded_points = np.round(all_points, decimals=decimals)
#     unique_points = np.unique(rounded_points, axis=0)
#     print(f"去重后点数：{unique_points.shape[0]}")
#     pcd = o3d.geometry.PointCloud()   
#     pcd.points = o3d.utility.Vector3dVector(unique_points)
#     # 法线估算
#     pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
#     # Poisson重建
#     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
#     # o3d.io.write_triangle_mesh(os.path.join(data_dir, "mesh_png.ply"), mesh)
#     o3d.io.write_point_cloud(os.path.join(data_dir, "sjw819.ply"), pcd)

# if __name__ == "__main__":
#     test_tracking()










#####关键帧success

# import argparse
# import os

# import cv2
# import numpy as np
# import open3d as o3d
# import yaml

# from gs_sdk.gs_reconstruct import Reconstructor
# from normalflow.registration import normalflow
# from normalflow.utils import erode_contact_mask, gxy2normal
# from normalflow.viz_utils import annotate_coordinate_system
# from normalflow.utils import height2pointcloud
# from scipy.spatial.transform import Rotation as R

# def test_tracking():
#     parser = argparse.ArgumentParser(
#         description="基于关键帧的NormalFlow点云拼接"
#     )
#     parser.add_argument(
#         "-d",
#         "--device",
#         type=str,
#         choices=["cuda", "cpu"],
#         default="cpu",
#         help="The device to load and run the neural network model.",
#     )
#     args = parser.parse_args()

#     # 加载配置
#     model_path = os.path.join(os.path.dirname(__file__), "models", "nnmodel.pth")
#     config_path = os.path.join(os.path.dirname(__file__), "configs", "gsmini.yaml")
#     data_dir = os.path.join(os.path.dirname(__file__), "data")
#     with open(config_path, "r") as f:
#         config = yaml.safe_load(f)
#         ppmm = config["ppmm"]
#         imgh = config["imgh"]
#         imgw = config["imgw"]

#     # 创建重建器
#     recon = Reconstructor(model_path, device=args.device)
#     bg_image = cv2.imread(os.path.join(data_dir, "819background.png"))
#     recon.load_bg(bg_image)

#     # 读取视频帧
#     video_path = os.path.join(data_dir, "sjw819.avi")
#     video = cv2.VideoCapture(video_path)
#     fps = video.get(cv2.CAP_PROP_FPS)
#     tactile_images = []
#     while True:
#         ret, tactile_image = video.read()
#         if not ret:
#             break
#         tactile_images.append(tactile_image)
#     video.release()
#     print("总帧数：", len(tactile_images))

#     # 初始化索引
#     keyframe_idx = 0  # 关键帧索引
#     prev_idx = 0      # 前一帧索引
#     curr_idx = 0      # 当前帧索引

#     # 初始化关键帧、前一帧、当前帧的表面信息
#     G_key, H_key, C_key = recon.get_surface_info(tactile_images[keyframe_idx], ppmm)
#     C_key = erode_contact_mask(C_key)
#     N_key = gxy2normal(G_key)
#     pc_key = height2pointcloud(H_key, ppmm)
#     pc_key = pc_key[C_key.reshape(-1)]

#     # 关键帧点云与索引
#     keyframe_pointclouds = [pc_key]
#     keyframe_indices = [keyframe_idx]
#     last_keyframe_pose = np.eye(4)  # 关键帧在全局的位姿
#     pTk = np.eye(4)  # 前一帧到关键帧的累计变换

#     # 阈值
#     trans_thresh = 0.003  # 2mm
#     rot_thresh = np.deg2rad(3)  # 2度

#     # 记录可视化
#     tracked_tactile_images = []

#     # 关键帧中心坐标
#     contours_key, _ = cv2.findContours(
#         (C_key * 255).astype(np.uint8),
#         cv2.RETR_EXTERNAL,
#         cv2.CHAIN_APPROX_SIMPLE,
#     )
#     M_key = cv2.moments(max(contours_key, key=cv2.contourArea))
#     cx_key, cy_key = int(M_key["m10"] / M_key["m00"]), int(M_key["m01"] / M_key["m00"])

#     for curr_idx in range(1, len(tactile_images)):
#         # 当前帧
#         tactile_image = tactile_images[curr_idx]
#         G_curr, H_curr, C_curr = recon.get_surface_info(tactile_image, ppmm)
#         C_curr = erode_contact_mask(C_curr)
#         N_curr = gxy2normal(G_curr)

#         # 前一帧
#         G_prev, H_prev, C_prev = recon.get_surface_info(tactile_images[prev_idx], ppmm)
#         C_prev = erode_contact_mask(C_prev)
#         N_prev = gxy2normal(G_prev)

#         # 1. 当前帧到关键帧的变换 cTk
#         cTk = normalflow(
#             N_key, C_key, H_key,
#             N_curr, C_curr, H_curr,
#             np.eye(4), ppmm
#         )

#         # 2. 当前帧到前一帧的变换 cTp
#         cTp = normalflow(
#             N_prev, C_prev, H_prev,
#             N_curr, C_curr, H_curr,
#             np.eye(4), ppmm
#         )

#         # 3. 前一帧到关键帧的历史累计变换 pTk
#         # pTk = 上一轮的累计变换
#         # 4. 误差判断
#         # Err = (cTp)^-1 * cTk 与 pTk 的差异
#         err_mat = np.linalg.inv(cTp) @ cTk
#         err_pose = np.linalg.inv(pTk) @ err_mat
#         trans_err = np.linalg.norm(err_pose[:3, 3])
#         rot_err = R.from_matrix(err_pose[:3, :3]).magnitude()

#         # 判断是否切换关键帧
#         if trans_err > trans_thresh or rot_err > rot_thresh:
#             # 变换当前帧点云到全局坐标系
#             last_keyframe_pose = last_keyframe_pose @ np.linalg.inv(cTk)
#             # last_keyframe_pose = last_keyframe_pose @ cTk
#             pc_curr = height2pointcloud(H_curr, ppmm)
#             pc_curr = pc_curr[C_curr.reshape(-1)]
#             # 当前帧点云变换到全局
#             pc_curr_obj = (last_keyframe_pose[:3, :3] @ pc_curr.T).T + last_keyframe_pose[:3, 3]

#             keyframe_pointclouds.append(pc_curr_obj)
#             keyframe_indices.append(curr_idx)
#             print(f"切换关键帧: {curr_idx}，平移误差: {trans_err:.4f}m，旋转误差: {np.rad2deg(rot_err):.2f}°")
#             # 更新关键帧
#             G_key, H_key, C_key = G_curr, H_curr, C_curr
#             N_key = N_curr
#             pTk = np.eye(4)  # 新关键帧，累计变换重置

#             # 更新关键帧中心
#             contours_key, _ = cv2.findContours(
#                 (C_key * 255).astype(np.uint8),
#                 cv2.RETR_EXTERNAL,
#                 cv2.CHAIN_APPROX_SIMPLE,
#             )
#             M_key = cv2.moments(max(contours_key, key=cv2.contourArea))
#             cx_key, cy_key = int(M_key["m10"] / M_key["m00"]), int(M_key["m01"] / M_key["m00"])
#         else:
#             # 累计前一帧到关键帧的变换
#             pTk = pTk @ cTp

#         # 更新前一帧索引
#         prev_idx = curr_idx

#         # 可视化
#         image_l = tactile_images[keyframe_indices[-1]].copy()
#         cv2.putText(
#             image_l,
#             f"Keyframe {keyframe_indices[-1]}",
#             (20, 20),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.5,
#             (255, 255, 255),
#             2,
#         )
#         center_key = np.array([cx_key, cy_key]).astype(np.int32)
#         unit_vectors_key = np.eye(3)[:, :2]
#         annotate_coordinate_system(image_l, center_key, unit_vectors_key)

#         image_r = tactile_image.copy()
#         cv2.putText(
#             image_r,
#             f"Current Frame {curr_idx}",
#             (20, 20),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.5,
#             (255, 255, 255),
#             2,
#         )
#         center_3d_key = (
#             np.array([(cx_key - imgw / 2 + 0.5), (cy_key - imgh / 2 + 0.5), 0])
#             * ppmm
#             / 1000.0
#         )
#         unit_vectors_3d_key = np.eye(3) * ppmm / 1000.0


#         remapped_center_3d_key = (
#             np.dot(cTk[:3, :3], center_3d_key) + cTk[:3, 3]
#         )
#         remapped_cx_key = remapped_center_3d_key[0] * 1000 / ppmm + imgw / 2 - 0.5
#         remapped_cy_key = remapped_center_3d_key[1] * 1000 / ppmm + imgh / 2 - 0.5
#         remapped_center_key = np.array([remapped_cx_key, remapped_cy_key]).astype(
#             np.int32
#         )
#         remapped_unit_vectors_key = (
#             np.dot(cTk[:3, :3], unit_vectors_3d_key.T).T * 1000 / ppmm
#         )[:, :2]
#         annotate_coordinate_system(
#             image_r, remapped_center_key, remapped_unit_vectors_key
#         )
#         tracked_tactile_images.append(cv2.hconcat([image_l, image_r]))

#     # 保存可视化视频
#     save_path = os.path.join(data_dir, "tracked_test1.avi")
#     fourcc = cv2.VideoWriter_fourcc(*"FFV1")
#     video = cv2.VideoWriter(
#         save_path,
#         fourcc,
#         fps,
#         (tracked_tactile_images[0].shape[1], tracked_tactile_images[0].shape[0]),
#     )
#     for tracked_tactile_image in tracked_tactile_images:
#         video.write(tracked_tactile_image)
#     video.release()

#     # 保存关键帧点云为 .ply 文件，并进行 Poisson 重建
#     if len(keyframe_pointclouds) > 0:
#         global_pointcloud = np.concatenate(keyframe_pointclouds, axis=0)
#         print("关键帧拼接后点数：", global_pointcloud.shape[0])
#         # rounded-unique方法：先四舍五入再去重，兼顾效率和效果
#         decimals = 4
#         rounded_points = np.round(global_pointcloud, decimals=decimals)
#         unique_points = np.unique(rounded_points, axis=0)
#         print(f"rounded-unique去重后点数（decimals={decimals}）：", unique_points.shape[0])

#         # 创建点云对象
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(unique_points)

#         # 法线估算
#         pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=20))
#         pcd.orient_normals_consistent_tangent_plane(100)

#         # 移除离群点
#         cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
#         inlier_cloud = pcd.select_by_index(ind)
#         print(f"移除离群点后点数：{len(inlier_cloud.points)}")

#         # Poisson重建
#         mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
#             inlier_cloud, depth=10, width=0, scale=1.1, linear_fit=False
#         )
#         vertices_to_remove = densities < np.quantile(densities, 0.05)
#         mesh.remove_vertices_by_mask(vertices_to_remove)
#         print(f"密度过滤后三角形数量：{len(mesh.triangles)}")

#         # 保存结果
#         o3d.io.write_triangle_mesh(os.path.join(data_dir, "819sjwmesh.ply"), mesh)
#         o3d.io.write_point_cloud(os.path.join(data_dir, "819sjw.ply"), inlier_cloud)

# if __name__ == "__main__":
#     test_tracking()









###真实机械臂位姿
# import os
# import numpy as np
# import open3d as o3d
# import cv2
# import yaml
# from gs_sdk.gs_reconstruct import Reconstructor
# from normalflow.utils import erode_contact_mask, gxy2normal, height2pointcloud

# def euler_to_matrix(rx, ry, rz):
#     """将欧拉角 (deg) 转换为旋转矩阵"""
#     rx, ry, rz = np.deg2rad([rx, ry, rz])
#     Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
#     Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
#     Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
#     return Rz @ Ry @ Rx

# def pose_to_matrix(pose):
#     """将 [x,y,z,rx,ry,rz] 转换为 4x4 变换矩阵"""
#     x, y, z, rx, ry, rz = pose
#     T = np.eye(4)
#     T[:3, :3] = euler_to_matrix(rx, ry, rz)
#     T[:3, 3] = np.array([x, y, z]) / 1000.0  # 转为米
#     return T

# def test_tracking():
#     data_dir = os.path.join(os.path.dirname(__file__), "data")
#     model_path = os.path.join(os.path.dirname(__file__), "models", "nnmodel.pth")
#     config_path = os.path.join(os.path.dirname(__file__), "configs", "gsmini.yaml")
#     image_files = ["72111.png", "72112.png", "72113.png"]



#         # [603.9814, 176.9128, -237.9643, 177.2154, -0.4926, 16.1101],  # frame 1
#         # [612.4009, 182.1575, -238.1155, 177.3151, -0.9584, 3.8559],   # frame 2 (reference)
#         # [625.8934, 186.3103, -238.2120, 177.7004, -1.6834, -12.7387] 

# # [599.1135, 213.4891, -225.5, 180.0, -0.0, 0.0]
# # [602.1135, 213.4891, -226.5, 180.0, -2.0, -2.0]
# # [599.1135, 214.4891, -225.5, -180.0, 2.0, 2.0]
#     poses_deg = np.array([
#         [599.1135, 213.4891, -225.5, 180.0, -0.0, 0.0],  # frame 1
#         [602.1135, 213.4891, -225.5, 180.0, -2.0, -2.0],   # frame 2 (reference)
#         [599.1135, 214.4891, -225.5, -180.0, 2.0, 2.0]  # frame 3
#     ])

#     with open(config_path, "r") as f:
#         config = yaml.safe_load(f)
#         ppmm = config["ppmm"]

#     recon = Reconstructor(model_path, device="cpu")
#     bg_image = cv2.imread(os.path.join(data_dir, "background721.png"))
#     recon.load_bg(bg_image)

#     # 获取三帧的机械臂位姿矩阵
#     T1 = pose_to_matrix(poses_deg[0])# 参考帧（图1）
#     T2 = pose_to_matrix(poses_deg[1])  
#     T3 = pose_to_matrix(poses_deg[2])

#     # 计算图1/图3 到 图2的变换矩阵（不需要手眼标定）
#     # T_1_to_2 = np.linalg.inv(T2) @ T1
#     # T_3_to_2 = np.linalg.inv(T2) @ T3
#     T_2_to_1 = np.linalg.inv(T1) @ T2
#     T_3_to_1 = np.linalg.inv(T1) @ T3



#     comp = np.array([1, -0.005, 0.0]) 
#     # 读取图像和点云转换
#     global_points = []
#     for idx, (img_file, T_relative) in enumerate(zip(image_files, [T_2_to_1, np.eye(4), T_3_to_1])):
#         img_path = os.path.join(data_dir, img_file)
#         img = cv2.imread(img_path)

#         G, H, C = recon.get_surface_info(img, ppmm)
#         C = erode_contact_mask(C)
#         pc = height2pointcloud(H, ppmm)
#         pc = pc[C.reshape(-1)]
#         #pc_trans = (T_relative[:3, :3] @ pc.T).T + T_relative[:3, 3]
#         pc_trans = (T_relative[:3, :3] @ pc.T).T + T_relative[:3, 3]+comp
#         global_points.append(pc_trans)

#     # 合并并保存
#     all_points = np.concatenate(global_points, axis=0)
#     print("原始点数：", all_points.shape[0])

#     rounded = np.round(all_points, 4)
#     unique_points = np.unique(rounded, axis=0)
#     print("去重后点数：", unique_points.shape[0])

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(unique_points)
#     pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=20))
#     pcd.orient_normals_consistent_tangent_plane(100)

#     cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
#     inlier_cloud = pcd.select_by_index(ind)

#     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(inlier_cloud, depth=10)
#     vertices_to_remove = densities < np.quantile(densities, 0.05)
#     mesh.remove_vertices_by_mask(vertices_to_remove)

#     o3d.io.write_point_cloud(os.path.join(data_dir, "relative_pose_recon.ply"), inlier_cloud)
#     o3d.io.write_triangle_mesh(os.path.join(data_dir, "relative_pose_mesh.ply"), mesh)
#     print("[完成] 已保存三帧相对位姿拼接的点云与网格。")

# if __name__ == "__main__":
#     test_tracking()



####五张图,机械臂，只涉及平移#####
# import os
# import numpy as np
# import open3d as o3d
# import cv2
# import yaml
# from gs_sdk.gs_reconstruct import Reconstructor
# from normalflow.utils import erode_contact_mask, height2pointcloud

# def test_tracking():
#     data_dir = os.path.join(os.path.dirname(__file__), "data")
#     model_path = os.path.join(os.path.dirname(__file__), "models", "nnmodel.pth")
#     config_path = os.path.join(os.path.dirname(__file__), "configs", "gsmini.yaml")
#     # image_files = ["8191.png","8192.png","8193.png","8194.png","8195.png","8196.png","8197.png","8198.png"]  # 按顺序
#     image_files = ["8191.png","8192.png"]  # 按顺序
#     #image_files = ["7211.png", "7214.png","7215.png"]  # 按顺序

# #     [599.5601, 213.1957, -225.5, 180.0, -0.0, 0.0]
# # [599.5601, 210.1957, -225.5, 180.0, -0.0, 0.0]
# # [599.5601, 216.1957, -225.5, 180.0, -0.0, 0.0]
# # [602.5601, 213.1957, -225.5, 180.0, -0.0, 0.0]
# # [596.5601, 213.1957, -225.5, 180.0, -0.0, 0.0]


#     # 机械臂五帧的平移位姿（单位mm），顺序与image_files一致
#     poses = np.array([
#         # [599.5601, 213.1957, -225.5],  # 第一帧
#         # [599.5601, 210.1957, -225.5],  # 第二帧
#         # [599.5601, 216.1957, -225.5],  # 第三帧
#         # [602.5601, 213.1957, -225.5],  # 第四帧
#         # [596.5601, 213.1957, -225.5],  # 第五帧
#         # [213.1957, 599.5601, -225.5],  # 第一帧
#         # [210.1957, 599.5601, -225.5],  # 第二帧
#         # [216.1957, 599.5601, -225.5],  # 第三帧
#         # [213.1957, 602.5601, -225.5],  # 第四帧
#         # [213.1957, 596.5601, -225.5],  # 第五帧
#         [277.9799,450.0719,-25.5962],
#         [291.3917,453.0706,-23.9339],
#         # [286.7993,436.9666,-24.1143],
#         # [289.827,429.0074,-24.405],
#         # [297.2724,426.3873,-24.405],
#         # [300.639,428.5298,-24.405],
#         # [295.26,420.3284,-24.405],
#         # [303.8809,423.0389,-24.405],
#     ])





#     with open(config_path, "r") as f:
#         config = yaml.safe_load(f)
#         ppmm = config["ppmm"]

#     recon = Reconstructor(model_path, device="cpu")
#     bg_image = cv2.imread(os.path.join(data_dir, "819background.png"))
#     recon.load_bg(bg_image)

#     # 第一帧的平移（参考系，单位m）
#     t_ref = poses[0] / 1000.0

#     global_points = []
#     for img_file, t in zip(image_files, poses):
#         img_path = os.path.join(data_dir, img_file)
#         img = cv2.imread(img_path)
#         G, H, C = recon.get_surface_info(img, ppmm)
#         C = erode_contact_mask(C)
#         pc = height2pointcloud(H, ppmm)
#         pc = pc[C.reshape(-1)]
#         # t_gelsight= t /1000.0+hand2cam_offset
#         # # 只做平移，变换到第一帧坐标系
#         # t_delta=t_gelsight-t_ref
#         # pc_trans = pc + t_delta
#         t_delta = (t - poses[0]) / 1000.0  
#         pc_trans = pc + t_delta  
#         #global_points.append(pc_trans)
#         global_points.append(pc_trans)

#     # 合并点云并保存
#     all_points = np.concatenate(global_points, axis=0)
#     print("原始点数：", all_points.shape[0])
#     rounded = np.round(all_points, 4)
#     unique_points = np.unique(rounded, axis=0)
#     print("去重后点数：", unique_points.shape[0])

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(unique_points)
#     pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=20))
#     pcd.orient_normals_consistent_tangent_plane(100)
#     cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
#     inlier_cloud = pcd.select_by_index(ind)
#     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(inlier_cloud, depth=10)
#     vertices_to_remove = densities < np.quantile(densities, 0.05)
#     mesh.remove_vertices_by_mask(vertices_to_remove)
#     o3d.io.write_point_cloud(os.path.join(data_dir, "sjw819.ply"), inlier_cloud)
#     o3d.io.write_triangle_mesh(os.path.join(data_dir, "sjw_mesh819.ply"), mesh)
#     print("[完成] 已保存八帧平移拼接的点云与网格。")

# if __name__ == "__main__":
#     test_tracking()



# 点云不拼接，高度图直接转点云
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
    image_files = ["/home/shen/sjw/test/youxiao/24.jpg"]  # 按顺序

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        ppmm = config["ppmm"]

    recon = Reconstructor(model_path, device="cpu")
    bg_image = cv2.imread(os.path.join(data_dir, "819background.png"))
    recon.load_bg(bg_image)

    all_points = []
    for img_file in image_files:
        img_path = os.path.join(data_dir, img_file)
        img = cv2.imread(img_path)
        G, H, C = recon.get_surface_info(img, ppmm)
        C = erode_contact_mask(C)
        pc = height2pointcloud(H, ppmm)
        pc = pc[C.reshape(-1)]
        all_points.append(pc)

    # 合并所有点云
    all_points = np.concatenate(all_points, axis=0)
    print("合并后点数：", all_points.shape[0])

    # 保存为一个点云文件
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=20))
    pcd.orient_normals_consistent_tangent_plane(100)
    o3d.io.write_point_cloud(os.path.join(data_dir, "tie6.ply"), pcd)
    print("已保存所有点云到 all_clouds.ply")

if __name__ == "__main__":
    test_tracking()


