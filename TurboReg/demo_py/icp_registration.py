
import os
import numpy as np
import open3d as o3d
import time
import copy

# ====== 路径配置======
src_file = "/home/shen/sjw/TurboReg/tie346.ply"  # 源点云
dst_file = "/home/shen/normalflow/normalflow-master/examples/data/tie6.ply"  # 目标点云

# 初始体素大小
voxel_size = 0.000106


# ====== 可视化工具 ======
def draw_registration_result(source, target, transformation, title=""):
    src_temp = copy.deepcopy(source)
    tgt_temp = copy.deepcopy(target)
    src_temp.transform(transformation)
    src_temp.paint_uniform_color([1.0, 0.0, 0.0])   # 红：变换后的源
    tgt_temp.paint_uniform_color([0.0, 0.0, 1.0])   # 蓝：目标
    o3d.visualization.draw_geometries([src_temp, tgt_temp], window_name=title)


# ====== 配准流程函数 ======
def estimate_voxel_size(source, target):
    """自动估计合适的体素大小（边界框对角线的约 1/150）"""
    source_bbox = source.get_axis_aligned_bounding_box()
    target_bbox = target.get_axis_aligned_bounding_box()
    source_diagonal = np.linalg.norm(source_bbox.max_bound - source_bbox.min_bound)
    target_diagonal = np.linalg.norm(target_bbox.max_bound - target_bbox.min_bound)
    avg_diagonal = (source_diagonal + target_diagonal) / 2
    estimated_voxel_size = max(1e-6, avg_diagonal / 150.0)

    print(f"源点云边界框对角线: {source_diagonal:.6f}")
    print(f"目标点云边界框对角线: {target_diagonal:.6f}")
    print(f"建议体素大小: {estimated_voxel_size:.6f}")
    return estimated_voxel_size


def load_data():
    """加载两个指定的 PLY 文件"""
    assert os.path.exists(src_file), f"源文件不存在: {src_file}"
    assert os.path.exists(dst_file), f"目标文件不存在: {dst_file}"
    src = o3d.io.read_point_cloud(src_file)
    dst = o3d.io.read_point_cloud(dst_file)
    return src, dst


def preprocess_point_cloud(pcd, voxel):
    """下采样 + 估计法线"""
    print(f"Original points: {len(pcd.points)}")
    pcd_down = pcd.voxel_down_sample(voxel)
    print(f"Downsampled points: {len(pcd_down.points)}")
    radius_normal = max(voxel * 2.0, 1e-6)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    return pcd_down


def compute_fpfh_feature(pcd, voxel):
    """计算 FPFH 特征（要求已估计法线）"""
    radius_feature = max(voxel * 5.0, 1e-6)
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel):
    """全局配准（RANSAC 基于特征）"""
    distance_threshold = voxel * 1.5
    print("执行全局配准（RANSAC）...")
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )

    if result.fitness < 0.1:
        print(f"警告: 全局配准 fitness 较低: {result.fitness:.6f}，尝试放宽参数...")
        distance_threshold_loose = voxel * 3.0
        result_loose = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold_loose,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.7),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold_loose)
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 0.99)
        )
        if result_loose.fitness > result.fitness:
            print(f"使用宽松参数得到了更好结果: {result_loose.fitness:.6f}")
            result = result_loose

    return result




def refine_registration(source, target, init_T, voxel):
    """多层次 ICP 精细配准（在下采样点云上进行）"""
    print("执行多层次 ICP 精细配准...")

    th1 = voxel * 2.0
    print(f"Step1 ICP (Point-to-Point), threshold={th1:.6f}")
    res1 = o3d.pipelines.registration.registration_icp(
        source, target, th1, init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    print(f"  fitness={res1.fitness:.6f}, rmse={res1.inlier_rmse:.6f}")

    th2 = voxel * 1.0
    print(f"Step2 ICP (Point-to-Plane), threshold={th2:.6f}")
    res2 = o3d.pipelines.registration.registration_icp(
        source, target, th2, res1.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    print(f"  fitness={res2.fitness:.6f}, rmse={res2.inlier_rmse:.6f}")

    th3 = voxel * 0.5
    print(f"Step3 ICP (Point-to-Plane Fine), threshold={th3:.6f}")
    res3 = o3d.pipelines.registration.registration_icp(
        source, target, th3, res2.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    print(f"  fitness={res3.fitness:.6f}, rmse={res3.inlier_rmse:.6f}")

    candidates = [("P2P step1", res1), ("P2Plane step2", res2), ("P2Plane fine step3", res3)]
    best_method, best_res = max(candidates, key=lambda x: x[1].fitness)

    print("\n=== ICP 配准总结 ===")
    print(f"采用: {best_method}")
    print(f"最佳 fitness: {best_res.fitness:.6f} | 最佳 rmse: {best_res.inlier_rmse:.6f}")
    return best_res, best_method


def evaluate_registration(source, target, T, voxel):
    """简单评估：配准后源->目标的最近邻距离统计"""
    src_tf = copy.deepcopy(source)
    src_tf.transform(T)
    distances = np.asarray(src_tf.compute_point_cloud_distance(target))
    mean_d = float(np.mean(distances)) if distances.size > 0 else np.inf
    std_d = float(np.std(distances)) if distances.size > 0 else np.inf
    thr = voxel * 2.0
    inlier_ratio = float(np.sum(distances < thr)) / len(distances) if distances.size > 0 else 0.0

    print("配准评估：")
    print(f"  平均距离: {mean_d:.6f}")
    print(f"  距离标准差: {std_d:.6f}")
    print(f"  内点比例(阈值 {thr:.6f}): {inlier_ratio:.3f}")
    return mean_d, inlier_ratio


def fuse_and_save(source, target, T, out_path,
                  colorize=True,
                  dedup_voxel=None,
                  recompute_normals=True,
                  base_voxel_for_normal=None):
    """
    融合为单个点云并保存为 PLY。
    - colorize: 红=配准后的源，蓝=目标
    - dedup_voxel: 体素去重（None 不去重）
    - recompute_normals: 保存前重算法线
    - base_voxel_for_normal: 控制法线估计半径用的基准体素（默认取 dedup_voxel 或 估计体素）
    """
    src_tf = copy.deepcopy(source)
    src_tf.transform(T)

    if colorize:
        src_tf.paint_uniform_color([1.0, 0.0, 0.0])  # 红
        tgt_col = copy.deepcopy(target)
        tgt_col.paint_uniform_color([0.0, 0.0, 1.0])  # 蓝
    else:
        tgt_col = copy.deepcopy(target)

    merged = o3d.geometry.PointCloud()
    pts_src = np.asarray(src_tf.points)
    pts_tgt = np.asarray(tgt_col.points)
    merged.points = o3d.utility.Vector3dVector(np.vstack([pts_src, pts_tgt]))

    # 合并颜色（即使原始无颜色，也会提供红/蓝）
    cols_src = np.asarray(src_tf.colors) if src_tf.has_colors() else np.tile([1, 0, 0], (pts_src.shape[0], 1))
    cols_tgt = np.asarray(tgt_col.colors) if tgt_col.has_colors() else np.tile([0, 0, 1], (pts_tgt.shape[0], 1))
    merged.colors = o3d.utility.Vector3dVector(np.vstack([cols_src, cols_tgt]))

    # 去重（避免重叠处过密或文件过大）
    if dedup_voxel is not None and dedup_voxel > 0:
        merged = merged.voxel_down_sample(dedup_voxel)

    # 重算法线（MeshLab 中一些渲染/滤波会用到法线）
    if recompute_normals:
        base = base_voxel_for_normal if base_voxel_for_normal else (dedup_voxel if dedup_voxel else 1e-3)
        rad = max(base * 3.0, 1e-6)
        merged.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=rad, max_nn=50))

    o3d.io.write_point_cloud(out_path, merged, write_ascii=True)
    print(f"✅ 融合后的双云已保存（红=配准源，蓝=目标）：{out_path}")


# ====== 主流程 ======
def main():
    print("=== Open3D ICP 点云配准 + 可视化 + 融合导出 ===")

    # 1) 加载
    print("\n1. 加载点云...")
    source, target = load_data()
    print(f"源点云: {len(source.points)} 点")
    print(f"目标点云: {len(target.points)} 点")

    # 1.5) 显示原始点云
    print("\n显示原始点云（红=源，蓝=目标）...")
    src_show = copy.deepcopy(source); src_show.paint_uniform_color([1.0, 0.0, 0.0])
    tgt_show = copy.deepcopy(target); tgt_show.paint_uniform_color([0.0, 0.0, 1.0])
    o3d.visualization.draw_geometries([src_show, tgt_show], window_name="原始点云 (红=源, 蓝=目标)")

    # 2) 自动估计体素大小
    print("\n2. 估计体素大小...")
    est_voxel = estimate_voxel_size(source, target)
    if est_voxel > voxel_size * 2 or est_voxel < voxel_size / 2:
        print(f"当前设置的体素大小 {voxel_size} 可能不合适，改用估计值 {est_voxel:.6f}")
        actual_voxel = est_voxel
    else:
        print(f"使用当前设置的体素大小: {voxel_size}")
        actual_voxel = voxel_size

    # 3) 预处理（下采样 + 法线）用于特征与 ICP
    print("\n3. 预处理下采样点云...")
    source_down = preprocess_point_cloud(source, actual_voxel)
    target_down = preprocess_point_cloud(target, actual_voxel)

    # 4) 计算 FPFH
    print("\n4. 计算 FPFH 特征...")
    src_fpfh = compute_fpfh_feature(source_down, actual_voxel)
    tgt_fpfh = compute_fpfh_feature(target_down, actual_voxel)

    # 5) 全局配准（RANSAC）
    print("\n5. 全局配准（RANSAC）...")
    t0 = time.time()
    res_ransac = execute_global_registration(source_down, target_down, src_fpfh, tgt_fpfh, actual_voxel)
    t_ransac = time.time() - t0
    print(f"RANSAC 用时: {t_ransac:.3f}s | fitness={res_ransac.fitness:.6f} | rmse={res_ransac.inlier_rmse:.6f}")

    # 5.5) 可视化全局配准结果（在原始分辨率上显示）
    print("\n显示全局配准结果（红=配准源，蓝=目标）...")
    draw_registration_result(source, target, res_ransac.transformation, title="全局配准结果 (RANSAC)")

    if res_ransac.fitness < 0.05:
        print("警告: 全局配准很弱，使用单位阵作为初始变换继续 ICP（可能失败）")
        res_ransac.transformation = np.eye(4)

    # 6) ICP 精细配准（在下采样点云上）
    print("\n6. ICP 精细配准...")
    t1 = time.time()
    res_icp, best_icp_method = refine_registration(source_down, target_down, res_ransac.transformation, actual_voxel)
    t_icp = time.time() - t1
    print(f"ICP 用时: {t_icp:.3f}s | fitness={res_icp.fitness:.6f} | rmse={res_icp.inlier_rmse:.6f}")

    # 6.5) 可视化最终 ICP 配准结果（在原始分辨率上显示）
    print("\n显示最终 ICP 配准结果（红=配准源，蓝=目标）...")
    draw_registration_result(source, target, res_icp.transformation, title=f"最终 ICP 配准结果 ({best_icp_method})")

    # 7) 评估（把变换应用到原始分辨率源云上再评估）
    print("\n7. 评估配准质量（基于原始目标）...")
    evaluate_registration(source, target, res_icp.transformation, actual_voxel)

    # 8) 输出变换矩阵
    print("\n8. 最终变换矩阵 T：")
    T = res_icp.transformation
    np.set_printoptions(suppress=True, precision=9)
    print(T)

    R = T[:3, :3]; t = T[:3, 3]
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    rot_deg = np.degrees(np.arccos(tr))
    print(f"变换概览：|t|={np.linalg.norm(t):.6f} ，旋转角≈{rot_deg:.3f}°")

    # 9) 融合并保存为一个 PLY（tie346.ply），放在源点云所在目录
    print("\n9. 融合并保存单一 PLY ...")
    out_dir = os.path.dirname(os.path.abspath(src_file))
    os.makedirs(out_dir, exist_ok=True)
    out_fused = os.path.abspath(os.path.join(out_dir, "tie346.ply"))

    dedup_voxel = actual_voxel * 0.5  # 不想体素去重可改为 None
    fuse_and_save(
        source, target, T, out_fused,
        colorize=True,
        dedup_voxel=dedup_voxel,
        recompute_normals=True,
        base_voxel_for_normal=actual_voxel
    )

    # 10) 同步保存变换矩阵与配准信息
    print("\n10. 保存配准信息 ...")
    out_T = os.path.abspath(os.path.join(out_dir, "tie346_transform.txt"))
    np.savetxt(out_T, T, fmt="%.9f")
    print(f"已保存变换矩阵：{out_T}")

    info_file = os.path.abspath(os.path.join(out_dir, "tie346_registration_info.txt"))
    with open(info_file, "w", encoding="utf-8") as f:
        f.write("=== ICP 点云配准结果信息 ===\n")
        f.write(f"源点云: {src_file}\n")
        f.write(f"目标点云: {dst_file}\n")
        f.write(f"源点数: {len(source.points)}\n")
        f.write(f"目标点数: {len(target.points)}\n")
        f.write(f"使用体素大小: {actual_voxel:.9f}\n")
        f.write(f"RANSAC: fitness={res_ransac.fitness:.6f}, rmse={res_ransac.inlier_rmse:.6f}, time={t_ransac:.3f}s\n")
        f.write(f"ICP方法: {best_icp_method}, fitness={res_icp.fitness:.6f}, rmse={res_icp.inlier_rmse:.6f}, time={t_icp:.3f}s\n")
        f.write(f"变换矩阵 T:\n{np.array2string(T, formatter={'float_kind':lambda x: f'{x:.9f}'})}\n")
    print(f"已保存配准信息：{info_file}")

    print("\n🎉 完成。")
    print(f"可视化窗口已展示；融合文件（单一 PLY）：{out_fused}")


if __name__ == "__main__":
    main()
