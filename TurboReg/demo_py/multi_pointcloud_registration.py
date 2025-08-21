import os
import numpy as np
import open3d as o3d
import time
import copy
from typing import List, Tuple

# 修改为你的五个PLY文件路径
ply_files = [
      # 参考点云/home/shen/normalflow/normalflow-master/examples/data/tie.ply
    "/home/shen/normalflow/normalflow-master/examples/data/tie3.ply",  # 第二个点云
    "/home/shen/normalflow/normalflow-master/examples/data/tie5.ply",  # 第三个点云
]

voxel_size = 0.004876 # 体素大小0.001
ENABLE_VISUALIZATION = True  # 设置为True以查看每步配准结果
PRESERVE_ORIGINAL_COLORS = True  # 设置为True保留原始点云颜色，False使用统一颜色方案

def estimate_voxel_size(point_clouds: List[o3d.geometry.PointCloud]):
    """基于所有点云自动估计合适的体素大小"""
    total_diagonal = 0
    for pcd in point_clouds:
        bbox = pcd.get_axis_aligned_bounding_box()
        diagonal = np.linalg.norm(bbox.max_bound - bbox.min_bound)
        total_diagonal += diagonal
    
    avg_diagonal = total_diagonal / len(point_clouds)
    estimated_voxel_size = avg_diagonal / 150
    
    print(f"平均边界框对角线长度: {avg_diagonal:.6f}")
    print(f"建议的体素大小: {estimated_voxel_size:.6f}")
    
    return estimated_voxel_size

def load_all_point_clouds():
    """加载所有PLY文件"""
    point_clouds = []
    print("正在加载所有点云文件...")
    
    for i, file_path in enumerate(ply_files):
        if not os.path.exists(file_path):
            print(f"警告: 文件 {file_path} 不存在，跳过...")
            continue
            
        pcd = o3d.io.read_point_cloud(file_path)
        if len(pcd.points) == 0:
            print(f"警告: 文件 {file_path} 为空，跳过...")
            continue
            
        point_clouds.append(pcd)
        print(f"点云 {i+1}: {len(pcd.points)} 个点 - {file_path}")
    
    if len(point_clouds) < 2:
        raise ValueError("至少需要2个有效的点云文件进行配准")
    
    return point_clouds

def preprocess_point_cloud(pcd, voxel_size):
    """预处理点云：下采样和估计法线"""
    # 下采样
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # 估计法线
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    
    return pcd_down

def compute_fpfh_feature(pcd, voxel_size):
    """计算FPFH特征"""
    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """执行全局配准（RANSAC）"""
    distance_threshold = voxel_size * 1.5
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], 
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    
    # 如果全局配准质量太低，尝试更宽松的参数
    if result.fitness < 0.1:
        print(f"  全局配准质量较低 (fitness: {result.fitness:.6f})，尝试宽松参数...")
        
        distance_threshold_loose = voxel_size * 3.0
        result_loose = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold_loose,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.7),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold_loose)
            ], 
            o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 0.99)
        )
        
        if result_loose.fitness > result.fitness:
            result = result_loose
    
    return result

def refine_registration(source, target, initial_transformation, voxel_size):
    """使用多层次ICP精细配准"""
    # Point-to-Point ICP
    distance_threshold_1 = voxel_size * 2.0
    result_1 = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold_1, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    
    # Point-to-Plane ICP
    distance_threshold_2 = voxel_size * 1.0
    result_2 = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold_2, result_1.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    
    # 精细Point-to-Plane ICP
    distance_threshold_3 = voxel_size * 0.5
    result_3 = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold_3, result_2.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    
    # 选择最佳结果
    results = [
        ("Point-to-Point", result_1),
        ("Point-to-Plane", result_2), 
        ("Point-to-Plane (精细)", result_3)
    ]
    
    best_method, best_result = max(results, key=lambda x: x[1].fitness)
    return best_result, best_method

def register_single_pair(source, target, voxel_size, pair_name=""):
    """配准单对点云"""
    print(f"\n--- 配准 {pair_name} ---")
    
    # 预处理
    print("  预处理点云...")
    source_down = preprocess_point_cloud(source, voxel_size)
    target_down = preprocess_point_cloud(target, voxel_size)
    
    print(f"  源点云下采样: {len(source.points)} -> {len(source_down.points)} 点")
    print(f"  目标点云下采样: {len(target.points)} -> {len(target_down.points)} 点")
    
    # 计算特征
    print("  计算FPFH特征...")
    source_fpfh = compute_fpfh_feature(source_down, voxel_size)
    target_fpfh = compute_fpfh_feature(target_down, voxel_size)
    
    # 全局配准
    print("  执行全局配准...")
    start_time = time.time()
    result_ransac = execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size
    )
    global_time = time.time() - start_time
    
    print(f"  全局配准用时: {global_time:.3f} 秒")
    print(f"  全局配准fitness: {result_ransac.fitness:.6f}")
    
    # 如果全局配准失败，使用单位矩阵
    if result_ransac.fitness < 0.05:
        print("  全局配准效果很差，使用单位矩阵作为初始变换...")
        result_ransac.transformation = np.eye(4)
    
    # ICP精细配准
    print("  执行ICP精细配准...")
    start_time = time.time()
    result_icp, best_method = refine_registration(
        source_down, target_down, result_ransac.transformation, voxel_size
    )
    icp_time = time.time() - start_time
    
    print(f"  ICP配准用时: {icp_time:.3f} 秒")
    print(f"  最佳ICP方法: {best_method}")
    print(f"  最终fitness: {result_icp.fitness:.6f}")
    print(f"  最终inlier_rmse: {result_icp.inlier_rmse:.6f}")
    
    return result_icp.transformation, result_icp.fitness, best_method

def visualize_registration_step(point_clouds_transformed, step_name, preserve_colors=None):
    """可视化当前配准步骤的结果"""
    if not ENABLE_VISUALIZATION:
        return
        
    print(f"  显示配准结果: {step_name}")
    
    # 如果没有指定preserve_colors，使用全局配置
    if preserve_colors is None:
        preserve_colors = PRESERVE_ORIGINAL_COLORS
    
    display_clouds = []
    
    if preserve_colors:
        # 保留原始点云颜色
        for i, pcd in enumerate(point_clouds_transformed):
            if pcd is not None:
                pcd_copy = copy.deepcopy(pcd)
                
                # 检查点云是否有颜色信息
                if len(pcd_copy.colors) == 0:
                    # 如果没有颜色，使用默认颜色方案
                    colors = [
                        [1.0, 0.0, 0.0],  # 红色 - 参考点云
                        [0.0, 1.0, 0.0],  # 绿色
                        [0.0, 0.0, 1.0],  # 蓝色
                        [1.0, 1.0, 0.0],  # 黄色
                        [1.0, 0.0, 1.0],  # 品红色
                        [0.0, 1.0, 1.0],  # 青色
                    ]
                    color_idx = i % len(colors)
                    pcd_copy.paint_uniform_color(colors[color_idx])
                    print(f"    点云 {i+1}: 无原始颜色，使用默认颜色")
                else:
                    print(f"    点云 {i+1}: 保留原始颜色 ({len(pcd_copy.colors)} 个颜色点)")
                
                display_clouds.append(pcd_copy)
    else:
        # 使用统一颜色方案
        colors = [
            [1.0, 0.0, 0.0],  # 红色 - 参考点云
            [0.0, 1.0, 0.0],  # 绿色
            [0.0, 0.0, 1.0],  # 蓝色
            [1.0, 1.0, 0.0],  # 黄色
            [1.0, 0.0, 1.0],  # 品红色
            [0.0, 1.0, 1.0],  # 青色
        ]
        
        for i, pcd in enumerate(point_clouds_transformed):
            if pcd is not None:
                pcd_copy = copy.deepcopy(pcd)
                color_idx = i % len(colors)
                pcd_copy.paint_uniform_color(colors[color_idx])
                display_clouds.append(pcd_copy)
    
    if display_clouds:
        o3d.visualization.draw_geometries(display_clouds, window_name=step_name)

def sequential_registration(point_clouds, voxel_size):
    """顺序配准：将所有点云配准到第一个参考点云"""
    print("\n=== 开始顺序配准 ===")
    print("配准策略: 将所有点云配准到第一个点云（参考点云）")
    
    if len(point_clouds) < 2:
        raise ValueError("至少需要2个点云进行配准")
    
    # 第一个点云作为参考
    reference_cloud = point_clouds[0]
    transformations = [np.eye(4)]  # 参考点云的变换矩阵是单位矩阵
    registered_clouds = [copy.deepcopy(reference_cloud)]
    
    print(f"参考点云: 点云1 ({len(reference_cloud.points)} 点)")
    
    # 配准信息记录
    registration_info = []
    
    # 依次将其他点云配准到参考点云
    for i in range(1, len(point_clouds)):
        source = point_clouds[i]
        target = reference_cloud
        
        pair_name = f"点云{i+1} -> 点云1(参考)"
        
        # 执行配准
        transformation, fitness, method = register_single_pair(
            source, target, voxel_size, pair_name
        )
        
        # 记录配准信息
        info = {
            'source_index': i,
            'target_index': 0,
            'transformation': transformation,
            'fitness': fitness,
            'method': method,
            'pair_name': pair_name
        }
        registration_info.append(info)
        
        # 应用变换
        source_registered = copy.deepcopy(source)
        source_registered.transform(transformation)
        
        transformations.append(transformation)
        registered_clouds.append(source_registered)
        
        # 可视化当前步骤
        current_clouds = [reference_cloud] + registered_clouds[1:i+1]
        visualize_registration_step(current_clouds, f"步骤 {i}: {pair_name}")
        
        # 评估配准质量
        if fitness > 0.8:
            quality = "优秀"
        elif fitness > 0.5:
            quality = "良好"
        elif fitness > 0.3:
            quality = "一般"
        else:
            quality = "较差"
        
        print(f"  配准质量: {quality}")
    
    return registered_clouds, transformations, registration_info

def pairwise_registration(point_clouds, voxel_size):
    """配对配准：相邻点云依次配准"""
    print("\n=== 开始配对配准 ===")
    print("配准策略: 相邻点云依次配准 (1->2, 2->3, 3->4, 4->5)")
    
    if len(point_clouds) < 2:
        raise ValueError("至少需要2个点云进行配准")
    
    registered_clouds = [copy.deepcopy(point_clouds[0])]
    transformations = [np.eye(4)]  # 第一个点云的累积变换是单位矩阵
    registration_info = []
    
    print(f"起始点云: 点云1 ({len(point_clouds[0].points)} 点)")
    
    # 累积变换矩阵
    accumulated_transform = np.eye(4)
    
    for i in range(1, len(point_clouds)):
        source = point_clouds[i]
        target = registered_clouds[-1]  # 配准到前一个已配准的点云
        
        pair_name = f"点云{i+1} -> 点云{i}"
        
        # 执行配准
        transformation, fitness, method = register_single_pair(
            source, target, voxel_size, pair_name
        )
        
        # 累积变换
        accumulated_transform = transformation @ accumulated_transform
        
        # 记录配准信息
        info = {
            'source_index': i,
            'target_index': i-1,
            'transformation': transformation,
            'accumulated_transformation': accumulated_transform.copy(),
            'fitness': fitness,
            'method': method,
            'pair_name': pair_name
        }
        registration_info.append(info)
        
        # 应用变换到原始点云
        source_registered = copy.deepcopy(point_clouds[i])
        source_registered.transform(accumulated_transform)
        
        transformations.append(accumulated_transform.copy())
        registered_clouds.append(source_registered)
        
        # 可视化当前步骤
        visualize_registration_step(registered_clouds, f"步骤 {i}: {pair_name}")
        
        # 评估配准质量
        if fitness > 0.8:
            quality = "优秀"
        elif fitness > 0.5:
            quality = "良好"
        elif fitness > 0.3:
            quality = "一般"
        else:
            quality = "较差"
        
        print(f"  配准质量: {quality}")
    
    return registered_clouds, transformations, registration_info

def save_results(registered_clouds, transformations, registration_info, method_name):
    """保存配准结果"""
    print(f"\n=== 保存配准结果 ({method_name}) ===")
    
    # 创建结果目录
    result_dir = f"registration_results_{method_name.lower()}"
    os.makedirs(result_dir, exist_ok=True)
    
    # 保存配准后的点云
    for i, pcd in enumerate(registered_clouds):
        filename = os.path.join(result_dir, f"registered_cloud_{i+1}.ply")
        o3d.io.write_point_cloud(filename, pcd)
        print(f"  保存点云 {i+1}: {filename}")
    
    # 保存变换矩阵
    for i, transform in enumerate(transformations):
        filename = os.path.join(result_dir, f"transformation_matrix_{i+1}.txt")
        np.savetxt(filename, transform)
        print(f"  保存变换矩阵 {i+1}: {filename}")
    
    # 保存合并的点云文件
    print("  合并所有配准后的点云...")
    merged_cloud = o3d.geometry.PointCloud()
    
    # 为每个点云分配不同颜色（如果原始点云没有颜色）
    merge_colors = [
        [1.0, 0.0, 0.0],  # 红色
        [0.0, 1.0, 0.0],  # 绿色  
        [0.0, 0.0, 1.0],  # 蓝色
        [1.0, 1.0, 0.0],  # 黄色
        [1.0, 0.0, 1.0],  # 品红色
        [0.0, 1.0, 1.0],  # 青色
    ]
    
    for i, pcd in enumerate(registered_clouds):
        pcd_copy = copy.deepcopy(pcd)
        
        # 如果点云没有颜色，分配默认颜色
        if len(pcd_copy.colors) == 0:
            color_idx = i % len(merge_colors)
            pcd_copy.paint_uniform_color(merge_colors[color_idx])
            print(f"    点云 {i+1}: 分配颜色 {merge_colors[color_idx]}")
        else:
            print(f"    点云 {i+1}: 保留原始颜色")
        
        # 合并到总点云
        merged_cloud += pcd_copy
    
    # 保存合并的点云
    merged_filename = os.path.join(result_dir, "merged_registered_clouds.ply")
    o3d.io.write_point_cloud(merged_filename, merged_cloud)
    print(f"  保存合并点云: {merged_filename}")
    print(f"  合并点云总点数: {len(merged_cloud.points)}")
    
    # 保存配准信息
    info_file = os.path.join(result_dir, "registration_info.txt")
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(f"=== {method_name} 点云配准结果信息 ===\n\n")
        f.write(f"输入文件数量: {len(ply_files)}\n")
        f.write("输入文件列表:\n")
        for i, file_path in enumerate(ply_files):
            f.write(f"  点云 {i+1}: {file_path}\n")
        f.write(f"\n使用的体素大小: {voxel_size:.6f}\n")
        f.write(f"配准方法: {method_name}\n")
        f.write(f"合并点云文件: merged_registered_clouds.ply\n")
        f.write(f"合并点云总点数: {len(merged_cloud.points)}\n\n")
        
        f.write("详细配准信息:\n")
        for i, info in enumerate(registration_info):
            f.write(f"\n步骤 {i+1}: {info['pair_name']}\n")
            f.write(f"  ICP方法: {info['method']}\n")
            f.write(f"  Fitness: {info['fitness']:.6f}\n")
            f.write(f"  变换矩阵文件: transformation_matrix_{info['source_index']+1}.txt\n")
        
        f.write("\n输出文件列表:\n")
        for i in range(len(registered_clouds)):
            f.write(f"  单独点云 {i+1}: registered_cloud_{i+1}.ply\n")
        f.write(f"  合并点云: merged_registered_clouds.ply\n")
    
    print(f"  保存配准信息: {info_file}")
    print(f"  所有结果保存在目录: {result_dir}")

def main():
    """主函数：执行多点云配准"""
    print("=== 多点云配准系统 ===")
    
    # 打印配置信息
    print(f"可视化配置:")
    print(f"  启用可视化: {ENABLE_VISUALIZATION}")
    print(f"  保留原始颜色: {PRESERVE_ORIGINAL_COLORS}")
    
    # 1. 加载所有点云
    try:
        point_clouds = load_all_point_clouds()
    except Exception as e:
        print(f"错误: {e}")
        return
    
    print(f"\n成功加载 {len(point_clouds)} 个点云文件")
    
    # 2. 估计体素大小
    print("\n估计合适的体素大小...")
    estimated_voxel_size = estimate_voxel_size(point_clouds)
    
    # 选择体素大小
    if abs(estimated_voxel_size - voxel_size) > voxel_size:
        print(f"当前设置的体素大小 {voxel_size} 可能不合适")
        print(f"建议使用估计的体素大小: {estimated_voxel_size:.6f}")
        actual_voxel_size = estimated_voxel_size
    else:
        print(f"使用当前设置的体素大小: {voxel_size}")
        actual_voxel_size = voxel_size
    
    # 3. 显示原始点云
    if ENABLE_VISUALIZATION:
        print("\n显示原始点云...")
        
        # 检查点云是否有颜色信息
        has_colors = any(len(pcd.colors) > 0 for pcd in point_clouds)
        
        if has_colors:
            print("检测到点云有原始颜色，将保留原始颜色显示")
            display_clouds = []
            for i, pcd in enumerate(point_clouds):
                pcd_copy = copy.deepcopy(pcd)
                if len(pcd_copy.colors) == 0:
                    # 为没有颜色的点云分配默认颜色
                    colors = [
                        [1.0, 0.0, 0.0],  # 红色
                        [0.0, 1.0, 0.0],  # 绿色
                        [0.0, 0.0, 1.0],  # 蓝色
                        [1.0, 1.0, 0.0],  # 黄色
                        [1.0, 0.0, 1.0],  # 品红色
                    ]
                    color_idx = i % len(colors)
                    pcd_copy.paint_uniform_color(colors[color_idx])
                    print(f"  点云 {i+1}: 无颜色信息，使用默认颜色")
                else:
                    print(f"  点云 {i+1}: 保留原始颜色 ({len(pcd_copy.colors)} 个颜色点)")
                display_clouds.append(pcd_copy)
        else:
            print("未检测到点云颜色信息，使用默认颜色方案")
            colors = [
                [1.0, 0.0, 0.0],  # 红色
                [0.0, 1.0, 0.0],  # 绿色
                [0.0, 0.0, 1.0],  # 蓝色
                [1.0, 1.0, 0.0],  # 黄色
                [1.0, 0.0, 1.0],  # 品红色
            ]
            
            display_clouds = []
            for i, pcd in enumerate(point_clouds):
                pcd_copy = copy.deepcopy(pcd)
                color_idx = i % len(colors)
                pcd_copy.paint_uniform_color(colors[color_idx])
                display_clouds.append(pcd_copy)
        
        o3d.visualization.draw_geometries(display_clouds, 
                                          window_name="原始点云 (保留原始颜色或使用默认颜色)")
    
    # 4. 选择配准策略
    print("\n选择配准策略:")
    print("1. 顺序配准 (推荐): 将所有点云配准到第一个参考点云")
    print("2. 配对配准: 相邻点云依次配准")
    
    while True:
        try:
            choice = input("请选择配准策略 (1 或 2): ").strip()
            if choice in ['1', '2']:
                break
            else:
                print("请输入 1 或 2")
        except KeyboardInterrupt:
            print("\n用户取消操作")
            return
    
    # 5. 执行配准
    start_time = time.time()
    
    if choice == '1':
        registered_clouds, transformations, registration_info = sequential_registration(
            point_clouds, actual_voxel_size
        )
        method_name = "Sequential"
    else:
        registered_clouds, transformations, registration_info = pairwise_registration(
            point_clouds, actual_voxel_size
        )
        method_name = "Pairwise"
    
    total_time = time.time() - start_time
    
    # 6. 显示最终结果
    print(f"\n=== 配准完成 ===")
    print(f"总用时: {total_time:.3f} 秒")
    print(f"配准方法: {method_name}")
    print(f"成功配准 {len(registered_clouds)} 个点云")
    
    # 最终可视化
    if ENABLE_VISUALIZATION:
        print("\n显示最终配准结果...")
        visualize_registration_step(registered_clouds, f"最终配准结果 - {method_name}")
    
    # 7. 保存结果
    save_results(registered_clouds, transformations, registration_info, method_name)
    
    # 7.5 显示最终合并点云
    if ENABLE_VISUALIZATION:
        print("\n显示最终合并点云...")
        # 创建合并点云用于最终显示
        final_merged_cloud = o3d.geometry.PointCloud()
        merge_colors = [
            [1.0, 0.0, 0.0],  # 红色
            [0.0, 1.0, 0.0],  # 绿色  
            [0.0, 0.0, 1.0],  # 蓝色
            [1.0, 1.0, 0.0],  # 黄色
            [1.0, 0.0, 1.0],  # 品红色
            [0.0, 1.0, 1.0],  # 青色
        ]
        
        display_clouds_final = []
        for i, pcd in enumerate(registered_clouds):
            pcd_copy = copy.deepcopy(pcd)
            
            # 如果启用了保留原始颜色且点云有颜色，则保留原始颜色
            if PRESERVE_ORIGINAL_COLORS and len(pcd_copy.colors) > 0:
                print(f"  最终显示 - 点云 {i+1}: 保留原始颜色")
            else:
                # 否则使用区分颜色
                color_idx = i % len(merge_colors)
                pcd_copy.paint_uniform_color(merge_colors[color_idx])
                print(f"  最终显示 - 点云 {i+1}: 使用区分颜色 {merge_colors[color_idx]}")
            
            display_clouds_final.append(pcd_copy)
        
        o3d.visualization.draw_geometries(display_clouds_final, 
                                          window_name=f"最终合并结果 - {method_name} (所有配准后点云)")
    
    # 8. 配准质量统计
    print(f"\n=== 配准质量统计 ===")
    fitness_scores = [info['fitness'] for info in registration_info]
    if fitness_scores:
        avg_fitness = np.mean(fitness_scores)
        min_fitness = np.min(fitness_scores)
        max_fitness = np.max(fitness_scores)
        
        print(f"平均 Fitness: {avg_fitness:.6f}")
        print(f"最低 Fitness: {min_fitness:.6f}")
        print(f"最高 Fitness: {max_fitness:.6f}")
        
        if avg_fitness > 0.8:
            print("✅ 整体配准质量: 优秀")
        elif avg_fitness > 0.5:
            print("✅ 整体配准质量: 良好")
        elif avg_fitness > 0.3:
            print("⚠️  整体配准质量: 一般")
        else:
            print("❌ 整体配准质量: 较差")
    
    print("\n配准完成！请查看保存的结果文件。")
    print("\n=== 输出文件总结 ===")
    result_dir = f"registration_results_{method_name.lower()}"
    print(f"结果目录: {result_dir}/")
    print("输出文件:")
    for i in range(len(registered_clouds)):
        print(f"  - registered_cloud_{i+1}.ply (单独的配准后点云 {i+1})")
    print(f"  - merged_registered_clouds2.ply (包含所有5个配准后点云的合并文件)")
    print("变换矩阵:")
    for i in range(len(transformations)):
        print(f"  - transformation_matrix_{i+1}.txt (点云 {i+1} 的变换矩阵)")
    print("配准信息:")
    print(f"  - registration_info.txt (详细的配准过程和结果信息)")
    print(f"\n📁 主要文件: {result_dir}/merged_registered_clouds.ply")
    print("   ↳ 这个文件包含了所有5个配准后的点云，可以直接在点云查看器中打开")

if __name__ == "__main__":
    main()
