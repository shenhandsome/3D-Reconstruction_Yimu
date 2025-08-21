import open3d as o3d
import numpy as np

def load_patch(idx):
    return o3d.io.read_point_cloud(f"examples/data/08060{idx}.ply")

def apply_transform(pcd, transform):
    pcd_copy = pcd.translate((0,0,0), relative=False)
    pcd_copy.transform(transform)
    return pcd_copy

#加载点云
pcd1 = load_patch(1)  # 蓝
pcd2 = load_patch(2)  # 粉
pcd3 = load_patch(3)  # 红
pcd4 = load_patch(4)  # 绿


def create_transform(angle_deg, radius=0.009):
    """最简单的变换矩阵构建"""
    theta = np.radians(angle_deg)
    
    # 直接构建变换矩阵，避免复杂的矩阵运算
    transform = np.eye(4)
    
    # 旋转和平移同时设置
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # 旋转矩阵（绕Y轴）
    transform[0, 0] = cos_theta
    transform[0, 2] = sin_theta
    transform[2, 0] = -sin_theta
    transform[2, 2] = cos_theta
    
    # 平移向量
    transform[0, 3] = radius * cos_theta-0.002
    transform[2, 3] = radius * sin_theta+0.008
    
    return transform



def create_transform2(angle_deg, radius=0.009):
    """最简单的变换矩阵构建"""
    theta = np.radians(angle_deg)
    
    # 直接构建变换矩阵，避免复杂的矩阵运算
    transform = np.eye(4)
    
    # 旋转和平移同时设置
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # 旋转矩阵（绕Y轴）
    transform[0, 0] = cos_theta
    transform[0, 2] = sin_theta
    transform[2, 0] = -sin_theta
    transform[2, 2] = cos_theta
    
    # 平移向量
    transform[0, 3] = radius * cos_theta-0.012
    transform[2, 3] = radius * sin_theta+0.002
    
    return transform


def create_transform3(angle_deg, radius=0.009):
    """最简单的变换矩阵构建"""
    theta = np.radians(angle_deg)
    
    # 直接构建变换矩阵，避免复杂的矩阵运算
    transform = np.eye(4)
    
    # 旋转和平移同时设置
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # 旋转矩阵（绕Y轴）
    transform[0, 0] = cos_theta
    transform[0, 2] = sin_theta
    transform[2, 0] = -sin_theta
    transform[2, 2] = cos_theta
    
    # 平移向量
    #横
    transform[0, 3] = radius * cos_theta -0.0102
    #竖
    transform[2, 3] = radius * sin_theta +0.011

    return transform

import numpy as np

def create_transform4(angle_deg, radius=0.009):
    """创建绕X轴的变换矩阵"""
    theta = np.radians(angle_deg)
    
    # 初始化单位矩阵
    transform = np.eye(4)
    
    # 计算三角函数值
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # 绕X轴的旋转矩阵（影响Y和Z轴）
    transform[1, 1] = cos_theta
    transform[1, 2] = -sin_theta  # 注意负号位置
    transform[2, 1] = sin_theta
    transform[2, 2] = cos_theta
    
    # 平移向量（在YZ平面内移动）
    transform[1, 3] = radius * cos_theta+0.01  # Y方向平移
    transform[2, 3] = radius * sin_theta  # Z方向平移
    
    return transform



#应用粗略摆放：假设围绕 y 轴均匀分布
pcd2 = apply_transform(pcd2, create_transform(180))
pcd3 = apply_transform(pcd3, create_transform2(180))
pcd4 = apply_transform(pcd4, create_transform4(180))
pcd4 = apply_transform(pcd4, create_transform3(270))

#合并为一个闭合结构
merged = pcd1 + pcd2 + pcd3 + pcd4


#可视化检查
o3d.visualization.draw_geometries([merged])

# 1. 统计离群点移除 - 清除噪声和纹理不规则性
print("📊 步骤1: 移除统计离群点")
cl, ind = merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
merged_processed = merged.select_by_index(ind)
print(f"  移除离群点后点数: {len(merged_processed.points)}")
o3d.visualization.draw_geometries([merged_processed])




# 2. 体素下采样 - 平滑表面并降低密度
print("📊 步骤2: 体素下采样")
voxel_size = 0.0001  # 控制平滑程度
merged_processed = merged_processed.voxel_down_sample(voxel_size=voxel_size)
print(f"  下采样后点数: {len(merged_processed.points)}")
o3d.visualization.draw_geometries([merged_processed])

# 3. 重新估计法向量
print("📊 步骤3: 重新估计法向量")
merged_processed.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30)
)
merged_processed.orient_normals_to_align_with_direction([0., 0., 1.])
o3d.visualization.draw_geometries([merged_processed])


# 4. 半径离群点移除 - 使形状更接近圆环
print("📊 步骤4: 半径离群点移除")
cl, ind = merged_processed.remove_radius_outlier(nb_points=16, radius=0.009)
merged_processed = merged_processed.select_by_index(ind)
print(f"  半径离群点移除后点数: {len(merged_processed.points)}")
o3d.visualization.draw_geometries([merged_processed])

# 5. 最终平滑处理
print("📊 步骤5: 最终平滑处理")
merged_processed = merged_processed.voxel_down_sample(voxel_size=0.0001)
merged_processed.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.003, max_nn=30)
)
o3d.visualization.draw_geometries([merged_processed])





print("📊 步骤6: Alpha Shape SDF重建")
try:
    # Alpha Shape重建 - 最适合圆环，需要指定alpha参数
    alpha = 0.01  # 调整这个值控制重建的紧密程度
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(merged_processed, alpha)
    
    # 网格平滑
    mesh = mesh.filter_smooth_simple(number_of_iterations=3)
    mesh = mesh.filter_smooth_taubin(number_of_iterations=5)
    
    # 从平滑后的网格重新采样点云
    merged_sdf = mesh.sample_points_uniformly(number_of_points=len(merged_processed.points)*2)
    merged_sdf.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.003, max_nn=30)
    )
    
    print("✅ Alpha Shape SDF处理完成")
    o3d.visualization.draw_geometries([merged_sdf])
    
    # # 保存结果
    o3d.io.write_point_cloud("merged_ring_smooth_sdf.ply", merged_sdf)
    o3d.io.write_triangle_mesh("merged_ring_mesh.ply", mesh)
    
except Exception as e:
    print(f"⚠️ Alpha Shape失败，使用体素重建: {e}")
















# #可视化检查
# o3d.visualization.draw_geometries([merged])