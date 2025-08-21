
##########表面重建成功########
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
pcd4 = apply_transform(pcd4, create_transform3(240))

#合并为一个闭合结构
merged = pcd1 + pcd2 + pcd3 + pcd4


#可视化检查
o3d.visualization.draw_geometries([merged])

print(f"拼接后点数：{len(merged.points)}")

# ===== 用你已有的 merged 点云 =====
pcd_raw = merged

# --- 0) 清洗点云 ---
# 去掉 NaN / Inf
pts_np = np.asarray(pcd_raw.points)
mask = np.isfinite(pts_np).all(axis=1)
if pcd_raw.has_colors():
    cols_np = np.asarray(pcd_raw.colors)
    mask &= np.isfinite(cols_np).all(axis=1)
pcd_raw = pcd_raw.select_by_index(np.where(mask)[0])

# 去重
pcd_raw.remove_duplicated_points()

# --- 1) 估计法线（BPA 需要法线） ---
diag = np.linalg.norm(pcd_raw.get_max_bound() - pcd_raw.get_min_bound())
pcd_raw.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=diag*0.03, max_nn=50)
)

# --- 2) 估计平均点距，用于 BPA 半径 ---
pcd_tree = o3d.geometry.KDTreeFlann(pcd_raw)
dists = []
pts_np = np.asarray(pcd_raw.points)
sample_n = min(2000, len(pts_np))
for i in range(sample_n):
    k, idx, _ = pcd_tree.search_knn_vector_3d(pts_np[i], 6)
    if k < 2:
        continue
    nn_pts = pts_np[idx[1:k]]
    d = np.linalg.norm(nn_pts - pts_np[i], axis=1).mean()
    dists.append(d)

if len(dists) == 0:
    avg_dist = max(diag * 1e-3, 1e-6)  # 兜底估计
else:
    avg_dist = float(np.median(dists))

print(f"[Info] 平均点距估计: {avg_dist:.6f}")

# --- 3) Ball Pivoting 网格重建 ---
radii = o3d.utility.DoubleVector([avg_dist*1.0, avg_dist*1.5, avg_dist*2.0])
mesh_bpa = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd_raw, radii
)

# 基础清理
mesh_bpa.remove_degenerate_triangles()
mesh_bpa.remove_duplicated_vertices()
mesh_bpa.remove_duplicated_triangles()
mesh_bpa.remove_non_manifold_edges()
mesh_bpa.compute_vertex_normals()

o3d.io.write_triangle_mesh("ring_bpa_raw.ply", mesh_bpa, write_vertex_normals=True)
print("[Save] ring_bpa_raw.ply 已保存")

# --- 4) 平滑处理（可选） ---
mesh_bpa = mesh_bpa.filter_smooth_taubin(number_of_iterations=1)
mesh_bpa.compute_vertex_normals()

# --- 5) Shrinkwrap 回投细节 ---
print("[Shrinkwrap] 开始回投细节...")
tree = o3d.geometry.KDTreeFlann(pcd_raw)
verts = np.asarray(mesh_bpa.vertices)
new_verts = verts.copy()

alpha = 0.5  # 回投强度
passes = 1   # 回投次数
fallback_radius = avg_dist * 3.0  # 半径兜底

for _ in range(passes):
    for i, v in enumerate(new_verts):
        k, idx, _ = tree.search_knn_vector_3d(v, 1)
        if k >= 1:
            nn = pts_np[idx[0]]
        else:
            # 半径搜索兜底
            k2, idx2, _ = tree.search_radius_vector_3d(v, fallback_radius)
            if k2 >= 1:
                nn = pts_np[idx2[0]]
            else:
                continue  # 实在没邻居就跳过
        new_verts[i] = (1 - alpha) * new_verts[i] + alpha * nn

mesh_bpa.vertices = o3d.utility.Vector3dVector(new_verts)
mesh_bpa.compute_vertex_normals()

# --- 6) 顶点色转移（如有颜色） ---
if pcd_raw.has_colors():
    print("[Color] 从点云烘焙颜色到网格...")
    colors = []
    raw_cols = np.asarray(pcd_raw.colors)
    for v in np.asarray(mesh_bpa.vertices):
        k, idx, _ = tree.search_knn_vector_3d(v, 1)
        if k >= 1:
            colors.append(raw_cols[idx[0]])
        else:
            # 半径兜底
            k2, idx2, _ = tree.search_radius_vector_3d(v, fallback_radius)
            if k2 >= 1:
                colors.append(raw_cols[idx2[0]])
            else:
                colors.append([0.5, 0.5, 0.5])  # 无邻居则灰色
    mesh_bpa.vertex_colors = o3d.utility.Vector3dVector(np.asarray(colors))

# --- 7) 保存最终结果 ---
o3d.io.write_triangle_mesh("ring_bpa_shrinkwrap_colored.ply", mesh_bpa,
                           write_vertex_normals=True, write_vertex_colors=True)
print("[Save] ring_bpa_shrinkwrap_colored.ply 已保存")

# --- 8) 可视化 ---
o3d.visualization.draw_geometries([mesh_bpa])
