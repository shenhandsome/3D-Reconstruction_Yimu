# import os
# import numpy as np
# import open3d as o3d
# import torch
# import turboreg_gpu
# from glob import glob
# from sklearn.neighbors import NearestNeighbors
# from utils_pcr import compute_transformation_error
# import time

# data_dir = "../demo_data"
# voxel_size = 1

# def numpy_to_torch32(device, *arrays):
#     return [torch.tensor(array, device=device, dtype=torch.float32) for array in arrays]

# def load_data(idx_str):
#     src = o3d.io.read_point_cloud(os.path.join(data_dir, f"{idx_str}_pts_src.ply"))
#     dst = o3d.io.read_point_cloud(os.path.join(data_dir, f"{idx_str}_pts_dst.ply"))
#     trans_gt = np.loadtxt(os.path.join(data_dir, f"{idx_str}_trans.txt"))
#     return src, dst, trans_gt

# def preprocess(pcd, voxel_size):
#     pcd_down = pcd.voxel_down_sample(voxel_size)
#     pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*3, max_nn=30))
#     fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#         pcd_down,
#         o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100)
#     )
#     return pcd_down, fpfh

# def draw_registration(src, dst, trans, title=""):
#     src_temp = src.transform(trans.copy())
#     src_temp.paint_uniform_color([1, 1, 0])
#     dst.paint_uniform_color([0, 1, 1])
#     o3d.visualization.draw_geometries([src_temp, dst], window_name=title)

# idx_list = sorted(set([os.path.basename(f).split('_')[0] for f in glob(os.path.join(data_dir, "*_pts_src.ply"))]))


# reger = turboreg_gpu.TurboRegGPU(6000, 0.1, 2500, 0.15, 0.4, "IN")
# while True:
#     for idx_str in idx_list:
#         print(f"\nProcessing index: {idx_str}")
#         src, dst, trans_gt = load_data(idx_str)

#         tmp_kpts_src = os.path.join(data_dir, f"{idx_str}_fpfh_kpts_src.txt")
#         tmp_kpts_dst = os.path.join(data_dir, f"{idx_str}_fpfh_kpts_dst.txt")
#         if os.path.exists(tmp_kpts_src) and os.path.exists(tmp_kpts_dst):
#             kpts_src = np.loadtxt(tmp_kpts_src)
#             kpts_dst = np.loadtxt(tmp_kpts_dst)
#         else:
#             src_down, src_fpfh = preprocess(src, voxel_size)
#             dst_down, dst_fpfh = preprocess(dst, voxel_size)

#             src_feats = np.array(src_fpfh.data).T
#             dst_feats = np.array(dst_fpfh.data).T

#             nn = NearestNeighbors(n_neighbors=1).fit(dst_feats)
#             _, indices = nn.kneighbors(src_feats)

#             kpts_src = np.asarray(src_down.points)
#             kpts_dst = np.asarray(dst_down.points)[indices[:, 0]]

#             np.savetxt(tmp_kpts_src, kpts_src)
#             np.savetxt(tmp_kpts_dst, kpts_dst)

#         # TurboReg
#         T0 = time.time()
#         kpts_src_torch, kpts_dst_torch = numpy_to_torch32("cuda:0", kpts_src, kpts_dst)
#         TIME_data_to_gpu = time.time() - T0
#         T1 = time.time()
#         trans = reger.run_reg(kpts_src_torch, kpts_dst_torch).cpu().numpy()
#         TIME_reg = time.time() - T1
#         print(trans, '\n', trans_gt)
#         rre, rte = compute_transformation_error(trans_gt, trans)
#         is_succ = (rre < 5) & (rte < 0.6)
#         print("SUCC: ", is_succ, " TIME_data_to_gpu: {:.3f} (ms)".format(TIME_data_to_gpu * 1000), " TIME_reg: {:.3f} (ms)".format(TIME_reg * 1000))






# import os
# import numpy as np
# import open3d as o3d
# import torch
# import turboreg_gpu
# from glob import glob
# from sklearn.neighbors import NearestNeighbors
# from utils_pcr import compute_transformation_error
# import time
# import copy

# # 修改为你的两个PLY文件路径
# src_file = "/root/autodl-tmp/dianyun/TurboReg/demo_point/0001_pointcloud_2.ply"  # 请替换为你的源点云文件路径demo_point/0001_pointcloud.ply
# dst_file = "/root/autodl-tmp/dianyun/TurboReg/demo_point/0003_pointcloud_2.ply"  # 请替换为你的目标点云文件路径demo_point/0003_pointcloud.ply
# voxel_size = 1

# def numpy_to_torch32(device, *arrays):
#     return [torch.tensor(array, device=device, dtype=torch.float32) for array in arrays]

# def load_data():
#     """加载两个指定的PLY文件"""
#     src = o3d.io.read_point_cloud(src_file)
#     dst = o3d.io.read_point_cloud(dst_file)
#     return src, dst

# def preprocess(pcd, voxel_size):
#     pcd_down = pcd.voxel_down_sample(voxel_size)
#     pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*3, max_nn=30))
#     fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#         pcd_down,
#         o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100)
#     )
#     return pcd_down, fpfh

# def draw_registration(src, dst, trans, title=""):
#     src_temp = src.transform(trans.copy())
#     src_temp.paint_uniform_color([1, 1, 0])
#     dst.paint_uniform_color([0, 1, 1])
#     o3d.visualization.draw_geometries([src_temp, dst], window_name=title)

# def main():
#     """主函数：加载两个PLY文件进行配准并可视化"""
#     print("Loading point clouds...")
#     src, dst = load_data()
    
#     print(f"Source point cloud: {len(src.points)} points")
#     print(f"Target point cloud: {len(dst.points)} points")
    
#     # 显示原始点云（配准前）
#     print("Displaying original point clouds...")
#     # src_orig = src.copy()
#     # dst_orig = dst.copy()
#     src_orig = copy.deepcopy(src)
#     dst_orig = copy.deepcopy(dst)
#     src_orig.paint_uniform_color([1, 0, 0])  # 红色
#     dst_orig.paint_uniform_color([0, 0, 1])  # 蓝色
#     o3d.visualization.draw_geometries([src_orig, dst_orig], 
#                                       window_name="Original Point Clouds (Red: Source, Blue: Target)")
    
#     # 检查是否存在缓存的关键点文件
#     tmp_kpts_src = "fpfh_kpts_src.txt"
#     tmp_kpts_dst = "fpfh_kpts_dst.txt"
    
#     if os.path.exists(tmp_kpts_src) and os.path.exists(tmp_kpts_dst):
#         print("Loading cached keypoints...")
#         kpts_src = np.loadtxt(tmp_kpts_src)
#         kpts_dst = np.loadtxt(tmp_kpts_dst)
#     else:
#         print("Computing FPFH features and finding correspondences...")
#         src_down, src_fpfh = preprocess(src, voxel_size)
#         dst_down, dst_fpfh = preprocess(dst, voxel_size)
        
#         print(f"Downsampled source: {len(src_down.points)} points")
#         print(f"Downsampled target: {len(dst_down.points)} points")

#         src_feats = np.array(src_fpfh.data).T
#         dst_feats = np.array(dst_fpfh.data).T

#         nn = NearestNeighbors(n_neighbors=1).fit(dst_feats)
#         _, indices = nn.kneighbors(src_feats)

#         kpts_src = np.asarray(src_down.points)
#         kpts_dst = np.asarray(dst_down.points)[indices[:, 0]]

#         # 保存关键点以便下次使用
#         np.savetxt(tmp_kpts_src, kpts_src)
#         np.savetxt(tmp_kpts_dst, kpts_dst)
#         print(f"Saved keypoints to {tmp_kpts_src} and {tmp_kpts_dst}")

#     print(f"Number of correspondences: {len(kpts_src)}")
    
#     # TurboReg配准
#     print("Running TurboReg registration...")
#     reger = turboreg_gpu.TurboRegGPU(6000, 0.1, 2500, 0.15, 0.4, "IN")
    
#     T0 = time.time()
#     kpts_src_torch, kpts_dst_torch = numpy_to_torch32("cuda:0", kpts_src, kpts_dst)
#     TIME_data_to_gpu = time.time() - T0
    
#     T1 = time.time()
#     trans = reger.run_reg(kpts_src_torch, kpts_dst_torch).cpu().numpy()
#     TIME_reg = time.time() - T1
    
#     print(f"Registration completed!")
#     print(f"Data to GPU time: {TIME_data_to_gpu * 1000:.3f} ms")
#     print(f"Registration time: {TIME_reg * 1000:.3f} ms")
#     print("Transformation matrix:")
#     print(trans)
    
#     # 可视化配准结果
#     print("Displaying registration result...")
#     draw_registration(src, dst, trans, "Registration Result (Yellow: Transformed Source, Cyan: Target)")
    
#     return trans

# if __name__ == "__main__":
#     main()







#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import open3d as o3d
import torch
import turboreg_gpu
from sklearn.neighbors import NearestNeighbors
import copy
import time
import random


src_file = "/home/shen/dianyun/TurboReg/demo_point/080601.ply"
dst_file = "/home/shen/dianyun/TurboReg/demo_point/080602.ply"



# ========= 其他配置 =========
ENABLE_VISUALIZATION = False   # 无显示环境请设为 False
SAVE_DIR = "./reg_results"     # 输出目录
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- 工具函数 ----------------
def log(s): print(s, flush=True)

def guess_scale(pcd: o3d.geometry.PointCloud):
    """按包围盒对角线估计场景尺度，并给出建议 voxel."""
    aabb = pcd.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(aabb.get_extent())
    # 典型经验：取对角线的 1% 做体素（限制在[2mm,5cm]）
    v = np.clip(diag * 0.01, 0.002, 0.05)
    return float(v)

def preprocess(pcd: o3d.geometry.PointCloud, voxel=None):
    """下采样 + 法线 + FPFH，半径按 voxel 自适应"""
    if voxel is None:
        voxel = guess_scale(pcd)
    pcd_down = pcd.voxel_down_sample(voxel)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*2.0, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*5.0, max_nn=100)
    )
    return pcd_down, fpfh, voxel

def build_correspondences(src_fpfh, dst_fpfh, top_percent=0.15, ratio_thresh=0.9, min_take=500):
    """双向一致 + ratio test + 分位数截断，返回 (idx_src, idx_dst)"""
    src_feats = np.asarray(src_fpfh.data).T  # [Ns, 33]
    dst_feats = np.asarray(dst_fpfh.data).T  # [Nd, 33]
    Ns, Nd = len(src_feats), len(dst_feats)
    # ratio test
    nn_dst = NearestNeighbors(n_neighbors=2).fit(dst_feats)
    d2, idx2 = nn_dst.kneighbors(src_feats, n_neighbors=2)
    ratio = d2[:,0] / (d2[:,1] + 1e-9)
    mask_ratio = ratio < ratio_thresh
    cand = np.where(mask_ratio)[0]  # src indices

    # 分位数截断（只留前 top_percent 的最小距离）
    d_sorted = np.sort(d2[mask_ratio, 0])
    if len(d_sorted) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    top_k = max(min_take, int(top_percent * len(d_sorted)))
    top_k = min(top_k, len(d_sorted))
    th = d_sorted[top_k-1]  # 阈值
    keep = np.where(mask_ratio & (d2[:,0] <= th))[0]

    # 双向一致
    nn_src = NearestNeighbors(n_neighbors=1).fit(src_feats)
    d_back, i_back = nn_src.kneighbors(dst_feats, n_neighbors=1)  # for mutual check

    pairs = []
    for i in keep:
        j = idx2[i,0]
        if i_back[j,0] == i:  # mutual
            pairs.append((i, j))

    # 去重（保持顺序）
    pairs = list(dict.fromkeys(pairs))
    if len(pairs) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    idx_src = np.array([p[0] for p in pairs], dtype=int)
    idx_dst = np.array([p[1] for p in pairs], dtype=int)
    return idx_src, idx_dst

def filter_too_close(src_pts, idx_src, idx_dst, min_sep):
    """按源点最近邻距离过滤过近点，减少退化三角形"""
    sel_src = src_pts[idx_src]
    if len(sel_src) < 2:
        return idx_src, idx_dst
    nn = NearestNeighbors(n_neighbors=2).fit(sel_src)
    d, _ = nn.kneighbors(sel_src)
    keep_mask = d[:,1] > min_sep
    return idx_src[keep_mask], idx_dst[keep_mask]

def numpy_to_torch32_cuda(arr):
    return torch.as_tensor(arr, device="cuda:0", dtype=torch.float32).contiguous()

def evaluate_mean_distance(src_pc: o3d.geometry.PointCloud, dst_pc: o3d.geometry.PointCloud):
    d = np.asarray(src_pc.compute_point_cloud_distance(dst_pc))
    return float(d.mean()) if len(d) > 0 else np.inf

def draw_registration(src, dst, trans, title=""):
    if not ENABLE_VISUALIZATION:
        log(f"(headless) skip visualization: {title}")
        return
    s = copy.deepcopy(src); t = copy.deepcopy(dst)
    s.transform(trans)
    s.paint_uniform_color([1,1,0])  # 黄
    t.paint_uniform_color([0,1,1])  # 青
    o3d.visualization.draw_geometries([s,t], window_name=title)

# ---------------- TurboReg 主流程 ----------------
def run_turboreg(kpts_src_xyz, kpts_dst_xyz, voxel):
    """按尺度自适应阈值跑 TurboReg，返回 4x4"""
    N = len(kpts_src_xyz)
    if N < 100:
        raise RuntimeError(f"correspondences too few for TurboReg: {N}")

    # 打散并截断到 max_N
    max_N = min(6000, N)
    perm = np.random.permutation(N)
    sel = perm[:max_N]
    ksrc = kpts_src_xyz[sel]
    kdst = kpts_dst_xyz[sel]

    # 自适应超参（可按需要微调）
    tau_length_consis = 1.5 * voxel
    radiu_nms        = 2.0 * voxel
    tau_inlier       = 3.0 * voxel
    num_pivot        = min(2500, len(ksrc) // 3)

    log(f"TurboReg params: max_N={len(ksrc)}, num_pivot={num_pivot}, "
        f"tau_len={tau_length_consis:.4f}, nms={radiu_nms:.4f}, inlier={tau_inlier:.4f}")

    # to CUDA
    src_t = numpy_to_torch32_cuda(ksrc)
    dst_t = numpy_to_torch32_cuda(kdst)
    torch.cuda.synchronize()

    reger = turboreg_gpu.TurboRegGPU(
        len(ksrc), float(tau_length_consis), int(num_pivot),
        float(radiu_nms), float(tau_inlier), "IN"
    )

    t0 = time.time()
    T = reger.run_reg(src_t, dst_t).cpu().numpy()
    torch.cuda.synchronize()
    log(f"TurboReg time: {(time.time()-t0)*1000:.2f} ms")
    return T

# ---------------- 兜底（FGR + ICP） ----------------
def fallback_fgr_icp(src_down, dst_down, voxel):
    """用 Open3D 的 Fast Global Registration + ICP 做备选"""
    radius_normal = voxel * 2.0
    radius_feature = voxel * 5.0

    # FGR 期望已有 FPFH，这里重算一次以简化依赖
    src_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    dst_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        src_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    dst_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        dst_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )

    option = o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=3.0*voxel
    )
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        src_down, dst_down, src_fpfh, dst_fpfh, option
    )
    T_init = result.transformation

    # ICP 精化
    icp = o3d.pipelines.registration.registration_icp(
        src_down, dst_down, max_correspondence_distance=2.0*voxel,
        init=T_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    return icp.transformation

# ---------------- 主程序 ----------------
def main():
    assert torch.cuda.is_available(), "需要 CUDA 环境来运行 turboreg_gpu"

    log("Loading point clouds...")
    src = o3d.io.read_point_cloud(src_file)
    dst = o3d.io.read_point_cloud(dst_file)
    log(f"Source points: {len(src.points)}, Target points: {len(dst.points)}")

    # 估计尺度并预处理（尝试三档 voxel）
    base_voxel = max(guess_scale(src), guess_scale(dst))
    voxel_list = [base_voxel, base_voxel*1.5, base_voxel*0.7]
    log(f"Base voxel={base_voxel:.5f}, try {['%.5f'%v for v in voxel_list]}")

    best = dict(n_corr=0)

    for v in voxel_list:
        log(f"\n>> Preprocess with voxel={v:.5f}")
        src_down, src_fpfh, _ = preprocess(src, v)
        dst_down, dst_fpfh, _ = preprocess(dst, v)
        log(f"Downsampled: src={len(src_down.points)}, dst={len(dst_down.points)}")

        idx_s, idx_d = build_correspondences(src_fpfh, dst_fpfh,
                                             top_percent=0.15, ratio_thresh=0.9, min_take=500)
        log(f"Initial correspondences: {len(idx_s)}")

        # 过滤过近点，防退化三角形
        src_pts_down = np.asarray(src_down.points)
        idx_s, idx_d = filter_too_close(src_pts_down, idx_s, idx_d, min_sep=0.5*v)
        log(f"After min-sep filtering: {len(idx_s)}")

        if len(idx_s) > best.get("n_corr", 0):
            best = dict(
                voxel=v,
                src_down=src_down, dst_down=dst_down,
                idx_s=idx_s, idx_d=idx_d,
                n_corr=len(idx_s)
            )

    if best["n_corr"] < 100:
        log(f"\n❌ 对应关系过少（{best['n_corr']}）——改用 FGR+ICP 兜底。")
        # 直接兜底
        v = base_voxel
        src_down, _, _ = preprocess(src, v)
        dst_down, _, _ = preprocess(dst, v)
        T = fallback_fgr_icp(src_down, dst_down, v)
    else:
        log(f"\nBest voxel={best['voxel']:.5f}, correspondences={best['n_corr']}")
        # 准备对应点坐标
        kpts_src = np.asarray(best["src_down"].points)[best["idx_s"]]
        kpts_dst = np.asarray(best["dst_down"].points)[best["idx_d"]]

        # TurboReg 主跑，失败则兜底
        try:
            T = run_turboreg(kpts_src, kpts_dst, best["voxel"])
        except Exception as e:
            log(f"TurboReg failed: {e}\n-> Falling back to FGR+ICP")
            T = fallback_fgr_icp(best["src_down"], best["dst_down"], best["voxel"])

    # ------- 评估与输出 -------
    log("\nEvaluating alignment...")
    src_tr = copy.deepcopy(src); src_tr.transform(T)
    # 用下采样目标评估
    dst_eval, _, _ = preprocess(dst, guess_scale(dst))
    mean_err_ds = evaluate_mean_distance(src_tr, dst_eval)
    # 可选：全分辨率复核（仅报告，不作为判断）
    mean_err_full = evaluate_mean_distance(src_tr, dst)

    log(f"Mean error (↓) vs downsampled target: {mean_err_ds:.6f}")
    log(f"Mean error (↓) vs full target      : {mean_err_full:.6f}")
    if mean_err_ds < 3.0 * guess_scale(dst):
        log("✅ Alignment quality: OK")
    else:
        log("⚠️ Alignment quality: might be improvable (consider tuning voxel/ratio)")

    # 保存结果
    np.save(os.path.join(SAVE_DIR, "transform.npy"), T)
    o3d.io.write_point_cloud(os.path.join(SAVE_DIR, "ring_turboreg.ply"), src_tr)
    log(f"Saved: {os.path.join(SAVE_DIR, 'transform.npy')} and source_transformed.ply")

    # 可视化
    log("Displaying registration result...")
    draw_registration(src, dst, T, "Registration (Yellow=Transformed Source, Cyan=Target)")

    # 打印最终 4x4
    np.set_printoptions(precision=6, suppress=True)
    log("Final 4x4 transform:\n" + str(T))

if __name__ == "__main__":
    main()

