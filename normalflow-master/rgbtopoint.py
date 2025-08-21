import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import cv2
import matplotlib.pyplot as plt

# 使用真实的相机内参
depth_scale = 0.001  # 深度值的转换因子，具体根据相机配置

# 自动检测深度尺度
def auto_detect_depth_scale(depth_image):
    """
    自动检测深度图的尺度
    """
    valid_depths = depth_image[depth_image > 0]
    if len(valid_depths) == 0:
        return 0.001  # 默认值
    
    depth_range = valid_depths.max() - valid_depths.min()
    depth_mean = valid_depths.mean()
    
    print(f"深度值范围: {valid_depths.min()} - {valid_depths.max()}")
    print(f"深度值平均值: {depth_mean:.2f}")
    
    # 如果深度值在几百到几万之间，可能是毫米单位
    if 100 < depth_mean < 10000:
        print("检测到深度值可能是毫米单位，使用 depth_scale = 0.001 (毫米转米)")
        return 0.001
    # 如果深度值在几万到几十万之间，可能是微米单位
    elif 10000 < depth_mean < 100000:
        print("检测到深度值可能是微米单位，使用 depth_scale = 0.000001 (微米转米)")
        return 0.000001
    # 如果深度值很小，可能已经是米单位
    elif depth_mean < 10:
        print("检测到深度值可能是米单位，使用 depth_scale = 1.0")
        return 1.0
    else:
        print(f"无法确定深度单位，使用默认值 depth_scale = 0.001")
        return 0.001

# RGB和深度相机的内参（根据您提供的数据）
intrinsics = rs.intrinsics()
intrinsics.width = 640
intrinsics.height = 480
intrinsics.ppx = 325.4742431640625
intrinsics.ppy = 246.349609375
intrinsics.fx = 607.806884765625
intrinsics.fy = 607.7139892578125
intrinsics.model = rs.distortion.none
intrinsics.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

# 将深度值转换为点云
def create_point_cloud(depth_image, color_image, depth_scale=None):
    if depth_scale is None:
        depth_scale = auto_detect_depth_scale(depth_image)
    
    points = []
    colors = []
    
    print(f"使用深度尺度: {depth_scale}")
    
    for v in range(depth_image.shape[0]):
        for u in range(depth_image.shape[1]):
            z = depth_image[v, u] * depth_scale  # 将深度值换算成实际的距离
            if z > 0:  # 仅处理有效的点
                x = (u - intrinsics.ppx) * z / intrinsics.fx
                y = (v - intrinsics.ppy) * z / intrinsics.fy
                points.append([x, y, z])
                colors.append(color_image[v, u] / 255.0)  # 归一化颜色值

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
    point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    return point_cloud

# 保存点云为PLY文件
def save_point_cloud_to_ply(point_cloud, filename):
    o3d.io.write_point_cloud(filename, point_cloud)
    print(f"点云已保存到: {filename}")

# 示例：从RealSense相机获取数据并创建点云
def capture_and_save_point_cloud():
    # 创建RealSense管道
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        # 启动管道
        profile = pipeline.start(config)
        
        # 等待获取第一帧
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            print("无法获取深度或彩色帧")
            return
        
        # 转换为numpy数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # 创建点云
        pcd = create_point_cloud(depth_image, color_image)
        
        # 保存为PLY文件
        filename = "captured_pointcloud.ply"
        save_point_cloud_to_ply(pcd, filename)
        
        # 可视化点云
        o3d.visualization.draw_geometries([pcd])
        
    except Exception as e:
        print(f"错误: {e}")
    finally:
        pipeline.stop()

# 如果您已经有深度图和彩色图像数据，可以直接使用这个函数
def process_existing_data(depth_image, color_image, output_filename="output_pointcloud.ply"):
    """
    处理已有的深度图和彩色图像数据
    
    Args:
        depth_image: 深度图像 (numpy数组)
        color_image: 彩色图像 (numpy数组)
        output_filename: 输出的PLY文件名
    """
    pcd = create_point_cloud(depth_image, color_image)
    save_point_cloud_to_ply(pcd, output_filename)
    return pcd

# 读取图片文件并创建点云
def read_images_and_create_pointcloud(depth_image_path, rgb_image_path, output_filename="output_pointcloud.ply"):
    """
    读取深度图和RGB图像文件，创建点云并保存为PLY文件
    
    Args:
        depth_image_path: 深度图像文件路径
        rgb_image_path: RGB图像文件路径
        output_filename: 输出的PLY文件名
    """
    try:
        # 读取深度图像
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
        if depth_image is None:
            print(f"无法读取深度图像: {depth_image_path}")
            return None
            
        # 读取RGB图像
        color_image = cv2.imread(rgb_image_path, cv2.IMREAD_COLOR)
        if color_image is None:
            print(f"无法读取RGB图像: {rgb_image_path}")
            return None
            
        print(f"深度图像尺寸: {depth_image.shape}")
        print(f"RGB图像尺寸: {color_image.shape}")
        
        # 显示深度图信息
        print(f"深度图数据类型: {depth_image.dtype}")
        print(f"深度图最小值: {depth_image.min()}")
        print(f"深度图最大值: {depth_image.max()}")
        print(f"深度图平均值: {depth_image.mean():.2f}")
        
        # 自动检测深度尺度
        global depth_scale
        depth_scale = auto_detect_depth_scale(depth_image)
        print(f"使用的深度尺度: {depth_scale}")
        
        # 可视化深度图
        visualize_depth_image(depth_image)
        
        # 检查图像尺寸是否匹配
        if depth_image.shape[:2] != color_image.shape[:2]:
            print("警告：深度图和RGB图尺寸不匹配，将调整RGB图尺寸")
            color_image = cv2.resize(color_image, (depth_image.shape[1], depth_image.shape[0]))
        
        # 创建点云
        pcd = create_point_cloud(depth_image, color_image)
        
        # 保存为PLY文件
        save_point_cloud_to_ply(pcd, output_filename)
        
        print(f"成功创建点云，包含 {len(pcd.points)} 个点")
        return pcd
        
    except Exception as e:
        print(f"处理图像时出错: {e}")
        return None

# 可视化深度图
def visualize_depth_image(depth_image):
    """
    可视化深度图，显示深度数据的分布
    """
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 原始深度图
    im1 = axes[0, 0].imshow(depth_image, cmap='viridis')
    axes[0, 0].set_title('原始深度图')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 深度图直方图
    valid_depths = depth_image[depth_image > 0]
    axes[0, 1].hist(valid_depths.flatten(), bins=50, alpha=0.7)
    axes[0, 1].set_title('深度值分布')
    axes[0, 1].set_xlabel('深度值')
    axes[0, 1].set_ylabel('像素数量')
    
    # 归一化深度图（用于显示）
    depth_normalized = depth_image.astype(np.float32)
    if depth_normalized.max() > 0:
        depth_normalized = (depth_normalized - depth_normalized.min()) / (depth_normalized.max() - depth_normalized.min())
    
    im2 = axes[1, 0].imshow(depth_normalized, cmap='plasma')
    axes[1, 0].set_title('归一化深度图')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # 深度图统计信息
    axes[1, 1].text(0.1, 0.8, f'图像尺寸: {depth_image.shape}', fontsize=12)
    axes[1, 1].text(0.1, 0.7, f'数据类型: {depth_image.dtype}', fontsize=12)
    axes[1, 1].text(0.1, 0.6, f'最小值: {depth_image.min()}', fontsize=12)
    axes[1, 1].text(0.1, 0.5, f'最大值: {depth_image.max()}', fontsize=12)
    axes[1, 1].text(0.1, 0.4, f'平均值: {depth_image.mean():.2f}', fontsize=12)
    axes[1, 1].text(0.1, 0.3, f'有效像素: {np.sum(depth_image > 0)}', fontsize=12)
    axes[1, 1].text(0.1, 0.2, f'总像素: {depth_image.size}', fontsize=12)
    axes[1, 1].set_title('深度图统计信息')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("深度图可视化完成！")
    print(f"\n数据类型分析: {depth_image.dtype}")
    
    if depth_image.dtype == np.uint16:
        print("检测到uint16数据类型，这是正常的深度图格式")
        print("uint16范围: 0 - 65535")
        print("RealSense相机通常使用毫米作为深度单位")
        print("建议的depth_scale值:")
        print("  - 如果深度值是毫米: depth_scale = 0.001 (毫米转米)")
        print("  - 如果深度值是微米: depth_scale = 0.000001 (微米转米)")
        print("  - 如果深度值已经是米: depth_scale = 1.0")
    
    print("\n如果深度值范围异常，请检查：")
    print("1. 深度图是否为16位格式")

    print("2. depth_scale 参数是否正确")
    print("3. 深度单位是否为毫米")

# 主程序
if __name__ == "__main__":
    # 直接处理您指定的图片文件
    depth_image_path = "/home/shen/sjw/20250820/0001_depth.png"
    rgb_image_path = "/home/shen/sjw/20250820/0001_rgb.jpg" 
    output_filename = "realsense_pointcloud_820.ply"

    
    print("开始处理图片文件...")
    print(f"深度图: {depth_image_path}")
    print(f"RGB图: {rgb_image_path}")
    
    # 读取图片并创建点云
    pcd = read_images_and_create_pointcloud(depth_image_path, rgb_image_path, output_filename)
    
    if pcd is not None:
        print("点云创建成功！")
        print("深度图已显示，请检查深度数据是否正确")
        print("如果深度数据正常，将显示点云...")
        
        # 等待用户确认深度图是否正确
        input("按回车键继续显示点云...")
        
        print("正在打开点云可视化窗口...")
        o3d.visualization.draw_geometries([pcd])

    else:
        print("点云创建失败！")
    
    print("\n其他可用的函数：")
    print("1. capture_and_save_point_cloud() - 实时捕获并保存点云")
    print("2. process_existing_data(depth_image, color_image, filename) - 处理已有的numpy数组数据")
    print("3. read_images_and_create_pointcloud(depth_path, rgb_path, filename) - 读取图片文件并创建点云")


