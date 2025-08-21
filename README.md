# 3D-Reconstruction_Yimu
## gs_sdk  

(1)进行采集   

    python examples/fast_stream_device.py  
    
    python examples/stream_device.py  
    
(2)其中fast_stream_device.py加入了视频写入器，可以采取视频流数据，图像直接通过可视化界面进行采集  

## normalflow(https://joehjhuang.github.io/normalflow/ ICRA 2025)  

(1)rgbtopoint.py是realsense采集的图像转为点云的代码  

  #在原来环境基础上需要安装pip install open3d pyrealsense2 matplotlib  
  
(2)test_tracking_init.py为normalflow官方代码  

(3)keyframes.py是基于gelsight采集的视频流的关键帧算法，进行点云拼接  

(4)icp.py是采用icp进行点云配准的相关代码  

(5)jixiebi_pingyi.py是依赖机械臂采集的位姿进行平移，不涉及欧拉角和z轴，需要注意的是，代码里的x,y是和机械臂的x,y是相反的  

(6)pinjie.py是实现ring闭环的相关代码，主要通过旋转先验+平移补偿，结合径向网格重建实现的  

(7)pinjie SDF.py在pinjie.py的基础上实现Alpha Shape SDF的，但针对镂空物体效果不佳，且纹理细节丢失，若只要几何形状，很适合用来拟合  

(8)test_tracking.py包含了上述代码的基础上，还有基于normalflow进行点云拼接，高度图直接转点云以及真实机械臂位姿进行点云拼接相关代码  

(9)其余安装环境参考本代码的文件的readme.md,2-8的代码是在examples文件下  

## TurboReg（https://github.com/Laka-3DV/TurboReg ICCV 2025）  

####该代码文件主要用于点云配准。####  

(1)demo_py/o3d_fpfh.py为重构的推理代码，即直接调用TurboReg进行点云配准，只要自己上传自己相关的ply文件，需要安装sklearn环境  

(2)demo_py/icp_registration.py为最有效的icp配准代码，**预处理+FPFH+RANSAC+ICP 精细配准，同时，采用自动估计体素大小，以及自动尝试多种icp，选出配准最有效的方法**，**icp优先用这个**

(3)demo_py/multi_pointcloud_registration.py为多个ply配准代码，在2的基础上，可以选择依次配准还是同时配准到第一个ply中  

(4)安装环境参考reademe.md  

## MiniGPT-3D(https://tangyuan96.github.io/minigpt_3d_project_page/ ACM MM 2024)  

详情见该项目README.md  

