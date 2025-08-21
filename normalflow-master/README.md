**normalflow(https://joehjhuang.github.io/normalflow/ ICRA 2025)**  

**主要包括一下内容：**  

1.rgbtopoint.py是realsense采集的图像转为点云的代码  

  #在原来环境基础上需要安装pip install open3d pyrealsense2 matplotlib  
  
2.test_tracking_init.py为normalflow官方代码  

3.keyframes.py是基于gelsight采集的视频流的关键帧算法，进行点云拼接  

4.icp.py是采用icp进行点云配准的相关代码  

5.jixiebi_pingyi.py是依赖机械臂采集的位姿进行平移，不涉及欧拉角和z轴，需要注意的是，代码里的x,y是和机械臂的x,y是相反的  

6.pinjie.py是实现ring闭环的相关代码，主要通过旋转先验+平移补偿，结合径向网格重建实现的  

7.pinjie SDF.py在pinjie.py的基础上实现Alpha Shape SDF的，但针对镂空物体效果不佳，且纹理细节丢失，若只要几何形状，很适合用来拟合  

8.test_tracking.py包含了上述代码的基础上，还有基于normalflow进行点云拼接，高度图直接转点云以及真实机械臂位姿进行点云拼接相关代码  

9.其余安装环境参考本代码的文件的README.md，2-8的代码是在examples文件下  

