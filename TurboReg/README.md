####该代码文件为武汉大学在ICCV2025的发表的TurboReg的相关代码（https://github.com/Laka-3DV/TurboReg），主要用于点云配准。####
####主要包含以下内容：####
1.demo_py/o3d_fpfh.py为重构的推理代码，即直接调用TurboReg进行点云配准，只要自己上传自己相关的ply文件，需要安装sklearn环境
2.demo_py/icp_registration.py为最有效的icp配准代码，预处理+FPFH+RANSAC+ICP 精细配准，同时，采用自动估计体素大小，以及自动尝试多种icp，选出配准最有效的方法，icp有先这个
3.demo_py/multi_pointcloud_registration.py为多个ply配准代码，在2的基础上，可以选择依次配准还是同时配准到第一个ply中
####安装环境参考README.md####