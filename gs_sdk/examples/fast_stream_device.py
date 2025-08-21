import os

import cv2
import yaml

from gs_sdk.gs_device import FastCamera

"""
This script demonstrates how to use the FastCamera class from the gs_sdk package.

It loads a configuration file, initializes the FastCamera, and streaming images with low latency.
This script is only for GelSight Mini so far as only GelSight Mini has the frame dropping issue.

Usage:
    python fast_stream_device.py

Press any key to quit the streaming session.
"""

config_dir = os.path.join(os.path.dirname(__file__), "configs")


def fast_stream_device():
    # Load the device configuration
    config_path = os.path.join(config_dir, "gsmini.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        device_name = config["device_name"]
        imgh = config["imgh"]
        imgw = config["imgw"]
        raw_imgh = config["raw_imgh"]
        raw_imgw = config["raw_imgw"]
        framerate = config["framerate"]

    # Create device and stream the device
    device = FastCamera(device_name, imgh, imgw, raw_imgh, raw_imgw, framerate)
    device.connect()
    print("\nPrss any key to quit.\n")

    # === 初始化视频写入器 ===
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 可改为'mp4v'保存mp4
    out = cv2.VideoWriter('sjw819.avi', fourcc, framerate, (imgw, imgh))

    while True:
        image = device.get_image()
        cv2.imshow(device_name, image)
        # === 写入每一帧到视频 ===
        out.write(image)
        key = cv2.waitKey(1)
        
        if key != -1:
            break
    device.release()
    cv2.destroyAllWindows()
    # === 释放视频写入器 ===
    out.release()



if __name__ == "__main__":
    fast_stream_device()
