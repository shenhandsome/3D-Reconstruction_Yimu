import argparse
import os

import cv2
import matplotlib.pyplot as plt
import yaml

from gs_sdk.gs_reconstruct import Reconstructor
from gs_sdk.viz_utils import plot_gradients

"""
This script demonstrates how to use the Reconstructor class from the gs_sdk package.

It loads a configuration file, initialize the Reconstructor, reconstruct surface information from images,
and save them to files in the "data/" directory.

Usage:
    python reconstruct.py --device {cuda, cpu}

Arguments:
    --device: The device to load the neural network model. Options are 'cuda' or 'cpu'.
"""


#########重建主要是调用recon.get_surface_info()函数，传入图像和ppmm参数，返回G, H, C三个结果
# G: Reconstructed gradients(x/y方向的法向信息)
# H: Reconstructed heights（重建的3D形貌）
# C: Predicted contact mask（表示哪些区域是有效接触的）

model_path = os.path.join(os.path.dirname(__file__), "models", "nnmodel.pth")
config_path = os.path.join(os.path.dirname(__file__), "configs", "gsmini.yaml")
data_dir = os.path.join(os.path.dirname(__file__), "data")

def reconstruct():
    # Argument Parser
    parser = argparse.ArgumentParser(description="Reconstruct surface info from data.")
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cpu",
        help="The device to load and run the neural network model.",
    )
    args = parser.parse_args()

    # Load the device configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        ppmm = config["ppmm"]

    # Create reconstructor
    recon = Reconstructor(model_path, device=args.device)
    # bg_image = cv2.imread(os.path.join(data_dir, "background.png"))
    bg_image = cv2.imread(os.path.join(data_dir, "background2.png"))
    recon.load_bg(bg_image)

    # Reconstruct the surface information from data and save them to files

    filenames = ["test.png","sjw.png","pensjw.png","pensjw2.png","pensjw3.png","phone.png"]
    #filenames = ["bead.png", "key.png", "seed.png"]
    for filename in filenames:
        image = cv2.imread(os.path.join(data_dir, filename))
        G, H, C = recon.get_surface_info(image, ppmm)

        # Plot the surface information
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("GelSight Image")
        plot_gradients(fig, axes[0, 1], G[:, :, 0], G[:, :, 1], mask=C, mode="rgb")
        axes[0, 1].set_title("Reconstructed Gradients")
        axes[1, 0].imshow(H, cmap="jet")
        axes[1, 0].set_title("Reconstructed Heights")
        axes[1, 1].imshow(C)
        axes[1, 1].set_title("Predicted Contact Mask")
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
        save_path = os.path.join(data_dir, "reconstructed_" + filename)
        plt.savefig(save_path)
        plt.close()
        print("Save results to %s" % save_path)


if __name__ == "__main__":
    reconstruct()
