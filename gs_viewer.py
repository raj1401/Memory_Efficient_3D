"""
This code uses nerfview to visualize 3DGS and saves the camera parameters.
"""
import os
from typing import Tuple
import time

import numpy as np
import torch
import pickle

import viser
import nerfview

# from gsplat.rendering import _rasterization as rasterization
from gsplat import rasterization
from point_cloud_processing import get_matrix_of_gaussians
from localize_gaussians import get_localized_gaussians


# Set the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scene_name = "SCENE_NAME"

##### GLOBAL VARIABLES #####
VIEWMATS = None
KS = None
WIDTH = 600
HEIGHT = 1200
#############################

gaussians_path = "PATH_TO_GS_POINT_CLOUD_DIR"
gaussians = get_matrix_of_gaussians(os.path.join(gaussians_path, scene_name + ".ply"))
gaussians = torch.tensor(gaussians, device=DEVICE, dtype=torch.float32)

# Crop Box for efficient rendering
gaussians = get_localized_gaussians(gaussians, torch.tensor([0.0, 0.0, 0.0], device=DEVICE), 2.0, "box")
print(f"Number of gaussians after localization: {gaussians.shape[0]}")

# Get the camera poses and target images
colmap_dir = "PATH_TO_COLMAP_DIR"


# Save Path
save_path = "PATH_TO_SAVE_CAMERA_INFO"


@torch.no_grad()
def render_fn(
    camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
) -> np.ndarray:
    c2w = camera_state.c2w
    viewmat = torch.tensor(c2w, device=DEVICE, dtype=torch.float32).inverse().unsqueeze(0)

    K = camera_state.get_K(img_wh)
    K = torch.tensor(K, device=DEVICE, dtype=torch.float32).unsqueeze(0)

    global VIEWMATS
    global KS

    VIEWMATS = viewmat
    KS = K

    img, _, _ = rasterization(
        means=gaussians[:, :3], colors=gaussians[:, 3:6], opacities=gaussians[:, 6],
        scales=gaussians[:, 7:10], quats=gaussians[:, 10:14], viewmats=viewmat,
        Ks=K, width=img_wh[0], height=img_wh[1]
    )
    img = img[0].cpu().numpy()
    return img


def get_local_Ks(K_matrix, width, height):
    K_matrix[:, 0, 2] = width / 2
    K_matrix[:, 1, 2] = height / 2
    return K_matrix

server = viser.ViserServer(verbose=False)

save_button = server.gui.add_button("Save Camera Matrix")

@save_button.on_click
@torch.no_grad()
def save_camera_info(event):
    local_K = get_local_Ks(KS, WIDTH, HEIGHT)

    rendered_img, _, _ = rasterization(
            means=gaussians[:, :3], colors=gaussians[:, 3:6], opacities=gaussians[:, 6],
            scales=gaussians[:, 7:10], quats=gaussians[:, 10:14], viewmats=VIEWMATS, Ks=local_K, width=WIDTH, height=HEIGHT
        )
    rendered_img = rendered_img.cpu().numpy()
    
    try:
        with open(os.path.join(save_path, f"{scene_name}.pkl"), "rb") as f:
            camera_info = pickle.load(f)
        next_idx = len(camera_info)
        camera_info[next_idx] = {"viewmats": VIEWMATS.cpu().numpy(), "Ks": local_K.cpu().numpy(), "width": WIDTH, "height": HEIGHT, "rendered_img": rendered_img}
    except FileNotFoundError:
        camera_info = {}
        camera_info[0] = {"viewmats": VIEWMATS.cpu().numpy(), "Ks": local_K.cpu().numpy(), "width": WIDTH, "height": HEIGHT, "rendered_img": rendered_img}
    
    with open(os.path.join(save_path, f"{scene_name}.pkl"), "wb") as f:
            pickle.dump(camera_info, f)


viewer = nerfview.Viewer(server=server, render_fn=render_fn, mode='rendering')

print("Viewer running... Ctrl+C to exit.")
time.sleep(100000)