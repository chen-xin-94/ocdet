"""
Usage:
    python datasets/generate_gt_heatmap_kp_coco.py [options]

Description:
    This script processes COCO dataset images to generate heatmaps based on keypoint framework, i.e fixed gaussian centered on keypoints, using either single- or multiprocessing depending on the command-line arguments.

Options:
    --train : If specified, the script processes the training split (train2017) of the COCO dataset. If not specified, it defaults to processing the validation split (val2017)
    -m, --multiprocessing : If specified, the script uses multiprocessing


Examples:
    # To generate centerness masks for the validation set using  a single process:
    python generate_gt_heatmap_kp_coco.py -m

    # To generate centerness masks for the training set using a multiprocessing:
    python generate_gt_heatmap_kp_coco.py --train --multiprocessing
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from utils.coco import COCO_ID_TO_CHANNEL, COCO_CHANNEL_TO_ID

from collections import defaultdict
import numpy as np
from typing import Any, List
from pycocotools.coco import COCO
from PIL import Image
import multiprocessing as mp
import argparse
from functools import partial
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import socket

COCO_FOLDER = "/mnt/ssd2/xxx/data/coco/"


def load_image(coco, root: str, id: int) -> Image.Image:
    path = coco.loadImgs(id)[0]["file_name"]
    return Image.open(os.path.join(root, path)).convert("RGB")


def load_target(coco, id: int) -> List[Any]:
    return coco.loadAnns(coco.getAnnIds(id))


def generate_gaussian_heatmap(image_size, heatmap_size, coords, sigma=2):
    """
    Generate a heatmap by applying 2D Gaussian at the given coordinates.

    Args:
    image_size (tuple): Size of the input image (height, width).
    heatmap_size (tuple): Size of the heatmap (height, width).
    coords (list of tuples): List of (x, y) coordinates where the Gaussians should be centered.
    sigma (float): Standard deviation of the Gaussian.

    Returns:
    numpy.ndarray: Heatmap with the 2D Gaussian applied at the given coordinates.
    """
    stride = image_size / heatmap_size

    target = np.zeros(shape=heatmap_size, dtype=np.float32)

    # # Generate gaussian, to be cropped later
    tmp_size = sigma * 3
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))

    for coord in coords:
        mu_y = int(coord[0] / stride[0] + 0.5)
        mu_x = int(coord[1] / stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if (
            ul[0] >= heatmap_size[1]
            or ul[1] >= heatmap_size[0]
            or br[0] < 0
            or br[1] < 0
        ):
            # If not, just move on to the next coord
            continue

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_size[1]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[0]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size[1])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[0])

        # assgin only if bigger
        target[img_y[0] : img_y[1], img_x[0] : img_x[1]] = np.where(
            g[g_y[0] : g_y[1], g_x[0] : g_x[1]]
            > target[img_y[0] : img_y[1], img_x[0] : img_x[1]],
            g[g_y[0] : g_y[1], g_x[0] : g_x[1]],
            target[img_y[0] : img_y[1], img_x[0] : img_x[1]],
        )

    return target


def generate_map_one_image(image, target, mask_func, resized_shape=None):
    """
    Generate masks for a single image based on the given target objects.
    Args:
        image (PIL.Image.Image): The input image.
        target (list): COCO's standard list of all annotations for the image.
        mask_func (function): A function that generates a mask given the image size and a list of coordinates.
        resized_shape (tuple): The shape to resize the mask to (width, height).
    Returns:
        numpy.ndarray: The generated mask for the image.
    """
    W, H = image.size
    if resized_shape:
        mask_img = np.zeros((80, resized_shape[1], resized_shape[0]))
    else:
        mask_img = np.zeros((80, H, W))  # 80 channels for coco dataset
    # get all centers
    coords = defaultdict(list)
    for obj in target:

        if obj["iscrowd"]:
            continue
        category_id = obj["category_id"]
        ch = COCO_ID_TO_CHANNEL[category_id]

        x, y, w, h = obj["bbox"]

        # # continue if the bounding box is too small
        # if w < 1 or h < 1:
        #     continue

        bxc = round(x + w / 2)
        byc = round(y + h / 2)

        # because of rounding, it's possible that the bounding box is outside of the image, so we need to clip it
        bxc = min(max(0, bxc), W - 1)
        byc = min(max(0, byc), H - 1)

        coords[ch].append((byc, bxc))

    for ch, coord in coords.items():
        mask_img[ch] = mask_func(
            image_size=np.array((H, W)),
            heatmap_size=(
                np.array((resized_shape[1], resized_shape[0]))
                if resized_shape
                else np.array((H, W))
            ),
            coords=coord,
        )

    return mask_img


def generate_maps_coco(
    mask_folder_name,
    mask_type,
    coco_folder=COCO_FOLDER,
    split="val2017",
    multiprocessing=False,
    num_workers=128,
    resized_shape=None,
    ids=None,
):
    # TODO: add support for other mask function
    # get function to generate mask
    if mask_type == "gaussian":
        mask_func = generate_gaussian_heatmap
    else:
        raise ValueError(f"mask_type {mask_type} not supported")

    # Set paths
    root = os.path.join(coco_folder, split)
    annFile = os.path.join(coco_folder, "annotations", f"instances_{split}.json")
    mask_folder = os.path.join(
        coco_folder, mask_folder_name, "PLACEHOLDER", split
    )  # PLACEHOLDER for category_id
    os.makedirs(mask_folder, exist_ok=True)

    # Load COCO dataset
    coco = COCO(annFile)
    if ids is None:
        ids = list(sorted(coco.imgs.keys()))

    if multiprocessing:
        with mp.Pool(processes=num_workers) as pool:
            with tqdm(total=len(ids), desc="Processing images") as pbar:
                for _ in pool.imap_unordered(
                    partial(
                        process_image,
                        coco=coco,
                        root=root,
                        mask_folder=mask_folder,
                        mask_func=mask_func,
                        resized_shape=resized_shape,
                    ),
                    ids,
                ):
                    pbar.update()
    else:  # single process
        for id in tqdm(ids, desc="Processing images"):
            process_image(id, coco, root, mask_folder, mask_func, resized_shape)


# Function to process a single image (to be used by each worker process)
def process_image(id, coco, root, mask_folder, mask_func, resized_shape=None):
    file_name = coco.loadImgs(id)[0]["file_name"]

    image = Image.open(os.path.join(root, file_name)).convert("RGB")
    target = coco.loadAnns(coco.getAnnIds(id))

    # Create the mask for one image
    mask_img = generate_map_one_image(image, target, mask_func, resized_shape)

    num_channels, height, width = mask_img.shape
    for ch in range(num_channels):
        channel_data = mask_img[ch]
        category_id = COCO_CHANNEL_TO_ID[ch]
        cur_mask_folder = mask_folder.replace("PLACEHOLDER", str(category_id))
        if np.any(channel_data):
            mask = Image.fromarray((channel_data * 255).astype(np.uint8))
            mask_path = os.path.join(cur_mask_folder, f"{file_name.split('.')[0]}.png")
            # resume
            if os.path.exists(mask_path):
                # print(f"Mask already exists at {mask_path}")
                continue
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            mask.save(mask_path)
            # print(f"Saved mask to {mask_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument(
        "--mask_type",
        default="gaussian",
        choices=[
            "guassian",
        ],
    )
    parser.add_argument(
        "--resized_shape",
        "-r",
        default=None,
        type=int,
        help="Resized shape, currently only support one integer for square shape",
    )
    parser.add_argument("-m", "--multiprocessing", action="store_true")
    parser.add_argument("--num_workers", default=128, type=int)
    args = parser.parse_args()

    split = "train2017" if args.train else "val2017"
    mask_folder_name = f"{args.mask_type[0]}maps"
    mask_folder_name += "_coord"  # to differentiate from mask generated from bbox
    if args.resized_shape:
        mask_folder_name += f"_r{args.resized_shape}"
        resized_shape = (args.resized_shape, args.resized_shape)

    generate_maps_coco(
        mask_folder_name,
        args.mask_type,
        COCO_FOLDER,
        split,
        args.multiprocessing,
        args.num_workers,
        resized_shape=resized_shape if args.resized_shape else None,
    )

    # # ids with unplotted gaussian blobs because of too small bounding boxes:
    # ids = [390267, 141426, 168890, 545566, 111930, 376491, 81768]

    # generate_maps_coco(
    #     mask_folder_name,
    #     args.mask_type,
    #     COCO_FOLDER,
    #     split,
    #     args.multiprocessing,
    #     args.num_workers,
    #     resized_shape if args.resized_shape else None,
    #     ids,
    # )
