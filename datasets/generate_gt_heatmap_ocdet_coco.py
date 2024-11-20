"""
Usage:
    python datasets/generate_gt_heatmap_ocdet_coco.py [options]

Description:
    This script processes COCO dataset images to generate centerness masks, using either single- or multiprocessing depending on the command-line arguments.

Options:
    --train : If specified, the script processes the training split (train2017) of the COCO dataset. If not specified, it defaults to processing the validation split (val2017)
    -m, --multiprocessing : If specified, the script uses multiprocessing


Examples:
    # To generate centerness masks for the validation set using  a single process:
    python generate_gt_heatmap_ocdet_coco.py -m --use_seg

    # To generate centerness masks for the training set using a multiprocessing:
    python generate_gt_heatmap_ocdet_coco.py --train --multiprocessing --use_seg
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from utils.coco import COCO_ID_TO_CHANNEL, COCO_CHANNEL_TO_ID

import numpy as np
from typing import Any, List
from pycocotools.coco import COCO
from PIL import Image
import multiprocessing as mp
import argparse
from functools import partial
from tqdm import tqdm

COCO_FOLDER = "/mnt/ssd2/xin/data/coco/"


def load_image(coco, root: str, id: int) -> Image.Image:
    path = coco.loadImgs(id)[0]["file_name"]
    return Image.open(os.path.join(root, path)).convert("RGB")


def load_target(coco, id: int) -> List[Any]:
    return coco.loadAnns(coco.getAnnIds(id))


def get_centerness(h, w):
    """
    given a bounding box of width w and height h, returns a centerness score, see FCOS paper
    """
    centerness = np.zeros((h, w))
    for i in range(w):
        for j in range(h):
            l = i + 0.5
            r = w - i - 0.5
            t = j + 0.5
            b = h - j - 0.5
            centerness[j][i] = np.sqrt(
                np.minimum(l, r)
                / np.maximum(l, r)
                * np.minimum(t, b)
                / np.maximum(t, b)
            )
    # normalize to [0, 1]
    if centerness.max() - centerness.min() != 0:
        centerness = (centerness - centerness.min()) / (
            centerness.max() - centerness.min()
        )
    return centerness


def get_generalized_centerness(h, w, alpha=0.5, beta=0.5):
    """
    given a bounding box of width w and height h, returns a weighted centerness score
    default values 0.5 for both alpha and beta gives the same result as centerness
    """
    centerness = np.zeros((h, w))

    # naive implementation
    # for i in range(w):
    #     for j in range(h):
    #         l = i + 0.5
    #         r = w - i - 0.5
    #         t = j + 0.5
    #         b = h - j - 0.5
    #         centerness[j][i] = (np.minimum(l, r) / np.maximum(l, r)) ** alpha * (
    #             np.minimum(t, b) / np.maximum(t, b)
    #         ) ** beta

    # Vectorized implementation
    # Create grids for i and j using meshgrid
    i_grid, j_grid = np.meshgrid(np.arange(w), np.arange(h))
    # Calculate l, r, t, b arrays for the entire grid at once
    l = i_grid + 0.5
    r = w - i_grid - 0.5
    t = j_grid + 0.5
    b = h - j_grid - 0.5
    # Calculate centerness using element-wise operations
    centerness = (np.minimum(l, r) / np.maximum(l, r)) ** alpha * (
        np.minimum(t, b) / np.maximum(t, b)
    ) ** beta

    # normalize to [0, 1]
    if centerness.max() - centerness.min() != 0:
        centerness = (centerness - centerness.min()) / (
            centerness.max() - centerness.min()
        )
    return centerness


def get_ellipse(h, w):
    """
    given a bounding box of width w and height h, returns an ellipse-shaped mask in [0,1]
    """
    ellipse = np.zeros((h, w))
    for i in range(w):
        for j in range(h):
            ellipse[j][i] = 1 - np.sqrt(
                4 * (i - w / 2) ** 2 / w**2 + 4 * (j - h / 2) ** 2 / h**2
            )
    ellipse[ellipse < 0] = 0
    # normalize to [0, 1]
    if ellipse.max() - ellipse.min() != 0:
        ellipse = (ellipse - ellipse.min()) / (ellipse.max() - ellipse.min())
    return ellipse


def generate_map_one_image(image, target, mask_obj_func, use_seg, coco):
    """
    Generate a mask for a single image based on the given target objects.
    Args:
        image (PIL.Image.Image): The input image.
        target (list): COCO's standard list of all annotations for the image.
        mask_obj_func (function): A function that generates a mask for an object given its h and w.
        use_seg (bool): If True, the function will use the segmentation mask to crop the centerness score.
    Returns:
        numpy.ndarray: The generated mask for the image.
    """
    W, H = image.size
    mask_img = np.zeros((80, H, W))  # 80 channels for coco dataset

    for obj in target:
        if obj["iscrowd"]:
            continue

        category_id = obj["category_id"]
        ch = COCO_ID_TO_CHANNEL[category_id]

        bbox = obj["bbox"]
        bx, by, bw, bh = map(round, bbox)

        # continue if the bounding box is too small
        if bw < 1 or bh < 1:
            continue

        # because of rounding, it's possible that the bounding box is outside of the image, so we need to clip it
        if bx + bw > W:
            bw = W - bx
        if by + bh > H:
            bh = H - by

        mask_obj = mask_obj_func(bh, bw)

        y_slice = slice(by, by + bh)
        x_slice = slice(bx, bx + bw)
        # get segmentation mask if exists
        if use_seg and "segmentation" in obj:
            mask_seg = coco.annToMask(obj)
            # cut out the bounding box part
            mask_seg = mask_seg[y_slice, x_slice]
            assert mask_obj.shape == mask_seg.shape
            # only keep centerness score that is on the segmentation mask
            mask_obj *= mask_seg
        # apply centerness score to the ground truth mask
        mask_img[ch, y_slice, x_slice] = np.maximum(
            mask_img[ch, y_slice, x_slice],
            mask_obj,
        )

    return mask_img


def generate_map_one_image_resize(
    image, target, mask_obj_func, use_seg, coco, resized_shape
):
    """
    Generate a mask for a single image based on the given target objects.
    Args:
        image (PIL.Image.Image): The input image.
        target (list): COCO's standard list of all annotations for the image.
        mask_obj_func (function): A function that generates a mask for an object given its h and w.
        use_seg (bool): If True, the function will use the segmentation mask to crop the centerness score.
        resized_shape (tuple): The shape to resize the mask to (width, height).
    Returns:
        numpy.ndarray: The generated mask for the image.
    """
    W, H = image.size
    W_RESIZED, H_RESIZED = resized_shape
    mask_img = np.zeros((80, H_RESIZED, W_RESIZED))  # 80 channels for coco dataset

    for obj in target:
        if obj["iscrowd"]:
            continue
        category_id = obj["category_id"]
        ch = COCO_ID_TO_CHANNEL[category_id]
        x, y, w, h = obj["bbox"]
        bx, by, bw, bh = map(round, obj["bbox"])

        # Map the bounding box coordinates to the resized space
        bx_resized = round(x * W_RESIZED / W)
        by_resized = round(y * H_RESIZED / H)
        bw_resized = round(w * W_RESIZED / W)
        bh_resized = round(h * H_RESIZED / H)

        # continue if the bounding box is too small
        if bw < 1 or bh < 1:
            continue

        # because of rounding, it's possible that the bounding box is outside of the image, so we need to clip it
        if bx_resized + bw_resized > W_RESIZED:
            bw_resized = W_RESIZED - bx_resized
        if by_resized + bh_resized > H_RESIZED:
            bh_resized = H_RESIZED - by_resized

        if bw_resized < 1 or bh_resized < 1:
            # if resizing the bounding box results in a width or height of 0, set center to 1
            # clip to [0, H_RESIZED - 1] and [0, W_RESIZED - 1]
            by_resized = min(max(by_resized, 0), H_RESIZED - 1)
            bx_resized = min(max(bx_resized, 0), W_RESIZED - 1)
            mask_img[ch, by_resized, bx_resized] = 1
            continue

        mask_obj = mask_obj_func(bh_resized, bw_resized)

        y_slice = slice(by_resized, by_resized + bh_resized)
        x_slice = slice(bx_resized, bx_resized + bw_resized)
        # get segmentation mask if exists
        if use_seg and "segmentation" in obj:
            mask_seg = coco.annToMask(obj)
            # cut out the bounding box part
            mask_seg = mask_seg[by : by + bh, bx : bx + bw]
            mask_seg_resized = np.array(
                Image.fromarray(mask_seg).resize(
                    (bw_resized, bh_resized), Image.NEAREST
                )
            )
            assert mask_obj.shape == mask_seg_resized.shape
            # only keep centerness score that is on the segmentation mask
            mask_obj *= mask_seg_resized
        # apply centerness score to the ground truth mask
        mask_img[ch, y_slice, x_slice] = np.maximum(
            mask_img[ch, y_slice, x_slice],
            mask_obj,
        )

    return mask_img


def generate_maps_coco(
    mask_folder_name,
    mask_type,
    coco_folder=COCO_FOLDER,
    split="val2017",
    multiprocessing=False,
    num_workers=128,
    use_seg=False,
    resized_shape=None,
):
    # get function to generate mask
    if mask_type == "centerness":
        # mask_obj_func = get_generalized_centerness
        mask_obj_func = get_centerness
    elif mask_type == "ellipse":
        mask_obj_func = get_ellipse
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
                        mask_obj_func=mask_obj_func,
                        use_seg=use_seg,
                        resized_shape=resized_shape,
                    ),
                    ids,
                ):
                    pbar.update()
    else:  # single process
        for id in tqdm(ids, desc="Processing images"):
            process_image(
                id, coco, root, mask_folder, mask_obj_func, use_seg, resized_shape
            )


# Function to process a single image (to be used by each worker process)
def process_image(
    id, coco, root, mask_folder, mask_obj_func, use_seg=False, resized_shape=None
):
    file_name = coco.loadImgs(id)[0]["file_name"]

    image = Image.open(os.path.join(root, file_name)).convert("RGB")
    target = coco.loadAnns(coco.getAnnIds(id))

    # Create the mask for one image
    if resized_shape:
        mask_img = generate_map_one_image_resize(
            image, target, mask_obj_func, use_seg, coco, resized_shape
        )
    else:
        mask_img = generate_map_one_image(image, target, mask_obj_func, use_seg, coco)

    num_channels, height, width = mask_img.shape
    for current_channel in range(num_channels):
        channel_data = mask_img[current_channel]
        category_id = COCO_CHANNEL_TO_ID[current_channel]
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
        "--mask_type", default="centerness", choices=["centerness", "ellipse"]
    )
    parser.add_argument("-m", "--multiprocessing", action="store_true")
    parser.add_argument("--num_workers", default=128, type=int)
    parser.add_argument("--use_seg", action="store_true")
    parser.add_argument(
        "--resized_shape",
        "-r",
        default=None,
        type=int,
        help="Resized shape, currently only support one integer for square shape",
    )
    args = parser.parse_args()

    split = "train2017" if args.train else "val2017"
    mask_folder_name = f"{args.mask_type[0]}maps"
    if args.use_seg:
        mask_folder_name += "_seg"
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
        args.use_seg,
        resized_shape if args.resized_shape else None,
    )
