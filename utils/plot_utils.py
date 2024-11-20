import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from matplotlib import colormaps as cm

import torch
from torchvision import transforms
import torchvision.transforms.functional as F

from ultralytics.utils.plotting import Annotator, Colors
from typing import Dict, Optional, Union
from pathlib import Path

import os
import sys

# Get the directory where the current script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory of the current dir (i.e. root dir of this repo) to the system path
sys.path.append(os.path.dirname(current_dir))
from metrics.cad import get_centroid_coordinates_from_map
from utils.coco import COCO_ID_TO_NAME, COCO_ID_TO_CHANNEL

# for plotting
inv_normalize = transforms.Compose(
    [
        transforms.Normalize(
            mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
    ]
)


class ColorsOCDet:

    def __init__(self):
        hexs = (
            "FF0000",  # Bright Red
            "FF9900",  # Neon Orange
            "FFFF00",  # Bright Yellow
            "00FF00",  # Neon Green
            "00FFFF",  # Cyan
            "0000FF",  # Bright Blue
            "FF00FF",  # Magenta
            "8B00FF",  # Purple
            "FF1493",  # Hot Pink
            "00FF7F",  # Spring Green
            "FF4500",  # Orange-Red
            "00CED1",  # Dark Turquoise
            "FF6347",  # Tomato Red
            "FFD700",  # Gold
            "7FFF00",  # Chartreuse Green
            "7CFC00",  # Lawn Green
            "00BFFF",  # Deep Sky Blue
            "FF69B4",  # Pink
            "FF7F50",  # Coral
            "40E0D0",  # Turquoise
            "FFB6C1",  # Light Pink
            "FF00FF",  # Fuchsia
            "DC143C",  # Crimson
            "1E90FF",  # Dodger Blue
            "ADFF2F",  # Green Yellow
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


COLORS = Colors()  # default to ultralytics colors
# COLORS = ColorsOCDet() # custom colors for OCDet


def show3(
    image: torch.Tensor,
    target: torch.Tensor,
    pred: torch.Tensor,
    image_id=None,
    titles=["Image", "Target", "Prediction"],
):
    target = overlay_half(image, target, color=(255, 0, 0))
    pred = overlay_half(image, pred, color=(0, 255, 0))
    image = F.to_pil_image(image)

    fig, axs = plt.subplots(1, 3, figsize=(28, 7))
    if image_id:
        fig.suptitle(f"Image ID: {image_id}")

    for i, (ax, img, title) in enumerate(zip(axs, [image, target, pred], titles)):
        w, h = img.size
        ax.imshow(img)
        ax.set_title(title + f" {w}x{h}")
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    return fig


def show4(
    image: torch.Tensor,
    target: torch.Tensor,
    proba: torch.Tensor,
    image_id=None,
    colormap="jet",
    titles=["Image", "Target", "Prediction Map", "Prediction Label"],
):
    pred_map = overlay_colormap_tensor(image, proba, colormap=colormap)
    pred_label = overlay_map(image, proba, color=(0, 255, 0))

    fig, axs = plt.subplots(1, 4, figsize=(28, 7))
    if image_id:
        fig.suptitle(f"Image ID: {image_id}")

    for i, (ax, img, title) in enumerate(
        zip(axs, [image, target, pred_map, pred_label], titles)
    ):
        if isinstance(img, torch.Tensor):
            img = F.to_pil_image(img)
        w, h = img.size
        ax.imshow(img)
        ax.set_title(title + f" {w}x{h}")
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    return fig


def show8(
    image: torch.Tensor,
    target: torch.Tensor,
    proba: torch.Tensor,
    image_id=None,
    colormap="jet",
    titles=[
        "GT mask on image",
        "Pred mask on image",
        "GT peaks on image",
        "Pred peaks on image",
        "GT mask",
        "Pred mask",
        "GT peaks on GT mask",
        "Pred peaks on pred mask",
    ],
):
    gt_peak_coords = get_centroid_coordinates_from_map(target).flip(1)
    pred_peak_coords = get_centroid_coordinates_from_map(proba).flip(1)
    gt_mask_on_image = overlay_colormap_tensor(image, target, colormap=colormap)
    pred_mask_on_image = overlay_colormap_tensor(image, proba, colormap=colormap)
    gt_peaks_on_image = plot_points_on_image(image, gt_peak_coords)
    pred_peaks_on_image = plot_points_on_image(image, pred_peak_coords)

    fig, axs = plt.subplots(2, 4, figsize=(28, 14))
    if image_id:
        fig.suptitle(f"Image ID: {image_id}")

    for i, (ax, img, title) in enumerate(
        zip(
            axs[0],
            [
                gt_mask_on_image,
                pred_mask_on_image,
                gt_peaks_on_image,
                pred_peaks_on_image,
            ],
            titles[:4],
        )
    ):
        if isinstance(img, torch.Tensor):
            img = F.to_pil_image(img)

        # w, h = img.size
        # ax.set_title(title + f" {w}x{h}")

        ax.imshow(img)
        ax.set_title(title)

        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    for i, (ax, img, title) in enumerate(
        zip(
            axs[1],
            [
                target,
                proba,
                target,
                proba,
            ],
            titles[4:],
        )
    ):
        if isinstance(img, torch.Tensor):
            img = F.to_pil_image(img)

        # w, h = img.size
        # ax.set_title(title + f" {w}x{h}")

        ax.imshow(img, cmap="gray")

        if i == 2:  # img==target
            # draw gt_peak_coords on the mask
            ax.plot(gt_peak_coords[:, 0], gt_peak_coords[:, 1], "ro", markersize=5)
        if i == 3:  # img==proba
            # draw pred_peak_coords on the mask
            ax.plot(pred_peak_coords[:, 0], pred_peak_coords[:, 1], "ro", markersize=5)

        ax.set_title(title)

        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    return fig


def draw_points_on_heatmap_img(
    img, centers, category_ids, radius=15, colors=COLORS, alpha=None, cmap="viridis"
):

    colormap = plt.get_cmap(cmap)  # which maps a 2D heatmap to a 3D RGB image

    # Check if the img is 2D or 3D
    if img.ndim == 2:  # for heatmap
        # If 2D, apply the viridis colormap
        heatmap_normalized = (img - np.min(img)) / (
            np.max(img) - np.min(img)
        )  # Normalize
        img_rgb = (colormap(heatmap_normalized)[:, :, :3] * 255).astype(
            np.uint8
        )  # Apply colormap and convert to RGB
    else:  # for normal img
        img_rgb = img
        # If 3D, assume it's already an RGB overlay (no need for colormap)
        if alpha:
            zeros_2d = np.zeros((img.shape[0], img.shape[1]))
            zero_colormap = (colormap(zeros_2d)[:, :, :3] * 255).astype(np.uint8)
            img_rgb = cv2.addWeighted(img_rgb, alpha, zero_colormap, 1 - alpha, 0)

    # Convert the heatmap to a PIL Image (if it's not already one)
    heatmap_pil = Image.fromarray(img_rgb)

    # Create a drawing context to draw the point
    draw = ImageDraw.Draw(heatmap_pil)

    for (cx, cy), c in zip(centers, category_ids):
        color = colors(c)
        # Draw a circle at (cx, cy) on the heatmap
        draw.ellipse(
            [(cx - radius, cy - radius), (cx + radius, cy + radius)],
            fill=color,
            outline=color,
        )

    return np.array(heatmap_pil)


def plot_points_on_image(
    image,
    points,
    color=(255, 0, 0),
    palette=COLORS,
    radius=None,
    radius_factor=40,
    color_ids=None,
    show=False,
    ret=True,
):
    """
    Plot peaks on image using numpy array
    Args:
        image: image that the peaks should be plotted on, it can be
                1. a tensor, shape [C, H, W]
                2. a numpy array, shape [H, W, C]
        points: peak coordinates, numpy array, shape [N, 2], where the second dim should be (x,y)
        color: color of the peaks
        palette: color palette for different categories, only used when category_ids is not None
        radius: radius of the circle to be drawn around the peak, if None, it will be set to max(H, W) // 64
        radius_factor: factor to determine the radius of the circle based on the image size, only used when radius is None
        category_ids: list of color ids, shape [N], if None, all peaks will be drawn with the same color
        show: whether to display the image
        ret: whether to return the image
    """

    # Convert the image to numpy array if necessary
    if isinstance(image, torch.Tensor):
        image = F.to_pil_image(image)
        image = np.array(image)
    elif isinstance(image, np.ndarray):
        # if not uint8, convert to uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
    elif isinstance(image, Image.Image):
        image = np.array(image)
    else:
        raise ValueError(
            "image should be either a torch.Tensor, np.ndarray or PIL.Image"
        )
    h, w = image.shape[:2]
    if radius is None:
        radius = max(h, w) // radius_factor
    # Draw circles on the image at the peak locations
    if color_ids is not None:  # use different colors for different categories
        for peak, c in zip(points, color_ids):
            peak = tuple(map(int, peak))
            image = cv2.circle(image, peak, radius, palette(c), -1)
    else:  # use specified color for all peaks
        for peak in points:
            peak = tuple(map(int, peak))  # Convert peak coordinates to integers
            image = cv2.circle(image, peak, radius, color, -1)

    # Display the image if requested
    if show:
        plt.imshow(image)
        plt.show()

    # Return the image if requested
    if ret:
        return image


def visualize_batch_single_cls(
    task, n, imgs, targets, y_probas, image_ids, normalized=True
):
    # sample n images
    # TODO: sample correct and incorrect predictionss
    inds = np.random.choice(range(len(imgs)), n, replace=False)
    figs = []
    for i in inds:
        if normalized:
            image = inv_normalize(imgs[i]).clone()
        else:
            image = imgs[i].clone()
        target = targets[i].squeeze().clone()
        y_proba = y_probas[i].clone()
        image_id = image_ids[i]

        if task == "half":
            fig = show3(image, target, y_proba, image_id)
        elif (
            "cmap" in task or task == "gmap_kp"
        ):  # task == "cmap" or task == "cmap_seg"
            fig = show8(image, target, y_proba, image_id)
        elif "map" in task:  #  task == "map" or task == "gmap"
            fig = show4(image, target, y_proba, image_id)
        else:
            raise ValueError("task not recognized")
        figs.append(fig)
    return figs


def visualize_batch_multi_cls(
    task, n, imgs, targets, y_probas, image_ids, category_ids, normalized=True
):
    """
    Inputs:
    - task: str, one of ["cmap", "cmap_seg", "gmap_kp"]
    - n: int, number of images to visualize
    - imgs: torch.Tensor, shape [N, 3, H, W]
    - targets: torch.Tensor, shape [N, 80, H, W]
    - y_probas: torch.Tensor, shape [N, 80, H, W]
    - image_ids: list of int, length N
    - category_ids: list of 1d torch.Tensor,
                    length of each tensor depends on the number of gt objects in each image
    """

    # sample n images
    # TODO: sample correct and incorrect predictionss
    batch_inds = np.random.choice(range(len(imgs)), n, replace=False)
    figs = []
    for batch_i in batch_inds:
        if normalized:
            image = inv_normalize(imgs[batch_i])
        else:
            image = imgs[batch_i]
        target_all = targets[batch_i]
        y_proba_all = y_probas[batch_i]
        image_id = image_ids[batch_i]
        category_id_all = category_ids[batch_i]

        if ("cmap" in task) or ("gmap_kp" in task):
            fig_all = show8_multi_cls(
                image,
                target_all,
                y_proba_all,
                category_id_all,
                image_id,
            )
        else:
            raise ValueError("task not recognized")
        figs.extend(fig_all)
    return figs


def show8_multi_cls(
    image,
    target_all,
    y_proba_all,
    category_id_all,
    image_id,
    colormap="jet",
    titles=[
        "GT mask on image",
        "Pred mask on image",
        "GT peaks on image",
        "Pred peaks on image",
        "GT mask",
        "Pred mask",
        "GT peaks on GT mask",
        "Pred peaks on pred mask",
    ],
):
    fig_all = []

    for category_id in set(category_id_all.tolist()):
        ch = COCO_ID_TO_CHANNEL[category_id]
        target = target_all[ch]
        y_proba = y_proba_all[ch]

        gt_peak_coords = get_centroid_coordinates_from_map(target).flip(1)
        pred_peak_coords = get_centroid_coordinates_from_map(y_proba).flip(1)
        gt_mask_on_image = overlay_colormap_tensor(image, target, colormap=colormap)
        pred_mask_on_image = overlay_colormap_tensor(image, y_proba, colormap=colormap)
        gt_peaks_on_image = plot_points_on_image(image, gt_peak_coords)
        pred_peaks_on_image = plot_points_on_image(image, pred_peak_coords)

        fig, axs = plt.subplots(2, 4, figsize=(28, 14))
        if image_id:
            fig.suptitle(
                f"Image ID: {image_id}, GT Category: {COCO_ID_TO_NAME[category_id]}"
            )

        for i, (ax, img, title) in enumerate(
            zip(
                axs[0],
                [
                    gt_mask_on_image,
                    pred_mask_on_image,
                    gt_peaks_on_image,
                    pred_peaks_on_image,
                ],
                titles[:4],
            )
        ):
            if isinstance(img, torch.Tensor):
                img = F.to_pil_image(img)

            # w, h = img.size
            # ax.set_title(title + f" {w}x{h}")

            ax.imshow(img)
            ax.set_title(title)

            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        for i, (ax, img, title) in enumerate(
            zip(
                axs[1],
                [
                    target,
                    y_proba,
                    target,
                    y_proba,
                ],
                titles[4:],
            )
        ):
            if isinstance(img, torch.Tensor):
                img = F.to_pil_image(img)

            # w, h = img.size
            # ax.set_title(title + f" {w}x{h}")

            ax.imshow(img, cmap="gray")

            if i == 2:  # img==target
                # draw gt_peak_coords on the mask
                ax.plot(gt_peak_coords[:, 0], gt_peak_coords[:, 1], "ro", markersize=5)
            if i == 3:  # img==proba
                # draw pred_peak_coords on the mask
                ax.plot(
                    pred_peak_coords[:, 0], pred_peak_coords[:, 1], "ro", markersize=5
                )

            ax.set_title(title)

            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        fig_all.append(fig)
    return fig_all


def apply_colormap_on_tensor(mask: torch.Tensor, colormap: str) -> torch.Tensor:
    """Apply a matplotlib colormap on a 1-channel image tensor."""
    # Ensure mask is in CPU and numpy format for matplotlib
    mask_np = mask.squeeze().cpu().numpy()
    cmap = plt.get_cmap(colormap)
    colored_mask = cmap(mask_np)[:, :, :3]  # Ignore alpha channel
    return (
        torch.from_numpy(colored_mask).permute(2, 0, 1).float()
    )  # Convert back to tensor and reorder dimensions


def overlay_colormap_tensor(
    img: torch.Tensor, mask: torch.Tensor, colormap: str = "jet", alpha: float = 0.5
) -> torch.Tensor:
    """
    Overlay a colormapped mask on a background image tensor.

    Args:
        img: image tensor, shape [C, H, W]
        mask: ground truth mask tensor, shape [H, W]
        colormap: colormap to be applied on the mask
        alpha: transparency of the background image
    """
    # Check input types and values
    if (not isinstance(img, torch.Tensor)) or (not isinstance(mask, torch.Tensor)):
        raise TypeError("img and mask arguments need to be torch.Tensor")
    if not isinstance(alpha, float) or alpha < 0 or alpha > 1:
        raise ValueError(
            "alpha argument is expected to be of type float between 0 and 1"
        )

    # add two dummy batch and channel dim to make the shape [1, 1, H, W], which is expected by interpolate
    mask = mask.unsqueeze(0).unsqueeze(0)

    # Resize mask to match image size
    mask_resized = torch.nn.functional.interpolate(
        mask, size=img.shape[1:], mode="nearest"
    ).squeeze(0)

    # Apply colormap on the mask
    colored_mask = apply_colormap_on_tensor(mask_resized, colormap)

    # Overlay images
    overlayed_img = alpha * img + (1 - alpha) * colored_mask

    # Ensure the overlayed image is clipped to valid range
    overlayed_img = torch.clamp(overlayed_img, 0, 1)

    return overlayed_img


def overlay_colormap(
    img: Union[Image.Image, torch.Tensor, np.ndarray],
    mask: Union[Image.Image, torch.Tensor, np.ndarray],
    colormap: str = "jet",
    alpha: float = 0.5,
    show=False,
    ret=True,
    save_path=None,
) -> Image.Image:
    """Overlay a colormapped mask on a background image

    Modified from https://github.com/frgfm/torch-cam/blob/main/torchcam/utils.py

    Args:
        img: background image, can be a PIL.Image, torch.Tensor (CHW), or np.ndarray (HWC)
        mask: mask to be overlayed in grayscale
        colormap: colormap to be applied on the mask
        alpha: transparency of the background image

    Returns:
        overlayed image

    Raises:
        TypeError: when the arguments have invalid types
        ValueError: when the alpha argument has an incorrect value
    """
    if not isinstance(img, Image.Image):
        if isinstance(img, torch.Tensor):
            img = F.to_pil_image(img)
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img).convert("RGB")
        else:
            raise TypeError(
                "img argument needs to be a torch.Tensor, np.ndarray, or PIL.Image"
            )

    if not isinstance(mask, Image.Image):
        if isinstance(mask, torch.Tensor):
            mask = F.to_pil_image(mask)
        elif isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)
        else:
            raise TypeError(
                "mask argument needs to be a torch.Tensor, np.ndarray, or PIL.Image"
            )

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError(
            "alpha argument is expected to be of type float between 0 and 1"
        )

    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    # overlay = mask.resize(img.size, resample=Image.BICUBIC)
    # overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)

    overlay = mask.resize(img.size, resample=Image.NEAREST)
    overlay = (255 * cmap(np.asarray(overlay))[:, :, :3]).astype(np.uint8)

    # Overlay the image with the mask
    overlayed_img = Image.fromarray(
        (alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8)
    )
    if save_path:
        overlayed_img.save(save_path)
    if show:
        plt.imshow(overlayed_img)
        plt.show()
    if ret:
        return overlayed_img


def overlay_map(
    img: torch.Tensor,
    centroid_map: torch.Tensor,
    color=(255, 0, 0),
    conf=0.5,
    alpha=0.2,
    show=False,
    ret=True,
):
    """
    overlay centroid map on image and color the centroids with a fixed color if the confidence is above conf

    Args:
        img: image tensor, shape [C, H, W]
        centroid_map: centroid map tensor, shape [H, W]
        color: color to be used for the overlay
        conf: confidence threshold for overlaying the color
        alpha: transparency of the overlay
        show: whether to display the image
        ret: whether to return the

    """
    img = img.clone()
    centroid_map = centroid_map.clone()
    _, w, h = img.shape
    # Create a tensor for the color with the same size as the image
    color = torch.tensor(color) / 255  # normalize
    color = color.view(3, 1, 1).repeat(1, w, h).to(img.dtype)  # repeat

    # resize centroid map to image size
    centroid_map = transforms.Resize(
        size=(w, h), interpolation=transforms.InterpolationMode.NEAREST
    )(centroid_map.unsqueeze(0))

    # get the 2d mask
    mask = centroid_map[0] > conf

    # # option 1: replace with the color
    # overlayed_img = img
    # overlayed_img[:, mask] = color[:, mask]

    # option 2: overlay with small alpha (almost replaced by color)
    overlayed_img = img
    overlayed_img[:, mask] = (
        alpha * overlayed_img[:, mask] + (1 - alpha) * color[:, mask]
    )

    # Ensure the overlayed image is clipped to valid range
    overlayed_img = torch.clamp(overlayed_img, 0, 1)
    overlayed_img = F.to_pil_image(overlayed_img)

    if show:
        plt.imshow(overlayed_img)
        plt.show()
    if ret:
        return overlayed_img


def plot_centroid(img, centroids, show=False, ret=True):
    """Plot centroids on image"""

    if isinstance(img, torch.Tensor):
        img = F.to_pil_image(img)
    img = img.copy()
    w, h = img.size
    img = np.array(img)

    if isinstance(centroids, torch.Tensor):
        centroids = centroids.cpu().clone().numpy()
    for centroid in centroids:
        centroid[0] = centroid[0] * w
        centroid[1] = centroid[1] * h
        centroid = tuple(map(int, centroid))  # Convert centroid coordinates to integers
        img = cv2.circle(img, centroid, 5, (255, 0, 0), -1)
    if show:
        plt.imshow(img)
        plt.show()
    if ret:
        return img


def plot_grid(plot_func, dataset, grid=(3, 3)):
    """Plot a grid of images extracted from datasets.dataset with plot_func"""
    n = grid[0] * grid[1]
    fig = plt.figure(figsize=(16, 16))
    for i in range(9):
        n = np.random.randint(0, len(dataset))
        img, centroids = dataset[n]
        img = plot_func(img, centroids, show=False)
        ax = fig.add_subplot(grid[0], grid[1], i + 1)
        ax.set_title("image {}".format(n))
        ax.imshow(img)
        plt.axis("off")
    plt.show()


def overlay_half(
    img: torch.Tensor,
    target: torch.Tensor,
    color=(255, 0, 0),
    alpha=0.5,
    conf=0.5,
    show=False,
    ret=True,
):
    img = img.clone().detach().cpu()
    target = target.clone().detach().cpu().flatten()
    image = F.to_pil_image(img)
    w, h = image.size
    # Create a rectangle image with the same size as the original image
    # but with a semi-transparent red rectangle on the left half
    rectangle = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(rectangle)
    if target[0] > conf:
        draw.rectangle([0, 0, w // 2, h], fill=color + (int(255 * alpha),))
    if target[1] > conf:
        draw.rectangle([w // 2, 0, w, h], fill=color + (int(255 * alpha),))

    # Overlay the rectangle on the original image
    image_with_rectangle = Image.alpha_composite(image.convert("RGBA"), rectangle)

    # Convert back to RGB if necessary and save or display the image
    image_with_rectangle = image_with_rectangle.convert("RGB")
    if show:
        plt.imshow(image_with_rectangle)
        plt.show()
    if ret:
        return image_with_rectangle


def annotate_image(
    img: Union[torch.Tensor, np.ndarray],
    boxes: Union[torch.Tensor, np.ndarray],
    category: Union[torch.Tensor, np.ndarray],
    confs: Optional[Union[torch.Tensor, np.ndarray]] = None,
    category_id_to_name: Dict[int, str] = None,
    colors=COLORS,
    conf_thres: float = 0.5,
    masks: Optional[Union[torch.Tensor, np.ndarray]] = None,
    save_path: Optional[Union[Path, str]] = None,
    line_width: Optional[float] = None,
    font_size: Optional[float] = None,
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Plot image grid with labels, bounding boxes, and masks, using ultralytics Annotator class.

    Args:
        img: one singel image to plot on. Shape: (channels, height, width).
        boxes: Bounding boxes for each detection. Shape: (num_detections, 4). Format: (x1, y1, x2, y2).
        category: Class id for each detection. Shape: (num_detections,).
        category_id_to_name: Dictionary mapping class id to class name.
        colors: Colors object for mapping class indices to colors.
        confs: Confidence scores for each detection. Shape: (num_detections,).
        conf_thres: Confidence threshold for displaying detections.
        masks: Instance segmentation masks. Shape: (num_detections, height, width)
        paths: List of file paths for each image in the batch.
        save_path: File path to save the plotted image.
        line_width: Line width for bounding boxes.
        font_size: Font size for class labels.
        alpha: Alpha value for masks.

    Returns:
        np.ndarray: Plotted image as a numpy array

    Note:
        This function supports both tensor and numpy array inputs. It will automatically
        convert tensor inputs to numpy arrays for processing.
    """

    if isinstance(img, torch.Tensor):
        img = img.cpu().float().numpy()
    if isinstance(category, torch.Tensor):
        category = category.cpu().numpy()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy().astype(int)

    # whether to show ground truth labels (without confidence) or predictions (with confidence)
    is_gt = confs is None

    # init annotator
    annotator = Annotator(im=img.copy(), line_width=line_width, font_size=font_size)

    for j, box in enumerate(boxes.astype(np.int64).tolist()):
        c = category[j]
        color = colors(c)
        c = category_id_to_name.get(c, c) if category_id_to_name else c

        if is_gt or confs[j] > conf_thres:

            # draw box with label and conf
            label = f"{c}" if is_gt else f"{c} {confs[j]:.2f}"
            annotator.box_label(box, label, color=color)

            # draw mask
            if masks is not None:
                mask = masks[j]
                annotator.im[mask == 1] = (
                    annotator.im[mask == 1] * (1 - alpha) + np.array(color) * alpha
                )
    if save_path:
        Image.fromarray(annotator.result()).save(save_path)
    return annotator.result()


def display_image(input, figsize=(12, 12)):
    """Display a numpy array or a torch.tensor as an image"""
    # tensor to array
    if isinstance(input, torch.Tensor):
        input = input.cpu().detach().numpy()
    # CHW to HWC
    shape = input.shape
    if len(shape) == 4 and shape[1] in [1, 3]:
        input = np.transpose(input, (0, 2, 3, 1))[0]
    if len(shape) == 3 and shape[0] in [1, 3]:
        input = np.transpose(input, (1, 2, 0))

    plt.figure(figsize=figsize)  # Set the figure size for better visibility
    plt.imshow(
        input, cmap="gray"
    )  # Display the image, you can specify cmap for grayscale if needed
    plt.axis("off")  # Turn off axis labels
    plt.show()  # Render the image
