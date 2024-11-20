"""
Usage:
python predict.py --input_path images/000000133418.jpg --trained weights/ocdet-x.pt --config configs/cmap_c80/ocdet-x.yaml --n_classes 80 --gpu 0 --min_distance 3 --threshold_abs 0.5 --input_size 320 --vis --save
"""

import matplotlib.pyplot as plt
import argparse
import yaml
import pprint
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
from build import build_model, fix_compatibility
from utils.utils import dotdict
from utils.plot_utils import plot_points_on_image, ColorsOCDet
from metrics.cad import get_centroid_coordinates_from_map
from utils.coco import COCO_CHANNEL_TO_ID

COLORS = ColorsOCDet()


def get_args():
    parser = argparse.ArgumentParser(description="Person Centroid Detection Prediction")

    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input image or directory.",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Configuration file *.yaml",
        type=str,
        required=True,
    )
    parser.add_argument("--n_classes", default=80, type=int, help="Number of classes.")
    parser.add_argument(
        "--trained", type=str, required=True, help="Path to trained model."
    )
    parser.add_argument(
        "--min_distance",
        default=3,
        type=int,
        help="Minimum distance between detected centroids.",
    )
    parser.add_argument(
        "--threshold_abs",
        default=0.5,
        type=float,
        help="Absolute threshold for detection.",
    )
    parser.add_argument(
        "--input_size",
        "-r",
        default=320,
        type=int,
        help="Input size for the model (assuming square input).",
    )
    parser.add_argument("--vis", action="store_true", help="Visualize the results.")
    parser.add_argument("--save", action="store_true", help="Save the visualization.")

    return parser.parse_args()


def get_image_files(directory):
    # Returns a list of image file paths
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if (
                file.lower().endswith(image_extensions)
                and "pred_coord_overlay" not in file
            ):
                image_files.append(os.path.join(root, file))
    return image_files


def predict_one_image(
    model,
    img_tensor,
    device,
    min_distance=3,
    threshold_abs=0.5,
    n_classes=1,
    ret_logits=False,
):
    """
    Predicts the coordinates of centroids from an input tensor using a trained model.

    Args:
        model: The trained PyTorch model for making predictions.
        input_tensor: The preprocessed input tensor representing the image.
        device: The device (CPU or CUDA) to perform the computation on.
        min_distance: Minimum distance between detected peaks (centroids).
        threshold_abs: Minimum threshold for peak detection.
        n_classes: The number of output classes/channels in the model prediction.

    Returns:
        coordinates_pred: A dictionary mapping each class to a torch tensor of shape (n, 2), with each coordinate in the format [y, x].
    """

    # suppose model is already in device and in eval mode

    # Move input tensor to the appropriate device
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        # Run the model on the input tensor
        y_logits = model(img_tensor).detach()

        # Upsample if necessary to match input size
        if y_logits.size()[2:] != img_tensor.size()[2:]:
            y_logits = F.interpolate(
                y_logits,
                size=img_tensor.size()[2:],
                mode="bilinear",
                align_corners=False,
            )

        # Apply sigmoid to get probabilities
        y_probas = torch.sigmoid(y_logits).cpu()

    # Initialize the dictionary to store predicted coordinates
    coordinates_pred = {}
    logits_pred = {}

    # Iterate over each class and extract centroid coordinates
    for ch in range(n_classes):
        y_proba = y_probas[0, ch]  # Assuming batch size is 1
        # Get centroid coordinates from the prediction map
        coordinates_pred[ch] = get_centroid_coordinates_from_map(
            y_proba, min_distance, threshold_abs
        )
        if ret_logits:
            logits_pred[ch] = y_proba.cpu()[
                coordinates_pred[ch][:, 0], coordinates_pred[ch][:, 1]
            ].view(-1, 1)
    if ret_logits:
        return coordinates_pred, logits_pred
    else:
        return coordinates_pred


def inference(
    input_path,
    trained_model_path,
    config_path,
    gpu=0,
    min_distance=3,
    threshold_abs=0.5,
    input_size=320,
    vis=False,
    save=False,
    n_classes=1,
):
    """
    Performs the inference pipeline, including loading the model, preprocessing input images,
    and calling the prediction function to get coordinates.

    Args:
        input_path: Path to the input image or directory of images.
        trained_model_path: Path to the trained model weights file.
        config_path: Path to the model configuration YAML file.
        gpu: The GPU index to use for inference (default 0). If not available, will fall back to CPU.
        min_distance: Minimum distance between peaks for centroid detection.
        threshold_abs: Threshold for peak detection in the output probability map.
        input_size: Size to which input images are resized for inference.
        vis: Whether to display the image with predicted centroids overlaid.
        save: Whether to save the output image with predicted centroids.
        n_classes: Number of output classes (or channels) in the model's prediction.

    Returns:
        None. Prints predicted coordinates and optionally visualizes or saves the image with overlaid centroids.
    """
    # Load config
    with open(config_path, "r") as f:
        config = dotdict(yaml.safe_load(f))
    print(f"Loaded config from {config_path}")

    # Fix compatibility for older configurations
    if "_wandb" in config:
        config.pop("_wandb")
        if "wandb_version" in config:
            config.pop("wandb_version")
        for k, v in config.items():
            config[k] = v["value"]

    config = fix_compatibility(config)

    # Update args with config, giving priority to function arguments
    args_dict = {}
    args_dict.update(config)
    function_args = {
        "gpu": gpu,
        "input_path": input_path,
        "trained": trained_model_path,
        "config": config_path,
        "min_distance": min_distance,
        "threshold_abs": threshold_abs,
        "input_size": input_size,
        "vis": vis,
        "save": save,
        "n_classes": n_classes,
        "resume": None,
    }
    # override config with function_args
    for key, value in function_args.items():
        args_dict[key] = value

    args = dotdict(args_dict)
    pprint.pprint(args)

    # Set device for computation
    if args.gpu != -1:
        device = torch.device("cuda:" + str(args.gpu))
        # print("Use GPU:{} for training".format(args.gpu)) # better to use CUDA_VISIBLE_DEVICES to specify the GPU
    else:
        device = torch.device("cpu")
        print("Use CPU for training")

    # Build and load the model
    model = build_model(args, device)
    state_dict = torch.load(args.trained, map_location=device)
    # state_dict = {k.replace("fpn.", "neck."): v for k, v in state_dict.items()}
    # state_dict = {k.replace("fpn_head.", "head."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Check if input_path is a file or a directory
    if os.path.isfile(input_path):
        image_files = [input_path]
    elif os.path.isdir(input_path):
        image_files = get_image_files(input_path)
        vis = False  # Force vis to False for directory input
        args.vis = False
    else:
        raise ValueError(f"{input_path} is not a valid file or directory")

    # Define the preprocessing pipeline
    preprocess = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    for image_file in image_files:
        print(f"Processing {image_file}")
        image = Image.open(image_file).convert("RGB")

        # Preprocess the image and add a batch dimension
        X = preprocess(image).unsqueeze(0)
        # image_resized = image.resize((input_size, input_size))

        # Predict coordinates
        coordinates_pred = predict_one_image(
            model,
            X,
            device,
            min_distance=min_distance,
            threshold_abs=threshold_abs,
            n_classes=n_classes,
        )

        # filp from coords from y,x to x,y for plotting
        coordinates_pred = {
            k: v.numpy()[:, [1, 0]] for k, v in coordinates_pred.items()
        }
        # resize coordinates to original image size
        w, h = image.size
        for k, v in coordinates_pred.items():
            coordinates_pred[k] = v * [w / input_size, h / input_size]

        if vis or save:
            for ch, coords in coordinates_pred.items():
                # c = COCO_CHANNEL_TO_ID[ch]
                image = plot_points_on_image(image, coords, COLORS(ch))

        if vis:
            plt.imshow(image)
            plt.axis("off")
            plt.show()

        if save:
            # Save the image with predicted coordinates overlay
            image_file_path = Path(image_file)
            output_image_path = image_file_path.with_name(
                f"{image_file_path.stem}_pred_coord_overlay{image_file_path.suffix}"
            )
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            plt.imsave(str(output_image_path), image)
            print(f"Saved overlay image to {output_image_path}")

        # # Print predicted coordinates
        # print(f"Predicted coordinates for {image_file}:")
        # pprint.pprint(coordinates_pred)


def main():
    args = get_args()
    inference(
        input_path=args.input_path,
        trained_model_path=args.trained,
        config_path=args.config,
        gpu=args.gpu,
        min_distance=args.min_distance,
        threshold_abs=args.threshold_abs,
        input_size=args.input_size,
        vis=args.vis,
        save=args.save,
        n_classes=args.n_classes if hasattr(args, "n_classes") else 1,  # Default to 1
    )


if __name__ == "__main__":
    main()
