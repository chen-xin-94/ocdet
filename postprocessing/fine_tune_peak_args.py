"""
Usage:
1. Use ground truth mask
python postprocessing/fine_tune_peak_args.py --gt
2. Use model prediction, need to specify the configuration file, weight file should be in the same directory as the configuration file and named as best_model.pt
python postprocessing/fine_tune_peak_args.py -c config
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.dataset import CocoCMap
from postprocessing.hungarian import HungarianMatcher
from postprocessing.find_peaks import peak_local_max
from postprocessing.hungarian import HungarianMatcher
from metrics.cad import (
    cad_one_image_one_class_with_bbox,
    get_centroid_coordinates_from_bbox,
)
from utils.utils import dotdict
from build import fix_compatibility, build_model
from predict import predict_one_image
import yaml
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pprint import pprint
import argparse
import numpy as np
import random

seed = 0
# Set the deterministic settings
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# OPTION
## variables to be tuned
ensure_min_distance = True

## experiment 1: find the best min_distance
# min_distances = list(range(1, 11))
# thresholds_abs = [0.5]

## experiment 2: find the best threshold_abs
min_distances = [3]
thresholds_abs = np.linspace(0, 0.9, 10)

## other experiments
# min_distances = [3]
# thresholds_abs = [None, 0.25, 0.5, 0.75]
# thresholds_abs = [None, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.75]

BATCH_SIZE = 1
COCO_FOLDER = "/mnt/ssd2/xxx/data/coco/"


parser = argparse.ArgumentParser()
parser.add_argument("--gt", action="store_true", help="use ground truth mask")
parser.add_argument(
    "-c",
    "--config",
    help="configuration file *.yaml",
    type=str,
    required=False,
    default="config",
)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

if args.gt:
    print("Using ground truth mask")
    resized_image_sizes = [(320, 320)]
    # resized_image_sizes = [None, (320, 320), (160, 160), (80, 80)]

    matcher = HungarianMatcher(logits_weight=0)
else:
    config_path = args.config
    print(f"Using config: {config_path}")
    # Load config
    with open(config_path, "r") as f:
        config = dotdict(yaml.safe_load(f))

    # Fix compatibility for older configurations
    if "_wandb" in config:
        config.pop("_wandb")
        if "wandb_version" in config:
            config.pop("wandb_version")
        for k, v in config.items():
            config[k] = v["value"]

    config = fix_compatibility(config)

    model = build_model(config, device)

    # weight should always be in the same directory as the config file and named as best_model.pt
    weitght_path = os.path.join(os.path.dirname(config_path), "best_model.pt")
    state_dict = torch.load(weitght_path, map_location=device)
    # chagne the key name for old models
    state_dict = {k.replace("fpn.", "neck."): v for k, v in state_dict.items()}
    state_dict = {k.replace("fpn_head.", "head."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    resized_image_sizes = [tuple(config.resized_image_size)]

    matcher = HungarianMatcher(logits_weight=1)

## data
coco_folder = COCO_FOLDER
# split = "train2017"
split = "val2017"
root = os.path.join(coco_folder, split)
annFile = os.path.join(coco_folder, "annotations", f"instances_{split}_person.json")

for resized_image_size in resized_image_sizes:

    if resized_image_size is None:
        transform = None
    else:
        _, transform = CocoCMap.build_transforms(resized_image_size)
    dataset = CocoCMap(root, annFile, transform, target_category_ids=[1])  # only person
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=CocoCMap.collate_fn,
        num_workers=8,
    )
    ## process

    cad_per_min_distance = {}
    md_per_min_distance = {}
    cp_per_min_distance = {}
    ce_per_min_distance = {}

    for thresh in thresholds_abs:
        for min_distance in min_distances:
            no_gt = 0
            n_acc = 0
            cad_acc = 0
            md_acc = 0
            cp_acc = 0
            counting_error = 0
            for i, batch in enumerate(tqdm(dataloader)):

                # assume batch size is always 1 and only person
                mask = batch["masks"][0][0]
                bbox = batch["bboxes"][0]

                if args.gt:  # get peaks from ground truth mask
                    mask_np = mask.cpu().numpy()
                    coordinates_pred = peak_local_max(
                        mask_np,
                        min_distance=min_distance,
                        exclude_border=False,
                        threshold_abs=thresh if thresh > 0 else None,
                        ensure_min_distance=ensure_min_distance,
                    )
                    coordinates_pred = torch.tensor(coordinates_pred)
                    logits_pred = torch.ones(len(coordinates_pred), 1)
                else:  # get peaks from model prediction
                    coordinates_pred, logits_pred = predict_one_image(
                        model,
                        batch["images"],
                        device,
                        min_distance=min_distance,
                        threshold_abs=thresh if thresh > 0 else None,
                        n_classes=1,  # only person
                        ret_logits=True,
                    )
                    coordinates_pred = coordinates_pred[0]  # only person
                    logits_pred = logits_pred[0]  # only person

                coordinates_gt = get_centroid_coordinates_from_bbox(bbox)
                logits_gt = (
                    mask.detach()
                    .cpu()[coordinates_gt[:, 0], coordinates_gt[:, 1]]
                    .view(-1, 1)
                )

                # if no gt coordinates, caused by severe resizing
                if coordinates_gt.size(0) == 0:
                    no_gt += 1
                    continue

                # if no pred coordinates after postprocessing
                if coordinates_pred.size(0) == 0:
                    n_acc += len(coordinates_gt)
                    cad_acc += len(coordinates_gt)
                    cp_acc += len(coordinates_gt)
                    counting_error += len(coordinates_gt)
                    continue

                # counting error
                counting_error += abs(len(coordinates_gt) - len(coordinates_pred))

                indices, cost_distance = matcher.match(
                    coordinates_gt, coordinates_pred, logits_gt, logits_pred
                )

                # only use the standard CAD with no box type differentiation
                (
                    box_type_to_center_alignment_discerpancy,
                    box_type_to_matched_discrepancy,
                    box_type_to_cardinality_penalty,
                    box_type_to_n_instances,
                ) = cad_one_image_one_class_with_bbox(
                    coordinates_gt,
                    coordinates_pred,
                    indices,
                    cost_distance,
                    bbox,
                    False,  # do_box_type,
                )
                cad_acc += box_type_to_center_alignment_discerpancy["all"]
                md_acc += box_type_to_matched_discrepancy["all"]
                cp_acc += box_type_to_cardinality_penalty["all"]
                n_acc += box_type_to_n_instances["all"]

            cad_per_min_distance[min_distance] = cad_acc / n_acc
            md_per_min_distance[min_distance] = md_acc / n_acc
            cp_per_min_distance[min_distance] = cp_acc / n_acc
            ce_per_min_distance[min_distance] = counting_error / n_acc

            print(
                f"Resize Image Size: {resized_image_size}, threshold_abs: {thresh}, min_distance: {min_distance}, Ensure Min Distance: {ensure_min_distance}, CAD: {cad_per_min_distance[min_distance]}, MD: {md_per_min_distance[min_distance]}, CP: {cp_per_min_distance[min_distance]}, Counting Error per image: {ce_per_min_distance[min_distance]}"
            )

        print("\n\n")
        print(f"Ensure Min Distance: {ensure_min_distance}")
        print(f"Resize Image Size: {resized_image_size}")
        print(f"Threshold: {thresh}")
        print("Center Alignment Discrepancy per min_distance")
        pprint(cad_per_min_distance)
        print("Matched Discrepancy per min_distance")
        pprint(md_per_min_distance)
        print("Cardinality Penalty per min_distance")
        pprint(cp_per_min_distance)
        print("Counting Error per min_distance")
        pprint(ce_per_min_distance)
        print("\n\n")

        if no_gt > 0:
            print(f"# images with no GT after resizeing: {no_gt}")
