"""
Usage:
CUDA_VISIBLE_DEVICES=1 python eval_yolo.py --r 320 --n_classes 80
"""

from postprocessing.hungarian import HungarianMatcher
from metrics.constants import BOX_TYPES, BOX_TYPE_TO_SIZE
from metrics.cad import (
    cad_one_image_one_class_with_bbox,
    get_centroid_coordinates_from_bbox,
)
from engine import (
    finalize_cad_results,
    finalize_pr_results,
    update_results_dict,
)
from utils.coco import COCO_ID_TO_CHANNEL
from build import build_dataloader
from datasets.dataset import CocoCMap

from ultralytics.utils.ops import ltwh2xyxy
from torchvision.ops import nms
import torch
import json
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
from collections import defaultdict
from tqdm import tqdm
import pprint
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-r",
    "--resized_image_size",
    default=320,
    type=int,
    help="resized image size (use int to represent square)",
)
parser.add_argument(
    "-n",
    "--n_classes",
    default=80,
    type=int,
    help="number of classes to evaluate",
)
args = parser.parse_args()

RESIZE = args.resized_image_size
MODEL_NAMES = [
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "yolov8l",
    "yolov8x",
    "yolo11n",
    "yolo11s",
    "yolo11m",
    "yolo11l",
    "yolo11x",
]

CONF_THRES = 0.25
IOU_THRES = 0.7

n_classes = args.n_classes
target_category_ids = list(COCO_ID_TO_CHANNEL.keys())[
    :n_classes
]  # NOTE: currently only support first n_classes

COCO_FOLDER = "/mnt/ssd2/xin/data/coco/"
coco_folder = Path(COCO_FOLDER)
split = "val2017"
root = coco_folder / split

if n_classes == 1:
    annFile = coco_folder / "annotations" / f"instances_{split}_person.json"
elif n_classes == 8:
    annFile = coco_folder / "annotations" / f"instances_{split}_c8.json"
elif n_classes == 80:
    annFile = coco_folder / "annotations" / f"instances_{split}.json"
else:
    raise NotImplementedError(f"n_classes={n_classes} not implemented")

coco = COCO(annFile)
transform = A.Compose(
    [
        A.Resize(RESIZE, RESIZE),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=1,
        ),
        ToTensorV2(transpose_mask=True),
    ],
    bbox_params=A.BboxParams(
        format="coco",
        min_visibility=0.3,
        min_area=1,
        label_fields=["category_ids"],
    ),
)
dataset = CocoCMap(
    root=root,
    annFile=annFile,
    transform=transform,
    mask_folder_name="cmaps",
    target_category_ids=target_category_ids,
)


for model_name in MODEL_NAMES:

    dataloader = build_dataloader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=8,
        prefetch_factor=2,
        pin_memory=True,
        drop_last=False,
        collate_fn=dataset.collate_fn,
    )

    json_path = f"/mnt/ssd2/xin/repo/ultralytics/customization/runs/detect/val_{model_name}_r{RESIZE}/predictions.json"
    json_dict_path = json_path.replace("predictions.json", "predicrtions_dict.json")
    cad_results_path = json_path.replace(
        "predictions.json", f"cad_results_c{n_classes}.json"
    )
    eval_dict_path = json_path.replace(
        "predictions.json", f"eval_dict_c{n_classes}.json"
    )
    pr_results_path = json_path.replace(
        "predictions.json", f"pr_results_c{n_classes}.json"
    )

    if not Path(json_path).parent.exists():
        print(f"val for {model_name} not exists in {Path(json_path).parent}")

    if (
        Path(cad_results_path).exists()
        and Path(eval_dict_path).exists()
        and Path(pr_results_path).exists()
    ):
        print(f"Skip {model_name}")
        continue

    # parse the predictions
    if Path(json_dict_path).exists():
        with open(json_dict_path, "r") as f:
            predictions_dict = json.load(f)
    else:
        # generate a dict version of the predictions
        with open(json_path, "r") as f:
            predictions = json.load(f)

        predictions_dict = {}
        for pred in predictions:
            image_id = str(
                pred["image_id"]
            )  # str, so that consistent with loaded json dict
            if image_id not in predictions_dict:
                predictions_dict[image_id] = defaultdict(list)
            predictions_dict[image_id]["boxes"].append(pred["bbox"])
            predictions_dict[image_id]["scores"].append(pred["score"])
            predictions_dict[image_id]["labels"].append(pred["category_id"])

        with open(json_dict_path, "w") as f:
            json.dump(predictions_dict, f)

    eval_dict = defaultdict(float)

    # creat a dict for each channel to save cad results
    cad_results = {
        ch: {
            "cad": {
                box_type: None for box_type in BOX_TYPE_TO_SIZE
            },  # box_type_to_center_alignment_discerpancy
            "cp": {
                box_type: None for box_type in BOX_TYPE_TO_SIZE
            },  # box_type_to_cardinality_penalty
            "md": {
                box_type: None for box_type in BOX_TYPE_TO_SIZE
            },  # box_type_to_matched_discrepancy
            "n": {
                box_type: None for box_type in BOX_TYPE_TO_SIZE
            },  # box_type_to_n_instances
        }
        for ch in range(n_classes)
    }

    pr_results = {
        ch: {
            "TP": 0,
            "FP": 0,
            "FN": 0,
        }
        for ch in range(args.n_classes)
    }

    matcher = HungarianMatcher(distance_weight=1, logits_weight=1)

    for batch_i, batch in enumerate(tqdm(dataloader, desc=f"Evaluation {model_name}")):
        image_ids, X, y, bboxes, category_ids = (
            batch["image_ids"],
            batch["images"],
            batch["masks"],
            batch["bboxes"],
            batch["category_ids"],
        )
        for batch_idx in range(len(batch["images"])):
            mask_all = y[batch_idx]  # (n_classes, h, w)
            gt_bbox_all = bboxes[batch_idx]  # (n_instances, 4)
            gt_cls_all = category_ids[batch_idx]  # (n_instances,)
            image_id = image_ids[batch_idx].item()  # (1,)

            for target_category_id in target_category_ids:
                ind_gt = gt_cls_all == target_category_id
                if ind_gt.sum() > 0:  # if there are gt instances in this class

                    ## get gt coordinates and logits
                    ch = COCO_ID_TO_CHANNEL[
                        target_category_id
                    ]  # NOTE: this channel assignment is not ideal if skip some classes in the middle
                    mask = mask_all[ch]  # (h, w)
                    gt_bbox = gt_bbox_all[ind_gt]
                    gt_cls = gt_cls_all[ind_gt]
                    coordinates_gt = get_centroid_coordinates_from_bbox(gt_bbox)
                    logits_gt = (
                        mask.detach()
                        .cpu()[coordinates_gt[:, 0], coordinates_gt[:, 1]]
                        .view(-1, 1)
                    )

                    ## get pred coordinates and logits
                    W, H = (
                        coco.loadImgs(image_id)[0]["width"],
                        coco.loadImgs(image_id)[0]["height"],
                    )
                    # filter by class
                    category_ids_pred = torch.tensor(
                        predictions_dict[str(image_id)]["labels"]
                    )
                    ind = category_ids_pred == target_category_id
                    # filter by confidence
                    logits_pred = torch.tensor(
                        predictions_dict[str(image_id)]["scores"]
                    )
                    ind2 = logits_pred > CONF_THRES
                    ind &= ind2
                    if torch.sum(ind) == 0:
                        coordinates_pred = torch.zeros((0, 2))
                        logits_pred = torch.zeros((0, 1))
                    else:
                        # nms
                        logits_pred = logits_pred[ind]
                        boxes_ltwh = torch.tensor(
                            predictions_dict[str(image_id)]["boxes"]
                        )[ind]
                        boxes_xyxy = ltwh2xyxy(boxes_ltwh)
                        ind_nms = nms(boxes_xyxy, logits_pred, IOU_THRES)

                        if torch.sum(ind_nms) == 0:
                            coordinates_pred = torch.zeros((0, 2))
                            logits_pred = torch.zeros((0, 1))
                        else:
                            # get coordinates from survived predicted bbox
                            logits_pred = logits_pred[ind_nms]
                            logits_pred = logits_pred.view(-1, 1)

                            boxes_ltwh = boxes_ltwh[ind_nms]
                            coordinates_pred = torch.zeros((len(boxes_ltwh), 2))
                            coordinates_pred[:, 1] = (
                                boxes_ltwh[:, 0] + boxes_ltwh[:, 2] / 2
                            )
                            coordinates_pred[:, 0] = (
                                boxes_ltwh[:, 1] + boxes_ltwh[:, 3] / 2
                            )
                            # norm then reisze for predictions
                            # since predictions are in original image size, but gt is in resized size
                            coordinates_pred = (
                                coordinates_pred / torch.tensor([H, W]) * RESIZE
                            ).long()
                    indices, cost_distance = matcher.match(
                        coordinates_gt, coordinates_pred, logits_gt, logits_pred
                    )

                    # update CAD metrics: by modifying the dict in place
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
                        gt_bbox,
                        True,  # do_box_type
                    )

                    # update PR metrics
                    bbox_diag = torch.sqrt(gt_bbox[:, 2] ** 2 + gt_bbox[:, 3] ** 2)
                    # rule 1: if the distance between matched gt and pred points is larger than the diagonal of the bounding box,
                    # it is considered as a mismatch, so we filter out these points
                    cost_distance /= bbox_diag[:, None]
                    cost = cost_distance[indices]
                    gt_ind = torch.tensor(indices[0])[cost <= 1]
                    pred_ind = torch.tensor(indices[1])[cost <= 1]
                    # update PR dict
                    pr_results[ch]["TP"] += len(gt_ind)
                    pr_results[ch]["FP"] += len(coordinates_pred) - pred_ind.size(0)
                    pr_results[ch]["FN"] += len(coordinates_gt) - gt_ind.size(0)

                    # accumulate the results
                    cad_results[ch]["cad"] = {
                        k: v + (cad_results[ch]["cad"].get(k) or 0.0)
                        for k, v in box_type_to_center_alignment_discerpancy.items()
                    }
                    cad_results[ch]["cp"] = {
                        k: v + (cad_results[ch]["cp"].get(k) or 0.0)
                        for k, v in box_type_to_cardinality_penalty.items()
                    }
                    cad_results[ch]["md"] = {
                        k: v + (cad_results[ch]["md"].get(k) or 0.0)
                        for k, v in box_type_to_matched_discrepancy.items()
                    }
                    cad_results[ch]["n"] = {
                        k: v + (cad_results[ch]["n"].get(k) or 0.0)
                        for k, v in box_type_to_n_instances.items()
                    }

    cad_results = finalize_cad_results(cad_results)
    pr_results = finalize_pr_results(
        pr_results,
    )
    update_results_dict(eval_dict, cad_results, pr_results)

    print(f"\nResults for {model_name}")
    print("eval_dict")
    pprint.pprint(eval_dict)
    print("cad_results")
    pprint.pprint(cad_results)
    print()
    pprint.pprint(pr_results)
    print()

    # save the results
    with open(cad_results_path, "w") as f:
        json.dump(cad_results, f)
    print(f"Saved CAD results to {cad_results_path}")
    with open(pr_results_path, "w") as f:
        json.dump(pr_results, f)
    print(f"Saved PR results to {cad_results_path}")
    with open(eval_dict_path, "w") as f:
        json.dump(eval_dict, f)
    print(f"Saved evaluation results to {eval_dict_path}")
