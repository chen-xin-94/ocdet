import torch
import numpy as np

from postprocessing.find_peaks import peak_local_max
from .constants import BOX_TYPE_TO_SIZE


def get_centroid_coordinates_from_map(mask, min_distance=3, threshold_abs=0.5):
    """
    takes a mask tensor and returns the coordinates of the predicted centroids using peak_local_max

    Returns:
        coordinates: A tensor of predicted centroid coordinates in the format [y, x].
    """

    mask_np = mask.detach().squeeze().cpu().numpy()  # H x W
    coordinates = peak_local_max(
        mask_np,
        min_distance=min_distance,
        exclude_border=False,
        threshold_abs=threshold_abs,
    )
    return torch.tensor(coordinates).long()


def get_centroid_coordinates_from_bbox(bbox):
    """
    get ground truth centroid coordinates from bounding box coordinates

    Parameters:
        bbox: A list of torch tensors of raw COCO bounding box coordinates in the format [x1, y1, w, h].

    Returns:
        coordinates_gt: A tensor of ground truth centroid coordinates in the format [y, x].
    """

    coordinates_gt = []
    for box in bbox:
        x1, y1, w, h = box
        x = x1 + w / 2
        y = y1 + h / 2
        coordinates_gt.append([y.round().item(), x.round().item()])
    return torch.tensor(coordinates_gt).long()


def cad_one_image_one_class_with_bbox(
    coordinates_gt,
    coordinates_pred,
    indices,
    cost_distance,
    bbox,
    do_box_type,
):
    """
    coordinates for either peaks or bbox are not supposed to be normalized to [0,1], they should both be in the same scale (currently specified by args.resized_image_size)
    """

    box_type_to_center_alignment_discerpancy = {
        box_type: 0.0 for box_type in BOX_TYPE_TO_SIZE
    }
    box_type_to_cardinality_penalty = {box_type: 0.0 for box_type in BOX_TYPE_TO_SIZE}
    box_type_to_matched_discrepancy = {box_type: 0.0 for box_type in BOX_TYPE_TO_SIZE}
    box_type_to_n_instances = {box_type: 0 for box_type in BOX_TYPE_TO_SIZE}

    bbox_diag = torch.sqrt(bbox[:, 2] ** 2 + bbox[:, 3] ** 2)

    ##  bbox aware CAD calculation
    # rule 1: if the distance between matched gt and pred points is larger than the diagonal of the bounding box,
    # it is considered as a mismatch, so we filter out these points
    cost_distance /= bbox_diag[:, None]
    cost = cost_distance[indices]
    gt_ind = torch.tensor(indices[0])[cost <= 1]
    pred_ind = torch.tensor(indices[1])[cost <= 1]
    indices = (gt_ind, pred_ind)

    # rule 2: the acutal distance cost for metrics should be scaled by the bounding box diagonal
    alignment_error = cost_distance[indices]
    matched_discrepancy = alignment_error.sum().item()

    # rule 3: for mismatch error (cardinality penalty), simply add 1 for each mismatch keypoint, since the distance cost is already scaled
    n = max(coordinates_gt.size(0), coordinates_pred.size(0))
    cardinality_penalty = n - len(indices[0])

    # change norm to outside this function, so only accumulate the error here
    center_alignment_discerpancy = matched_discrepancy + cardinality_penalty

    box_type_to_center_alignment_discerpancy["all"] += center_alignment_discerpancy
    box_type_to_cardinality_penalty["all"] += cardinality_penalty
    box_type_to_matched_discrepancy["all"] += matched_discrepancy
    box_type_to_n_instances["all"] += n

    if do_box_type:
        ## find out the contribution of each box type to the metrics and modify the dict accordingly
        bbox_area = bbox[:, 2] * bbox[:, 3]
        bbox_type = []
        box_type_limit = BOX_TYPE_TO_SIZE["medium"]
        for i, area in enumerate(bbox_area):
            if area <= box_type_limit[0]:
                bbox_type.append("small")
            elif area > box_type_limit[1]:
                bbox_type.append("large")
            else:
                bbox_type.append("medium")

        for t in bbox_type:
            box_type_to_n_instances[t] += 1
        for i in range(len(gt_ind)):
            t = bbox_type[i]
            box_type_to_matched_discrepancy[t] += alignment_error[i].item()
            box_type_to_center_alignment_discerpancy[t] += alignment_error[i].item()
        # remove the matched ones
        bbox_type = [e for i, e in enumerate(bbox_type) if i not in gt_ind]
        for t in bbox_type:
            box_type_to_cardinality_penalty[t] += 1
            box_type_to_center_alignment_discerpancy[t] += 1

    # return center_alignment_discerpancy, matched_discrepancy, cardinality_penalty, n
    return (
        box_type_to_center_alignment_discerpancy,
        box_type_to_matched_discrepancy,
        box_type_to_cardinality_penalty,
        box_type_to_n_instances,
    )
