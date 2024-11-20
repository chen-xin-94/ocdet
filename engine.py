from tqdm.auto import tqdm
from collections import defaultdict
import time
import wandb
import pprint
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc

from metrics.pixelwise import calculate_metrics, calculate_confusion_matrix
from utils.plot_utils import visualize_batch_single_cls, visualize_batch_multi_cls
from utils.utils import (
    AverageMeter,
    ProgressMeter,
)
from postprocessing.hungarian import HungarianMatcher
from metrics.constants import BOX_TYPES, BOX_TYPE_TO_SIZE
from metrics.cad import (
    cad_one_image_one_class_with_bbox,
    get_centroid_coordinates_from_map,
    get_centroid_coordinates_from_bbox,
)
from utils.deterministic import deterministic

KEYS_CAD = [
    "CAD",
    "MD",
    "CP",
    "CAD_small",
    "MD_small",
    "CP_small",
    "CAD_medium",
    "MD_medium",
    "CP_medium",
    "CAD_large",
    "MD_large",
    "CP_large",
    "CAD_preds",
]

KEYS_PR = ["TP", "FP", "FN", "P", "R", "F1"]


def train_val_one_epoch(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    warmup_scheduler,
    train_metrics: list,
    val_metrics: list,
    epoch: int,  # 0-indexed
    device: torch.device,
    args,
):
    train_times = {
        "batch_time": AverageMeter("Batch Time", ":6.3f"),
        "data_time": AverageMeter("Data Time", ":6.3f"),
        "metric_time": AverageMeter("Metric Time", ":6.3f"),
        "end": time.time(),  # a timer to be reset at the end of each batch
    }
    val_times = {
        "batch_time": AverageMeter("Batch Time", ":6.3f"),
        "data_time": AverageMeter("Data Time", ":6.3f"),
        "metric_time": AverageMeter("Metric Time", ":6.3f"),
        "all_time": AverageMeter("Time", ":6.3f"),
    }

    train_results = {"loss": AverageMeter("Loss", ":6.3f")}
    if args.deep_supervision:
        train_results["loss_r320"] = AverageMeter("Loss_r320", ":6.3f")
        train_results["loss_r80"] = AverageMeter("Loss_r80", ":6.3f")
    for metric_name in train_metrics:
        train_results[metric_name] = AverageMeter(metric_name.capitalize(), ":6.3f")

    val_results = {"loss": AverageMeter("Loss", ":6.3f")}
    for metric_name in val_metrics:
        val_results[metric_name] = AverageMeter(metric_name.capitalize(), ":6.3f")

    # NOTE: meters are defined after child objects of train_results and val_results,
    # which gets automatical updates after running train_stp and val_stpe using only shallow copies of the results
    train_progress = ProgressMeter(
        num_batches=len(train_dataloader),
        meters=[train_results["loss"]]  # make sure loss is first
        + [v for k, v in train_results.items() if "loss" not in k]
        + [
            train_times["batch_time"],
            train_times["data_time"],
            train_times["metric_time"],
        ],
        prefix="Train: ",
    )
    val_progress = ProgressMeter(
        num_batches=len(val_dataloader),
        meters=[val_results["loss"]]  # make sure loss is first
        + [v for k, v in val_results.items() if k != "loss"]
        + [
            # val_times["batch_time"],
            # val_times["data_time"],
            # val_times["metric_time"],
            val_times["all_time"]
        ],
        prefix="Val: ",
    )

    # Reset the timer for the first batch
    train_times["end"] = time.time()
    for batch_i, batch in enumerate(train_dataloader, 1):  # 1-indexed
        image_ids, X, y, bboxes, category_ids = (
            batch["image_ids"],
            batch["images"],
            batch["masks"],
            batch["bboxes"],
            batch["category_ids"],
        )
        y_r80 = batch["masks_r80"]

        step = (epoch - 1) * len(train_dataloader) + batch_i
        wandb.log({"epoch": epoch}, step=step)
        # Perform a training step
        train_step(
            model=model,
            X=X,
            y=y,
            y_r80=y_r80,
            criterion=criterion,
            optimizer=optimizer,
            train_metrics=train_metrics,
            train_times=train_times,
            train_results=train_results,
            metric_every=args.metric_every,
            step=step,
            device=device,
        )

        # update learning rate
        if args.lr_scheduler.lower() == "step":
            with warmup_scheduler.dampening():
                if batch_i + 1 == len(train_dataloader):
                    lr_scheduler.step()
        elif args.lr_scheduler.lower() == "cosine":
            with warmup_scheduler.dampening():
                lr_scheduler.step()
        else:
            raise NotImplementedError(
                f"lr_scheduler: {args.lr_scheduler} not implemented"
            )

        # whether to print training progress
        if batch_i % args.print_every == 0:
            train_progress.display(batch_i)

        # if it's the val_every step, but not the last step of the entire training process
        if batch_i % args.val_every == 0 and not (
            epoch == args.epochs and batch_i == len(train_dataloader)
        ):

            # whether it's the last val step for the current epoch:
            is_last = batch_i + args.val_every > len(train_dataloader)

            val_step(
                model=model,
                dataloader=val_dataloader,
                criterion=criterion,
                val_results=val_results,
                val_metrics=val_metrics,
                val_times=val_times,
                step=step,
                device=device,
                args=args,
                is_last=is_last,
            )
            # print results
            val_progress.display_summary()

        # Reset the timer for the next batch
        train_times["end"] = time.time()

    return train_results, val_results, step


def train_step(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    y_r80: torch.Tensor,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_metrics: list,
    train_times: dict,
    train_results: dict,
    metric_every: int,
    step: int,
    device,
):
    # switch to train mode
    model.train()

    # measure data loading time
    train_times["data_time"].update(time.time() - train_times["end"])
    train_times["end"] = time.time()

    # standard training steps
    X, y = X.to(device), y.to(device)

    y_logits = model(X)

    if y_r80[0] is not None:
        # if not args.deep_supervision, y_r80 should be a list of None,
        # otherwise, it should a tensor
        y_r80 = y_r80.to(device)
        loss_r80 = criterion(y_logits, y_r80)
    else:
        loss_r80 = 0

    # upsample if necessary
    if y_logits.size() != y.size():
        y_logits = F.interpolate(
            y_logits, size=y.size()[-2:], mode="bilinear", align_corners=False
        )
    loss_r320 = criterion(y_logits, y)
    loss = loss_r320 + loss_r80  # TODO: add weight
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure training time
    train_times["batch_time"].update(time.time() - train_times["end"])
    train_times["end"] = time.time()

    # update metrics only once every metric_every batches
    if train_metrics and step % metric_every == 0:
        tp, tn, fp, fn = calculate_confusion_matrix(y_logits, y).values()

        accuracy, precision, recall, f1 = calculate_metrics(tp, tn, fp, fn).values()

        for metric_name in train_metrics:
            train_results[metric_name].update(eval(metric_name), X.size(0))

        # measure metric time
        train_times["metric_time"].update(time.time() - train_times["end"])
        # log metrics
        wandb.log({f"train/{k}": v.val for k, v in train_results.items()}, step=step)

    # deliberately update loss after metrics to log loss separately for each batch as
    train_results["loss"].update(loss.item(), X.size(0))

    # log
    wandb.log(
        {
            "train/loss": loss.item(),
            "lr": optimizer.param_groups[0]["lr"],
        },
        step=step,
    )
    if y_r80[0] is not None:  # use deep supervision
        train_results["loss_r320"].update(loss_r320.item(), X.size(0))
        train_results["loss_r80"].update(loss_r80.item(), X.size(0))
        wandb.log(
            {"train/loss_r320": loss_r320.item(), "train/loss_r80": loss_r80.item()},
            step=step,
        )


def val_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    val_results: dict,
    val_metrics: list,
    val_times: dict,
    step: int,
    device,
    args,
    is_last: bool = False,
):
    model.eval()

    start = time.time()
    tp, tn, fp, fn = 0, 0, 0, 0
    count = 0  # count otal number of samples

    if is_last:
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
            for ch in range(args.n_classes)
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

    with torch.inference_mode():
        # end = time.time()
        for batch in tqdm(dataloader, desc="Val step"):
            image_ids, X, y, bboxes, category_ids = (
                batch["image_ids"],
                batch["images"],
                batch["masks"],
                batch["bboxes"],
                batch["category_ids"],
            )

            # # measure data loading time
            # val_times["data_time"].update(time.time() - end)
            # end = time.time()

            # standard validation steps
            X, y = X.to(device), y.to(device)
            y_logits = model(X).detach()
            # upsample if necessary
            if y_logits.size() != y.size():
                y_logits = F.interpolate(
                    y_logits, size=y.size()[-2:], mode="bilinear", align_corners=False
                )
            loss = criterion(y_logits, y)

            y_probas = torch.sigmoid(y_logits).cpu()

            ## update metrics
            # calculate confusion matrix
            val_results["loss"].update(loss.item(), X.size(0))
            _tp, _tn, _fp, _fn = calculate_confusion_matrix(y_logits, y).values()
            tp += _tp
            tn += _tn
            fp += _fp
            fn += _fn

            if is_last:
                cad_results, pr_results = calculate_metrics_batch(
                    batch,
                    y,
                    y_probas,
                    bboxes,
                    category_ids,
                    args.min_distance,
                    args.threshold_abs,
                    matcher,
                    BOX_TYPES,
                    cad_results,
                    args.channel_to_id,
                    pr_results,
                )

        ## accumulate metrics
        accuracy, precision, recall, f1 = calculate_metrics(tp, tn, fp, fn).values()
        if is_last:
            cad_results = finalize_cad_results(
                cad_results,
                BOX_TYPES,
            )
            pr_results = finalize_pr_results(pr_results)

        ## update val_results
        for metric_name in val_metrics:
            val_results[metric_name].update(eval(metric_name))
        if is_last:
            update_results_dict(
                val_results,
                cad_results,
                pr_results,
            )
            for k in KEYS_CAD + KEYS_PR:
                wandb.log({f"val/{k}": val_results[k]}, step=step)

        # val_times["metric_time"].update(time.time() - end)

        # log other metrics
        wandb.log(
            {
                f"val/{k}": v.val
                for k, v in val_results.items()
                if k not in KEYS_CAD + KEYS_PR
            },
            step=step,
        )

        # measure time
        val_times["all_time"].update(time.time() - start)


def evaluation(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    metrics: list,
    device: torch.device,
    args,
):
    """Evaluate the model on the given data_loader using the given metrics."""

    model.eval()

    y_true_all = []
    y_probas_all = []
    eval_dict = defaultdict(float)

    tp, tn, fp, fn = 0, 0, 0, 0

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
        for ch in range(args.n_classes)
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

    with torch.inference_mode():
        for batch_i, batch in enumerate(tqdm(dataloader, desc="Evaluation step")):
            image_ids, X, y, bboxes, category_ids = (
                batch["image_ids"],
                batch["images"],
                batch["masks"],
                batch["bboxes"],
                batch["category_ids"],
            )
            X, y = X.to(device), y.to(device)
            y_logits = model(X).detach()
            # upsample if necessary
            if y_logits.size() != y.size():
                y_logits = F.interpolate(
                    y_logits, size=y.size()[-2:], mode="bilinear", align_corners=False
                )
            y_probas = torch.sigmoid(y_logits).cpu()
            # y_preds = y_probas.view(-1) >= 0.5

            ## update metrics
            # calculate confusion matrix
            _tp, _tn, _fp, _fn = calculate_confusion_matrix(y_logits, y).values()
            tp += _tp
            tn += _tn
            fp += _fp
            fn += _fn

            cad_results, pr_results = calculate_metrics_batch(
                batch,
                y,
                y_probas,
                bboxes,
                category_ids,
                args.min_distance,
                args.threshold_abs,
                matcher,
                BOX_TYPES,
                cad_results,
                args.channel_to_id,
                pr_results,
            )

            if args.roc and args.n_classes == 1:
                y_true = y.detach().cpu().view(-1) >= 0.5
                # accumulate the results
                y_probas_all.append(y_probas.view(-1))
                y_true_all.append(y_true)

            if batch_i == 0 and args.vis:
                # use the first batch for visualization
                if args.n_classes == 1:
                    eval_dict["vis"] = visualize_batch_single_cls(
                        args.task,
                        10,
                        X.detach().cpu(),
                        y.detach().cpu(),
                        y_probas.squeeze(1),
                        image_ids,
                    )
                else:
                    eval_dict["vis"] = visualize_batch_multi_cls(
                        args.task,
                        10,
                        X.detach().cpu(),
                        y.detach().cpu(),
                        y_probas,
                        image_ids,
                        category_ids,
                    )

        ## accumulate metrics
        accuracy, precision, recall, f1 = calculate_metrics(tp, tn, fp, fn).values()
        cad_results = finalize_cad_results(
            cad_results,
            BOX_TYPES,
        )
        pr_results = finalize_pr_results(pr_results)

        ## update eval_dict
        for metric_name in metrics:
            eval_dict[metric_name] = eval(metric_name)

        update_results_dict(
            eval_dict,
            cad_results,
            pr_results,
        )

        eval_results = cad_results

        # calculate auc
        if args.roc:
            y_true_all = np.concatenate(y_true_all)
            y_probas_all = np.concatenate(y_probas_all)
            fpr, tpr, _ = roc_curve(y_true_all, y_probas_all)
            eval_dict["auc"] = auc(fpr, tpr)
            # log roc
            # required by wandb roc_curve: shape: (*y_true.shape, n_classes)
            y_probas_all = np.stack([1 - y_probas_all, y_probas_all], axis=1)
            wandb.log(
                {
                    "roc": wandb.plot.roc_curve(
                        y_true_all,
                        y_probas_all,
                        labels=["no person", "person"],
                        classes_to_plot=[1],
                    )
                }
            )

    return eval_dict, eval_results


def calculate_metrics_batch(
    batch,
    y,
    y_probas,
    bboxes,
    category_ids,
    min_distance,
    threshold_abs,
    matcher,
    BOX_TYPES,
    cad_results,
    channel_to_id,
    pr_results,
):
    for batch_idx in range(len(batch["images"])):
        mask_all = y[batch_idx]  # (n_classes, h, w)
        y_proba_all = y_probas[batch_idx]  # (n_classes, h, w)
        gt_bbox_all = bboxes[batch_idx]  # (n_instances, 4)
        gt_cls_all = category_ids[batch_idx]  # (n_instances,)

        for ch in range(y_proba_all.size(0)):
            ind_gt = gt_cls_all == channel_to_id[ch]
            if ind_gt.sum() > 0:  # if there are gt instances in this class
                mask = mask_all[ch]  # (h, w)
                y_proba = y_proba_all[ch]  # (h, w)
                gt_bbox = gt_bbox_all[ind_gt]
                # gt_cls = gt_cls_all[ind_gt]

                coordinates_pred = get_centroid_coordinates_from_map(
                    y_proba, min_distance, threshold_abs
                )

                coordinates_gt = get_centroid_coordinates_from_bbox(gt_bbox)

                if coordinates_pred.size(0) == 0:
                    # init
                    box_type_to_center_alignment_discerpancy = {
                        box_type: 0.0 for box_type in BOX_TYPE_TO_SIZE
                    }
                    box_type_to_cardinality_penalty = {
                        box_type: 0.0 for box_type in BOX_TYPE_TO_SIZE
                    }
                    box_type_to_matched_discrepancy = {
                        box_type: 0.0 for box_type in BOX_TYPE_TO_SIZE
                    }
                    box_type_to_n_instances = {
                        box_type: 0 for box_type in BOX_TYPE_TO_SIZE
                    }
                    # update CAD metrics
                    box_type_to_center_alignment_discerpancy["all"] = len(
                        coordinates_gt
                    )
                    box_type_to_cardinality_penalty["all"] = len(coordinates_gt)
                    box_type_to_n_instances["all"] = len(coordinates_gt)

                    # update PR metrics
                    pr_results[ch]["FN"] += len(coordinates_gt)

                else:
                    logits_gt = (
                        mask.detach()
                        .cpu()[coordinates_gt[:, 0], coordinates_gt[:, 1]]
                        .view(-1, 1)
                    )
                    logits_pred = y_proba.cpu()[
                        coordinates_pred[:, 0], coordinates_pred[:, 1]
                    ].view(-1, 1)

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

            # else: # do nothing if no gt instances in this class

    return cad_results, pr_results


def finalize_cad_results(
    cad_results,
    BOX_TYPES=BOX_TYPES,
    do_print=False,
):
    for ch in cad_results:
        ## preds are extra predicted centerpoints,
        # update preds for n
        if cad_results[ch]["n"]["all"] is None:
            cad_results[ch]["n"]["preds"] = None
        else:
            s = (
                cad_results[ch]["n"]["small"]
                if cad_results[ch]["n"]["small"] is not None
                else 0
            )
            m = (
                cad_results[ch]["n"]["medium"]
                if cad_results[ch]["n"]["medium"] is not None
                else 0
            )
            l = (
                cad_results[ch]["n"]["large"]
                if cad_results[ch]["n"]["large"] is not None
                else 0
            )
            cad_results[ch]["n"]["preds"] = cad_results[ch]["n"]["all"] - s - m - l

        # to avoid division by zero
        cad_results[ch]["n"] = {
            k: 1e-12 if v == 0 else v for k, v in cad_results[ch]["n"].items()
        }

        # update preds for cad
        if cad_results[ch]["cad"]["all"] is None:
            cad_results[ch]["cad"]["preds"] = None
        else:
            s = (
                cad_results[ch]["cad"]["small"]
                if cad_results[ch]["cad"]["small"] is not None
                else 0
            )
            m = (
                cad_results[ch]["cad"]["medium"]
                if cad_results[ch]["cad"]["medium"] is not None
                else 0
            )
            l = (
                cad_results[ch]["cad"]["large"]
                if cad_results[ch]["cad"]["large"] is not None
                else 0
            )
            if cad_results[ch]["n"]["preds"] is None:
                cad_results[ch]["cad"]["preds"] = None
            else:
                cad_results[ch]["cad"]["preds"] = (
                    cad_results[ch]["cad"]["all"] - s - m - l
                ) / cad_results[ch]["n"]["preds"]

        for box_type in BOX_TYPES:
            n_instances = cad_results[ch]["n"][box_type]
            if n_instances is not None:
                cad_results[ch]["cp"][box_type] = (
                    cad_results[ch]["cp"][box_type] / n_instances
                )
                cad_results[ch]["cad"][box_type] = (
                    cad_results[ch]["cad"][box_type] / n_instances
                )
                cad_results[ch]["md"][box_type] = (
                    cad_results[ch]["md"][box_type] / n_instances
                )
            # else: cad_results[ch]'s values are all None by init, so no need to update
    if do_print:
        pprint.pprint(cad_results)

    return cad_results


def finalize_pr_results(pr_results):
    for ch in pr_results:
        all_T = pr_results[ch]["TP"] + pr_results[ch]["FN"]
        all_P = pr_results[ch]["TP"] + pr_results[ch]["FP"]
        pr_results[ch]["P"] = pr_results[ch]["TP"] / all_P if all_P != 0 else 0
        pr_results[ch]["R"] = pr_results[ch]["TP"] / all_T if all_T != 0 else 0
        pr_results[ch]["F1"] = (
            (2 * pr_results[ch]["P"] * pr_results[ch]["R"])
            / (pr_results[ch]["P"] + pr_results[ch]["R"])
            if pr_results[ch]["P"] + pr_results[ch]["R"] != 0
            else 0
        )
    return pr_results


def update_results_dict(
    results_dict,
    cad_results,
    pr_results,
):

    # Now refactor the key assignments
    assign_weighted_mean(results_dict, "CAD", "cad", "all", cad_results)
    assign_weighted_mean(results_dict, "MD", "md", "all", cad_results)
    assign_weighted_mean(results_dict, "CP", "cp", "all", cad_results)

    assign_weighted_mean(results_dict, "CAD_small", "cad", "small", cad_results)
    assign_weighted_mean(results_dict, "MD_small", "md", "small", cad_results)
    assign_weighted_mean(results_dict, "CP_small", "cp", "small", cad_results)

    assign_weighted_mean(results_dict, "CAD_medium", "cad", "medium", cad_results)
    assign_weighted_mean(results_dict, "MD_medium", "md", "medium", cad_results)
    assign_weighted_mean(results_dict, "CP_medium", "cp", "medium", cad_results)

    assign_weighted_mean(results_dict, "CAD_large", "cad", "large", cad_results)
    assign_weighted_mean(results_dict, "MD_large", "md", "large", cad_results)
    assign_weighted_mean(results_dict, "CP_large", "cp", "large", cad_results)

    assign_weighted_mean(results_dict, "CAD_preds", "cad", "preds", cad_results)

    results_dict["TP"] = sum(pr_results[ch]["TP"] for ch in pr_results)
    results_dict["FP"] = sum(pr_results[ch]["FP"] for ch in pr_results)
    results_dict["FN"] = sum(pr_results[ch]["FN"] for ch in pr_results)
    results_dict["P"] = sum(pr_results[ch]["P"] for ch in pr_results) / len(pr_results)
    results_dict["R"] = sum(pr_results[ch]["R"] for ch in pr_results) / len(pr_results)
    results_dict["F1"] = sum(pr_results[ch]["F1"] for ch in pr_results) / len(
        pr_results
    )


# Helper function to calculate weighted mean
def weighted_mean(values, weights):
    if not values or not weights or np.sum(weights) == 0:
        return None
    return np.sum(np.array(values) * np.array(weights)) / np.sum(weights)


# Helper function to assign weighted mean to results_dict
def assign_weighted_mean(results_dict, key, subkey, box_type, cad_results):
    results_dict[key] = weighted_mean(
        [
            cad_results[ch][subkey][box_type]
            for ch in cad_results
            if cad_results[ch][subkey][box_type] is not None
        ],
        [
            cad_results[ch]["n"][box_type]
            for ch in cad_results
            if cad_results[ch][subkey][box_type] is not None
        ],
    )
