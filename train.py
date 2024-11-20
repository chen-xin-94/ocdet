"""
Usage:
QT_QPA_PLATFORM=offscreen CUDA_VISIBLE_DEVICES=0 python train.py --task=cmap_c80 -p ocdet --config=ocdet-n --vis --wandb
"""

import os
import argparse
from pathlib import Path
import random
import warnings
from tqdm.auto import tqdm
from collections import defaultdict
import yaml
import json
import pprint
import wandb
import numpy as np
import torch
import gc
import matplotlib.pyplot as plt
from build import build, fix_compatibility
from engine import train_val_one_epoch, evaluation
from utils.utils import dotdict, AverageMeter

tasks = [
    "cmap",
    "cmap_fpn",
    "cmap_fpn_backbones",
    "cmap_simple",
    "cmap_c8",
    "cmap_c80",
    "cmap_centerness",
    "cmap_seg",
    "gmap_kp",
    "gmap_kp_c80",
    "emap",
]

# pixel-wise metrics
train_metrics = ["accuracy", "precision", "recall", "f1"]
val_metrics = ["accuracy", "precision", "recall", "f1"]


def get_args():

    parser = argparse.ArgumentParser(description="Person Centroid Detection")

    parser.add_argument(
        "--task",
        default="cmap",
        type=str,
        choices=tasks,
        help="Type of ground truth",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="configuration file *.yaml",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument("-p", "--project_name", type=str, default=None)
    parser.add_argument("--save_folder", type=str, default="torch_models")
    parser.add_argument(
        "-j",
        "--workers",
        default=8,  # OPTION
        type=int,
        help="number of data loading workers",
    )
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
    parser.add_argument(
        "--data_folder",
        metavar="DIR",
        help="path to dataset",
        default="/mnt/ssd2/xin/data/coco/",  # OPTION
    )
    parser.add_argument(
        "--print_every",
        default=1,
        type=int,
        help="print every n batches",
    )
    parser.add_argument(
        "--metric_every",
        default=250,
        type=int,
        help="run metrics every n batches",
    )
    parser.add_argument(
        "--val_every",
        default=250,
        type=int,
        help="run validation every n batches",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    # visualization
    parser.add_argument(
        "--vis",
        action="store_true",
        help="visualize the training process and log to wandb",
    )
    # logging
    parser.add_argument("--roc", action="store_true", help="log ROC curve to wandb")
    # wandb
    parser.add_argument("--wandb", action="store_true", help="use wandb")
    parser.add_argument("--sweep", action="store_true", help="use wandb sweep")

    return parser.parse_args()


def main():
    args = get_args()
    if args.wandb:
        args.wandb_mode = "online"
    else:
        args.wandb_mode = "disabled"

    project_name = (
        "pcd_" + args.task if args.project_name is None else args.project_name
    )

    with wandb.init(project=project_name, mode=args.wandb_mode) as run:

        # load config file
        if args.sweep:  # use wandb sweep specified on website
            # update project name
            project_name = run.project
            config = fix_compatibility(wandb.config)
        else:  # otherwise use the local config file
            config_path = os.path.join("configs", args.task, args.config + ".yaml")
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            config = fix_compatibility(config)
            wandb.config.update(config)
        # NOTE: sweepable arguments are just meant to provide a interface for wandb sweep and is not supposed to be directly used in this script
        # Namespace to dict
        args = vars(args)

        ## update args
        # with config
        args.update(config)
        args = dotdict(args)

        if args.n_classes == 8 and "_256" not in args.train_ann_file:
            args.val_every = 500
            args.metric_every = 500
        elif args.n_classes == 80 and "_256" not in args.train_ann_file:
            args.val_every = 900
            args.metric_every = 900
        elif args.n_classes == 1 and args.batch_size == 32:
            args.val_every = 1000
            args.metric_every = 1000
        # else: use default values 250

        if args.workers > args.batch_size:
            args.workers = args.batch_size
            print(
                "set num_workers to",
                args.workers,
            )

        if args.seed is not None:
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True
            warnings.warn(
                "You have chosen to seed training. "
                "This will turn on the CUDNN deterministic setting, "
                "which can slow down your training considerably! "
                "You may see unexpected behavior when restarting "
                "from checkpoints."
            )
        if not torch.cuda.is_available():
            print("using CPU, this will be slow")
        if args.gpu != -1:
            device = torch.device("cuda:" + str(args.gpu))
            # print("Use GPU:{} for training".format(args.gpu)) # better to use CUDA_VISIBLE_DEVICES to specify the GPU
        else:
            device = torch.device("cpu")
            print("Use CPU for training")

        pprint.pprint(args)

        ## build model, dataloader, optimizer, lr_scheduler, criterion, etc.
        (
            model,
            train_dataloader,
            val_dataloader,
            optimizer,
            lr_scheduler,
            warmup_scheduler,
            criterion,
            name,
        ) = build(args, device)
        if not args.wandb_mode == "disabled":  # otherwise AttributeError
            run.name = name

        ## init for best model
        save_folder = os.path.join(args.save_folder, project_name, name)
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        results = defaultdict(list)
        best_metric = -np.inf
        best_model = None
        best_epoch = -1

        for epoch in tqdm(range(1, args.epochs + 1)):  # 1-indexed

            print(f"Epoch {epoch}/{args.epochs}")

            train_results, val_results, global_step = train_val_one_epoch(
                model,
                train_dataloader,
                val_dataloader,
                criterion,
                optimizer,
                lr_scheduler,
                warmup_scheduler,
                train_metrics,
                val_metrics,
                epoch,
                device,
                args,
            )

            ## At epoch end

            ## log weights and gradients of the model
            histograms = {}
            for tag, value in model.named_parameters():
                if value.grad is None:
                    continue
                tag = tag.replace("/", ".")
                if not (torch.isinf(value) | torch.isnan(value)).any():
                    histograms["Weights/" + tag] = wandb.Histogram(value.data.cpu())
                if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                    histograms["Gradients/" + tag] = wandb.Histogram(
                        value.grad.data.cpu()
                    )
            run.log(histograms, step=global_step)
            # change prefix to train or val,
            # use .avg for train_results since they are tested on each batch
            # NOTE: this is an underestimate of the current training set metrics,
            # sicne the model is getting better as the training goes on,
            # and previous batches are tested on old versions of the model
            train_results = {f"train/{k}": v.avg for k, v in train_results.items()}
            # use .val for val_results since they are tested on the whole val dataset
            val_results = {
                f"val/{k}": v.val if isinstance(v, AverageMeter) else v
                for k, v in val_results.items()
            }

            # append to final results
            for k, v in train_results.items():
                results[k].append(v)
            for k, v in val_results.items():
                results[k].append(v)

            # save best model according to selected metric
            # currently only use a weighted score of f1 and cae
            # NOTE: metric value should be a value, the bigger the better
            cur_metric = -results["val/CAD"][-1]
            if cur_metric > best_metric:
                best_metric = cur_metric
                best_model = model
                best_epoch = epoch
            torch.save(
                best_model.state_dict(), os.path.join(save_folder, "best_model.pt")
            )

            # garbage collection
            torch.cuda.empty_cache()
            gc.collect()
            print("-------------------")

            # early stopping
            if epoch - best_epoch > (args.epochs // 6):
                print("Early stopping at epoch", epoch)
                break

        ## After training

        # save final best model to the current wandb run
        wandb.save(os.path.join(save_folder, "best_model.pt"), base_path=save_folder)

        # save the results as a json file locally and to wandb
        resutls_save_path = os.path.join(save_folder, "results_per_epoch.json")
        with open(resutls_save_path, "w") as f:
            json.dump(results, f)
        wandb.save(resutls_save_path, base_path=save_folder)

        # save the config file
        config_save_path = os.path.join(save_folder, "config.yaml")
        with open(config_save_path, "w") as f:
            yaml.dump(dict(args), f)

        # run evaluation and log to wandb as summary
        eval_dict, eval_results = evaluation(
            model=best_model,
            dataloader=val_dataloader,
            metrics=val_metrics,
            device=device,
            args=args,
        )

        # save eval_results as a json file locally and to wandb
        eval_resutls_path = os.path.join(save_folder, "eval_results.json")
        with open(eval_resutls_path, "w") as f:
            json.dump(eval_results, f)
        wandb.save(eval_resutls_path, base_path=save_folder)
        print(f"Saved evaluation results to {eval_resutls_path}")

        if args.vis:
            figs = eval_dict.pop("vis", None)
            run.log(
                {
                    "Visualization of the best model on val dataset": [
                        wandb.Image(fig) for fig in figs
                    ]
                }
            )
            # close the figures
            for fig in figs:
                plt.close(fig)

        for k, v in eval_dict.items():
            run.summary[k] = v

        pprint.pprint(eval_dict)


if __name__ == "__main__":
    main()
