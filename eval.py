"""
Usage:
QT_QPA_PLATFORM=offscreen CUDA_VISIBLE_DEVICES=0 python eval.py --task cmap_c80 --project ocdet --trained weights/ocdet-n.pt --config ocdet-n --vis --wandb
"""

import os
import argparse
import yaml
import pprint
import torch
import wandb
from build import build, fix_compatibility
from engine import evaluation
from utils.utils import dotdict
from pathlib import Path
import json

COCO_FOLDER = "/mnt/ssd2/xxx/data/coco/"


def get_args():
    parser = argparse.ArgumentParser(description="Person Centroid Detection Evaluation")

    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
    parser.add_argument(
        "--data_folder", metavar="DIR", help="path to dataset", default=None
    )
    parser.add_argument(
        "--task",
        default="cmap",
        type=str,
        help="Type of ground truth",
    )
    parser.add_argument("-p", "--project_name", type=str, default=None)
    parser.add_argument("--save_folder", type=str, default="torch_models")
    parser.add_argument(
        "-c",
        "--config",
        help="configuration file *.yaml",
        type=str,
        required=False,
        default="base",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=8,
        type=int,
        help="number of data loading workers (default: 8)",
    )
    parser.add_argument(
        "--trained", type=str, default=None, help="path to trained model"
    )
    parser.add_argument(
        "--vis", action="store_true", help="visualize the evaluation results"
    )
    parser.add_argument("--roc", action="store_true", help="log ROC curve to wandb")
    parser.add_argument("--wandb", action="store_true", help="use wandb")

    # since some old config
    parser.add_argument("--min_distance", default=3, type=int)
    parser.add_argument("--threshold_abs", default=0.5, type=float)

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

        if (
            args.config == "config"
        ):  # use config downloaded from wandb and stored in root
            config_path = args.config + ".yaml"
        else:
            config_path = os.path.join("configs", args.task, args.config + ".yaml")

        with open(config_path, "r") as f:
            config = dotdict(yaml.safe_load(f))
        print(f"Loaded config from {config_path}")

        # if config downloaded from wandb, remove the wandb specific keys
        if "_wandb" in config:  # always have _wandb
            config.pop("_wandb")
            if "wandb_version" in config:  # sometimes have wandb_version
                config.pop("wandb_version")
            for k, v in config.items():
                config[k] = v["value"]
        config = fix_compatibility(config)
        wandb.config.update(config)
        args = vars(args)
        args.update(config)
        args = dotdict(args)
        if args.data_folder is None:  # use default COCO dataset
            args.data_folder = COCO_FOLDER
        pprint.pprint(args)

        device = (
            torch.device("cuda:" + str(args.gpu))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        model, _, val_dataloader, _, _, _, _, name = build(args, device, is_train=False)

        if not args.wandb_mode == "disabled":  # otherwise AttributeError
            run.name = "eval_" + name

        if args.trained:
            state_dict = torch.load(args.trained, map_location=device)
            # chagne the key name for old models
            state_dict = {k.replace("fpn.", "neck."): v for k, v in state_dict.items()}
            state_dict = {
                k.replace("fpn_head.", "head."): v for k, v in state_dict.items()
            }
            model.load_state_dict(state_dict)

        eval_dict, eval_results = evaluation(
            model=model,
            dataloader=val_dataloader,
            metrics=["accuracy", "precision", "recall", "f1"],
            device=device,
            args=args,
        )

        # save eval_results as a json file locally and to wandb
        save_folder = os.path.join(args.save_folder, project_name, name)
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        eval_resutls_path = os.path.join(save_folder, "eval_results.json")
        with open(eval_resutls_path, "w") as f:
            json.dump(eval_results, f)
        print(f"Saved evaluation results to {eval_resutls_path}")
        wandb.save(eval_resutls_path, base_path=save_folder)

        if args.vis:
            figs = eval_dict.pop("vis", None)
            run.log(
                {
                    "Visualization of the model on val dataset": [
                        wandb.Image(fig) for fig in figs
                    ]
                }
            )

        for k, v in eval_dict.items():
            wandb.run.summary[k] = v

        pprint.pprint(eval_dict)


if __name__ == "__main__":
    main()
