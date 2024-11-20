import torch
import torch.nn as nn
import pytorch_warmup as warmup
from torch.utils.data import DataLoader
import os
from functools import partial

from datasets.dataset import CocoMap
from models.unet import UNet, MobileNetV4Unet
from models.fpn import OCDFPN, OCDPAFPN
from models.simple_baseline import MobilenetV4SimpleBaseline
from utils.losses import (
    mse_loss_with_logits,
    weighted_mse_loss_with_logits,
    focal_loss,
    balanced_continuous_focal_loss,
    quality_focal_loss,
    FocalDiceLoss,
    CFocalCDiceLoss,
    CFocalDiceLoss,
)
from utils.coco import COCO_CHANNEL_TO_ID, COCO_ID_TO_CHANNEL


def build_dataloader(
    dataset,
    batch_size,
    shuffle=True,
    num_workers=8,
    prefetch_factor=2,
    pin_memory=True,
    drop_last=True,
    collate_fn=None,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


def build_dataset(args, is_train=True):
    root = os.path.join(args.data_folder, "train2017/" if is_train else "val2017/")
    ann_file = os.path.join(
        args.data_folder,
        "annotations",
        args.train_ann_file if is_train else args.val_ann_file,
    )

    transform_train, transform_val = CocoMap.build_transforms(
        tuple(args.resized_image_size)
    )

    transform = transform_train if is_train else transform_val

    if "cmap" in args.task:
        mask_folder_name = "cmaps"
        if "cmap_centerness" in args.task and not (args.eta == 0.5 and args.phi == 0.5):
            mask_folder_name = f"cmaps_{args.eta}_{args.phi}"
    elif args.task == "cmap_seg":
        if not args.n_classes == 1:  # assume person class
            raise NotImplementedError("Only support person class for cmap_seg")
        mask_folder_name = "cmaps_seg"
    elif "gmap_kp" in args.task:
        mask_folder_name = "gmaps_kp"
    elif "emap" in args.task:
        mask_folder_name = "emaps"
    else:
        raise ValueError("Invalid task type")

    return CocoMap(
        root=root,
        annFile=ann_file,
        transform=transform,
        mask_folder_name=mask_folder_name,
        target_category_ids=args.target_category_ids,
        id_to_channel=args.id_to_channel,
        deep_supervision=args.deep_supervision,
    )


def build_model(args, device):
    if args.arch == "unet":
        model = UNet(
            n_classes=args.n_classes,
            mode=args.mode,
            width_scale=args.width_scale,
        ).to(device)
        if args.pretrained and args.mode == "convtranspose" and args.width_scale == 1.0:
            state_dict = torch.load(os.path.join("pretrained_models", args.pretrained))
            del state_dict["outc.conv.weight"]
            del state_dict["outc.conv.bias"]
            model.load_state_dict(state_dict, strict=False)
        else:
            print("No pretrained model found for this configuration")
    elif args.arch == "mobilenetv4_unet":
        model = MobileNetV4Unet(
            backbone=args.backbone,
            mode=args.mode,
            n_classes=args.n_classes,
            pretrained=args.pretrained,
            width_scale=args.width_scale,
        ).to(device)
    elif args.arch == "fpn":
        model = OCDFPN(
            backbone=args.backbone,
            n_classes=args.n_classes,
            num_outs=args.num_outs,
            out_channel=args.out_channel,
            dropout_ratio=args.dropout_ratio,
        ).to(device)
    elif args.arch == "pafpn":
        model = OCDPAFPN(
            backbone=args.backbone,
            n_classes=args.n_classes,
            num_outs=args.num_outs,
            out_channel=args.out_channel,
            dropout_ratio=args.dropout_ratio,
        ).to(device)
    elif args.arch == "simple":
        model = MobilenetV4SimpleBaseline(
            backbone=args.backbone, n_classes=args.n_classes, mode=args.mode
        ).to(device)
    else:
        raise NotImplementedError(f"Invalid arch: {args.arch}")

    # resume
    if args.resume:
        # NOTE: currenly only support loading the model weight, not the optimizer and scheduler
        if ".pt" in args.resume:
            weight_path = args.resume
        else:  # assume only the folder is provided
            weight_path = os.path.join(
                args.save_folder, args.project_name, args.resume, "best_model.pt"
            )
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"state_dict loaded from {args.resume}")
    return model


def build_optimizer(args, model):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    return optimizer


def build_scheduler(args, optimizer, train_dataloader):
    if args.lr_scheduler.lower() == "step":
        orig_milestones = [8, 11]
        scale_milestones = max((args.epochs // 12), 1)
        milestones = [i * scale_milestones for i in orig_milestones]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1
        )
        warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    elif args.lr_scheduler.lower() == "cosine":
        max_steps = len(train_dataloader) * args.epochs
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_steps
        )
        warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    else:
        raise NotImplementedError(f"Invalid lr_scheduler: {args.lr_scheduler}")
    return lr_scheduler, warmup_scheduler


def build_criterion(args, device):
    loss = args.loss.lower().strip()
    if loss == "bce":
        pos_weight = args.pos_weight
        if args.get("pos_weight", None) is not None:
            pos_weight = torch.tensor([pos_weight]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    elif loss == "mse":
        criterion = mse_loss_with_logits
    elif loss == "weighted_mse":
        criterion = partial(weighted_mse_loss_with_logits, alpha=args.alpha)
    elif loss == "focal":
        criterion = partial(focal_loss, alpha=args.alpha, gamma=args.gamma)
    elif loss == "cfocal":
        criterion = partial(
            balanced_continuous_focal_loss, alpha=args.alpha, gamma=args.gamma
        )
    elif loss == "focal_dice":
        criterion = FocalDiceLoss(args.ratio_focal, args.alpha, args.gamma).to(device)
    elif loss == "cfocal_dice":
        criterion = CFocalDiceLoss(args.ratio_focal, args.alpha, args.gamma).to(device)
    elif loss == "cfocal_cdice":
        criterion = CFocalCDiceLoss(args.ratio_focal, args.alpha, args.gamma).to(device)
    elif loss.lower() == "qfl":
        criterion = partial(quality_focal_loss, gamma=args.gamma)
    else:
        raise NotImplementedError("Invalid loss function")
    return criterion


def build_name(args):
    name = f"{args.arch}"
    if args.arch == "unet":
        name += f"_{args.mode}_width{args.width_scale}"
    elif args.arch == "mobilenetv4_unet":
        name += f"_{args.backbone.split('.')[0]}_{args.mode}_w{args.width_scale}"
    elif args.arch == "fpn":
        name += f"_{args.backbone.split('.')[0]}_o{args.num_outs}_c{args.out_channel}_d{args.dropout_ratio}"
    elif args.arch == "simple":
        name += f"_{args.backbone.split('.')[0]}_{args.mode}"
    elif args.arch == "pafpn":
        name += f"_{args.backbone.split('.')[0]}_o{args.num_outs}_c{args.out_channel}_d{args.dropout_ratio}"
    elif args.arch == "lite_hrnet":
        pass

    if args.pretrained and args.mode == "convtranspose" and args.width_scale == 1.0:
        name += "_pretrained"

    name += f"_e{args.epochs}_lr{float(args.lr):.5f}_wd{float(args.weight_decay):.4f}_r{args.resized_image_size[0]}_{args.loss}"
    if args.get("loss", None) == "bce" and args.pos_weight:
        name += f"_pw{args.pos_weight:.2f}"
    elif "focal" in args.loss:
        name += f"_alpha{args.alpha:.3f}_gamma{args.gamma:.2f}"

    if args.task == "gmap":
        name += f"_sm{args.sigma_multiplier:.2f}"
    if "cmap_centerness" in args.task:
        name += f"_eta{args.eta}_phi{args.phi}"
    if "person" in args.train_ann_file:
        name += "_coco-person"
    else:
        name += "_coco"

    return name


def build(args, device, is_train=True):

    model = build_model(args, device)
    criterion = build_criterion(args, device)
    name = build_name(args)

    dataset_val = build_dataset(args, is_train=False)
    val_dataloader = build_dataloader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=(
            dataset_val.collate_fn if hasattr(dataset_val, "collate_fn") else None
        ),
    )

    if is_train:
        dataset_train = build_dataset(args, is_train=True)
        train_dataloader = build_dataloader(
            dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=(
                dataset_train.collate_fn
                if hasattr(dataset_train, "collate_fn")
                else None
            ),
        )
        optimizer = build_optimizer(args, model)
        lr_scheduler, warmup_scheduler = build_scheduler(
            args, optimizer, train_dataloader
        )
        # adjust metric_every and val_every
        args.metric_every = min(args.metric_every, len(train_dataloader) // 2)
        args.val_every = min(args.val_every, len(train_dataloader) // 2)
    else:
        train_dataloader = None
        optimizer = None
        lr_scheduler = None
        warmup_scheduler = None

    return (
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        lr_scheduler,
        warmup_scheduler,
        criterion,
        name,
    )


def fix_compatibility(config):
    """
    fix compatibility issues for old configs
    """
    # update pretrained
    if (
        config.get("arch") == "unet"
        and config.get("mode") != "convtranspose"
        or config.get("width_scale") != 1.0
    ):
        config["pretrained"] = None
    # some old config files do not have lr_scheduler
    if config.get("lr_scheduler") is None:
        config["lr_scheduler"] = "step"
    # some old config files do not have num_outs, dropout_ratio and out_channel options
    if config.get("arch") == "fpn":
        if config.get("num_outs") is None:
            config["num_outs"] = 4
        if config.get("dropout_ratio") is None:
            config["dropout_ratio"] = 0.1
        if config.get("out_channel") is None:
            config["out_channel"] = 32
        if config.get("fpn_type") is None:
            config["fpn_type"] = "mm"
    # some old config files use "scale_focal" instead of "ratio_focal"
    if config.get("scale_focal") is not None:
        config["ratio_focal"] = config["scale_focal"]
    # old config uses "-" to connect focal and dice loss
    config["loss"] = config.get("loss").replace("-", "_")
    if config.get("min_distance") is None:
        config["min_distance"] = 3
    if config.get("threshold_abs") is None:
        config["threshold_abs"] = 0.5
    if config.get("deep_supervision") is None:
        config["deep_supervision"] = False
    # overwrite n_classes if target_category_ids is provided
    if config.get("target_category_ids") is not None:
        # if int, change to list
        if isinstance(config["target_category_ids"], int):
            config["target_category_ids"] = [config["target_category_ids"]]
        config["n_classes"] = len(config["target_category_ids"])
    else:
        # take the first n_classes from COCO
        if config["n_classes"]:
            config["target_category_ids"] = list(sorted(COCO_ID_TO_CHANNEL.keys()))[
                : config["n_classes"]
            ]
        else:
            raise ValueError("Need to specify target_category_ids or n_classes")

    # channel_to_id
    if config["n_classes"] == 80:
        config["channel_to_id"] = COCO_CHANNEL_TO_ID
        config["id_to_channel"] = COCO_ID_TO_CHANNEL
    else:
        config["channel_to_id"] = {
            i: id for i, id in enumerate(sorted(config["target_category_ids"]))
        }
        config["id_to_channel"] = {v: k for k, v in config["channel_to_id"].items()}

    return config
