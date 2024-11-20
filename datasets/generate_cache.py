# from datasets.dataset import CocoCMap, CocoGMapCoord
from build import build_dataset, build_dataloader
from utils.utils import dotdict
import numpy as np

device = "cuda:3"

args = {
    "task": "cmap",
    # "task": "gmap_kp",
    # "data_folder": "/mnt/ssd2/xin/data/coco",
    "data_folder": "/mnt/ssd3/xin/data/coco",
    "train_ann_file": "instances_train2017.json",
    "val_ann_file": "instances_val2017.json",
    "backbone": "mobilenetv4_conv_small.e2400_r224_in1k",
    "n_classes": 1,
    "num_outs": 4,
    "resized_image_size": (320, 320),
    # "batch_size": 64,
    "batch_size": 128,
    "workers": 4,
    "deep_supervision": True,
}
args = dotdict(args)
train_dataset = build_dataset(args, is_train=True)
train_dataset.cache_images()
