import numpy as np
import torch
from torchvision.datasets import CocoDetection
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import cv2
import time
from PIL import Image
import os
import sys
from tqdm import tqdm

# Get the directory where the current script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory of the current dir (i.e. root dir of this repo) to the system path
sys.path.append(os.path.dirname(current_dir))
from utils.coco import COCO_ID_TO_CHANNEL

default_transforms_list = [
    transforms.Resize(
        (224, 224)
    ),  # if strictly adhering to the standard imagenet training, should be first resize to 256, then randomcrop to 224, for the simplicity, we just resize to 224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]


class CocoCentroidMap(CocoDetection):
    """
    (deprecated)
    COCO dataset but only returns a map showing whether there's a object centroid or not as the target
    """

    def __init__(
        self,
        root,
        annFile,
        stride=8,
        resized_image_size=(224, 224),  # NOTE: Only consider square images for now
        transform=transforms.Compose(
            default_transforms_list[1:]
        ),  # [1:], since this class handles resize internally
        target_transform=None,  # since the target is a map, and the resize is handled internally, we don't need to transform it
    ):
        super(CocoCentroidMap, self).__init__(
            root, annFile, transform, target_transform
        )
        self.stride = stride
        self.resized_image_size = resized_image_size

        assert f"r{self.resized_image_size[0]}" in annFile  # assume square images
        assert f"s{self.stride}" in annFile

        # insert resize to transform
        self.transform = transforms.Compose(
            [transforms.Resize(self.resized_image_size), self.transform]
            if self.transform is not None
            else [transforms.Resize(self.resized_image_size)]
        )

    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        anns = self._load_target(id)

        target = torch.zeros(
            self.resized_image_size[0] // self.stride,
            self.resized_image_size[1] // self.stride,
        )

        for ann in anns:
            x, y = ann["coord_map"]
            if x is not None and y is not None:  # for non-person or crowd person
                target[y, x] = 1

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


class CocoCentroidHalf(CocoDetection):
    """
    deprecated
    COCO dataset but only returns a 1x2 map showing whether there's a object centroid or not as the target
    """

    def __init__(
        self,
        root,
        annFile,
        transform=transforms.Compose(default_transforms_list),
        target_transform=None,  # since the target is a map, and the resize is handled internally, we don't need to transform it
    ):
        super(CocoCentroidHalf, self).__init__(
            root, annFile, transform, target_transform
        )

    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        anns = self._load_target(id)

        l, r = 0, 0
        for ann in anns:
            l |= ann["half"][0]
            r |= ann["half"][1]

        target = torch.tensor([l, r]).float()

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target


class CocoMap(CocoDetection):
    """COCO dataset but only returns a 1x2 map showing whether there's a object centroid or not as the target"""

    def __init__(
        self,
        root,
        annFile,
        transform=None,  # expect albumentations transform
        mask_folder_name="cmaps",
        target_category_ids=[1],
        id_to_channel=COCO_ID_TO_CHANNEL,
        deep_supervision=False,
        use_cache=True,
    ):
        super(CocoMap, self).__init__(root, annFile, transform)
        self.mask_folder_name = mask_folder_name
        self.target_category_ids = target_category_ids
        self.deep_supervision = deep_supervision
        self.root = Path(root)
        self.coco_folder, self.split = self.root.parent, self.root.name
        if use_cache:
            self.load_image = self._load_cache_image
        else:
            self.load_image = self._load_image
        self.id_to_channel = id_to_channel

    def check_cache(self):
        # check if first 10 and last 10 images are cached
        for id in self.ids:
            file_name = self.coco.loadImgs(id)[0]["file_name"]
            npy_path = (self.root / file_name).with_suffix(".npy")
            if not npy_path.exists():
                return False
        return True

    def cache_images(self):
        for id in tqdm(self.ids):
            file_name = self.coco.loadImgs(id)[0]["file_name"]
            image = np.array(Image.open(self.root / file_name).convert("RGB"))
            npy_path = (self.root / file_name).with_suffix(".npy")
            np.save(npy_path, image)

    def _load_image(self, id):
        file_name = self.coco.loadImgs(id)[0]["file_name"]
        image = np.array(Image.open(self.root / file_name).convert("RGB"))
        image = image.astype("float32") / 255  # uint8 -> float32
        return image

    def _load_cache_image(self, id):
        file_name = self.coco.loadImgs(id)[0]["file_name"].replace(".jpg", ".npy")
        return np.load(self.root / file_name).astype("float32") / 255

    def load_image_and_masks(self, id: int, category_ids: list):
        image = self.load_image(id)

        file_name = self.coco.loadImgs(id)[0]["file_name"]
        category_set = sorted(set(self.target_category_ids) & set(category_ids))
        channels = [self.id_to_channel[category_id] for category_id in category_set]
        mask_pruned = np.zeros((*image.shape[:2], len(category_set)), dtype=np.uint8)
        for i, category_id in enumerate(category_set):
            mask_folder = (
                self.coco_folder / self.mask_folder_name / str(category_id) / self.split
            )
            mask_path = (mask_folder / file_name).with_suffix(".png")
            if mask_path.exists():
                mask_pruned[:, :, i] = np.array(Image.open(mask_path).convert("L"))
            else:
                raise FileNotFoundError(f"Mask file not found: {mask_path}")
        mask_pruned = mask_pruned.astype("float32") / 255

        if self.deep_supervision:
            mask_r80_shape = (80, 80)
            mask_r80_pruned = np.zeros(
                (*mask_r80_shape, len(category_set)),
                dtype=np.uint8,
            )
            for i, category_id in enumerate(category_set):
                mask_folder = (
                    self.coco_folder
                    / (self.mask_folder_name + "_r80")
                    / str(category_id)
                    / self.split
                )
                mask_path = (mask_folder / file_name).with_suffix(".png")
                if mask_path.exists():
                    mask_r80_pruned[:, :, i] = np.array(
                        Image.open(mask_path).convert("L")
                    )
            mask_r80_pruned = mask_r80_pruned.astype("float32") / 255
        else:
            mask_r80_pruned = None

        return image, mask_pruned, mask_r80_pruned, channels

    def _load_bboxes_and_category_ids(self, id: int):
        target = self._load_target(id)
        bboxes = []
        category_ids = []
        for ann in target:
            # no crowd
            if ann["iscrowd"]:
                continue
            # only target category ids
            category_id = ann["category_id"]
            if ann["category_id"] not in self.target_category_ids:
                continue
            # only consider bbox with width and height >= 1
            bbox = ann["bbox"]
            _, _, bw, bh = map(round, bbox)
            # continue if the bounding box is too small
            if bw < 1 or bh < 1:
                continue
            bboxes.append(bbox)
            category_ids.append(category_id)
        return bboxes, category_ids

    def __getitem__(self, index):
        """
        NOTE:
        self.transform is expected to be albumentations transform including at least resize and normalize,
        where max_pixel_value for Normalize is set to 1, so the image should be in [0, 1] range before applying the transform which is done in _load_image_and_masks
        Returns:
            image: np.array (H, W, 3)
            mask: np.array (H, W, len(target_category_ids))
            bboxes: list of list of floats
            category_ids: list of ints
        """
        id = self.ids[index]

        # ## time begin
        # start = time.perf_counter()
        # ## time end

        bboxes, category_ids = self._load_bboxes_and_category_ids(id)

        # ## time bboxes and category ids
        # time_bboxes_and_category_ids = time.perf_counter() - start
        # ## time end

        if (
            len(category_ids) == 0
        ):  # handle case of empty annotations, e.g. for id 25593
            image = self.load_image(id)
            mask_pruned = np.empty(image.shape[:2] + (1,), dtype=np.float32)
            mask_r80_pruned = np.empty((80, 80) + (1,), dtype=np.float32)
        else:
            image, mask_pruned, mask_r80_pruned, channels = self.load_image_and_masks(
                id, category_ids
            )
        # ## time begin
        # time_image_and_masks = (
        #     time.perf_counter() - start - time_bboxes_and_category_ids
        # )
        # ## time end

        if self.transform is not None:
            augmented = self.transform(
                image=image, mask=mask_pruned, bboxes=bboxes, category_ids=category_ids
            )
            image, mask_pruned, bboxes, category_ids = (
                augmented["image"],
                augmented["mask"],
                augmented["bboxes"],
                augmented["category_ids"],
            )
        # NOTE: totensor is done by ToTensorV2(transpose_mask=True) in transform
        # expand mask_pruned by restoring zero channels
        mask = torch.zeros(len(self.target_category_ids), *mask_pruned.shape[1:])
        if len(category_ids) > 0:
            mask[channels, :, :] = mask_pruned

        # ## time begin
        # time_transform = (
        #     time.perf_counter()
        #     - start
        #     - time_image_and_masks
        #     - time_bboxes_and_category_ids
        # )
        # ## time end

        if self.deep_supervision:
            mask_r80 = torch.zeros(
                len(self.target_category_ids), *mask_r80_pruned.shape[:2]
            )
            if len(category_ids) > 0:
                # get rid of scale related augs.
                no_aug = ["RandomScale", "Resize", "RandomResizedCrop"]
                augmented["replay"]["transforms"] = [
                    t
                    for t in augmented["replay"]["transforms"]
                    if (t["__class_fullname__"] not in no_aug)
                ]
                # get rid of bbox_params
                augmented["replay"]["bbox_params"] = None
                # adjust
                augmented_partial = A.ReplayCompose.replay(
                    augmented["replay"],
                    image=np.zeros((80, 80, 3)),  # dummy image
                    mask=mask_r80_pruned,
                )
                mask_r80_pruned = augmented_partial["mask"]
                # expand mask_r80_pruned by restoring zero channels

                mask_r80[channels, :, :] = mask_r80_pruned
        else:
            mask_r80 = None

        # ## time begin
        # time_transform_r80 = (
        #     time.perf_counter()
        #     - start
        #     - time_image_and_masks
        #     - time_bboxes_and_category_ids
        #     - time_transform
        # )
        # print(f"Time image and masks: {time_image_and_masks}")
        # print(f"Time bboxes and category ids: {time_bboxes_and_category_ids}")
        # print(f"Time transform: {time_transform}")
        # print(f"Time transform r80: {time_transform_r80}")
        # ## time end

        return {
            "image_id": id,
            "image": image,
            "mask": mask,
            "mask_r80": mask_r80,
            "bboxes": bboxes,
            "category_ids": category_ids,
        }

    @staticmethod
    def build_transforms(resized_image_size):
        """
        Args:
            resized_image_size: tuple of ints (H, W)
        """

        transform_train = A.ReplayCompose(
            [
                # Randomly rotate the image within [-30°, 30°]
                A.Rotate(limit=(-30, 30), p=0.25),
                # Randomly scale the image within [0.75, 1.25]
                A.RandomScale(scale_limit=0.25, p=0.25),
                # Randomly flip horizontally with probability 50%
                A.HorizontalFlip(p=0.5),
                # Randomly flip vertically with probability 50%
                A.VerticalFlip(p=0.25),
                # A.Resize(*args.resized_image_size),
                A.RandomResizedCrop(size=resized_image_size, scale=(0.8, 1), p=1.0),
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
        transform_val = A.ReplayCompose(
            [
                A.Resize(*resized_image_size),
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
        return transform_train, transform_val

    @staticmethod
    def collate_fn(batch):
        """
        Input:
            a list of batch_size tuples,
            each of which in the format (id, image, mask, bbox, category_id) if the dataset is CocoMap
                id: int
                image: np.array (H, W, 3)
                mask: np.array (H, W, 1)
                bboxes: list of list of floats
                category_ids: list of ints

        Returns:
            A dictionary with the following keys:
            - 'image_ids': list of ints
            - 'images': torch.Tensor of shape (batch_size, 3, H, W)
            - 'masks': torch.Tensor of shape (batch_size, 1, H, W)
            - 'bboxes': list of torch.Tensor of shape (n, 4)
            - 'category_ids': list of ints
        """

        image_ids = []
        images = []
        masks = []
        masks_r80 = []
        # slightly abuse the plural here, in
        bboxes = []
        category_ids = []

        for item in batch:
            image_ids.append(torch.tensor(item["image_id"]))
            images.append(
                item["image"]
            )  # already a tensor, done by albumentations toTensorV2
            masks.append(
                item["mask"]
            )  # already a tensor, done by albumentations toTensorV2
            masks_r80.append(item["mask_r80"])
            bboxes.append(torch.tensor(item["bboxes"]))
            category_ids.append(torch.tensor(item["category_ids"]))

        images = torch.stack(images, dim=0)
        masks = torch.stack(masks, dim=0)
        if masks_r80[0] is not None:
            masks_r80 = torch.stack(masks_r80, dim=0)
            # otherwise, keep as a list of None
        # keep the bboxes, category_idsas list of tensors for now
        # another option is to stack them into a single tensor,
        # and then use batch_idx to keep track of the which image they belong to
        # (as did in yolov8)

        return {
            "image_ids": image_ids,
            "images": images,
            "masks": masks,
            "masks_r80": masks_r80,
            "bboxes": bboxes,
            "category_ids": category_ids,
        }


class CocoCMap(CocoMap):
    def __init__(self, *args, mask_folder_name="cmaps", **kwargs):
        super(CocoCMap, self).__init__(
            *args, mask_folder_name=mask_folder_name, **kwargs
        )


class CocoEMap(CocoMap):
    def __init__(self, *args, mask_folder_name="emaps", **kwargs):
        super(CocoEMap, self).__init__(
            *args, mask_folder_name=mask_folder_name, **kwargs
        )


class CocoGMapCoord(CocoMap):
    def __init__(self, *args, mask_folder_name="gmaps_kp", **kwargs):
        super(CocoGMapCoord, self).__init__(
            *args, mask_folder_name=mask_folder_name, **kwargs
        )
