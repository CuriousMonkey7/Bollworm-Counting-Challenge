import numpy as np

# Pytorch
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

# Sklearn
from sklearn.utils.class_weight import compute_class_weight

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Custom Import
from utils.dataset import FilterDataset
from configs import filter_config


CFG = filter_config.get_config()


def get_transforms():
    data_transforms = {
        "train": A.Compose(
            [
                A.RandomBrightnessContrast(
                    always_apply=False,
                    p=0.5,
                    brightness_limit=(-0.2, 0.57),
                    contrast_limit=(-0.2, 0.44),
                    brightness_by_max=True,
                ),
                A.Defocus(
                    always_apply=False, p=0.5, radius=(1, 70), alias_blur=(0.1, 7)
                ),
                A.Flip(always_apply=False, p=0.7),
                # A.PixelDropout(
                #     always_apply=False,
                #     p=0.7,
                #     dropout_prob=0.3,
                #     per_channel=0,
                #     drop_value=(0, 0, 0),
                #     mask_drop_value=None,
                # ),
                A.CoarseDropout(
                    always_apply=False,
                    p=0.7,
                    max_holes=50,
                    max_height=16,
                    max_width=16,
                    min_holes=20,
                    min_height=8,
                    min_width=8,
                    fill_value=(0, 0, 0),
                    mask_fill_value=None,
                ),
                A.GaussNoise(always_apply=False, p=0.5, var_limit=(83.88, 136.51), per_channel=True, mean=0.0),
                A.Rotate(p=0.5),
                A.Resize(*CFG.img_size),
                ToTensorV2(),
                
            ]
        ),
        "val": A.Compose(
            [
                #             T.Normalize(CFG.img_mean, CFG.img_std),
                A.Resize(*CFG.img_size),
                ToTensorV2(),
            ]
        ),
    }
    return data_transforms


def get_loaders(df,phase="train",test=False,preload=False):

    data_transforms = get_transforms()
    img_dir = "data/preproc/np"
    train_dataset = FilterDataset(
        df, img_dir, transforms=data_transforms[phase],test=test,preload=preload
    )

    loader = DataLoader(
        train_dataset,
        batch_size=CFG.train_bs ,
        num_workers=CFG.num_workers,
        shuffle=phase=="train",
        pin_memory=True,
        drop_last=False,
        
    )

    return loader
