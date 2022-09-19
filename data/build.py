# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com

and

@author:  davide zambrano
@contact: d.zambrano@sportradar.com


"""

import torch.nn
from torch.utils import data

from yacs.config import CfgNode

from .datasets.viewds import VIEWDS, CHALLENGE
from .datasets.viewds import SVIEWDS, GenerateSViewDS
from .transforms import build_transforms

DATASETS = {
    "viewds": VIEWDS,
    "sviewds": SVIEWDS,
    "challenge": CHALLENGE,
}


def build_dataset(
    cfg: CfgNode, transforms: torch.nn.Module, is_train: bool = True
) -> data.Dataset:
    """Create dataset.

    Args:
        cfg (CfgNode): config file
        transforms (torch.nn.Module): Applies transformation to the input data.
        is_train (bool, optional): Train or eval the model. Defaults to True.

    Returns:
        datasets (torch.utils.data.Dataset): the dataset to be used.
    """

    kwargs = {
        "root": "./",
        "train": is_train,
        "transform": transforms,
        "download": False,
    }
    if cfg.DATASETS.TEST == "challenge":
        return CHALLENGE("CHALLENGE", transform=transforms)
    if cfg.DATASETS.TRAIN == "sviewds":
        width, height = (
            cfg.INPUT.MULTIPLICATIVE_FACTOR * cfg.INPUT.GENERATED_VIEW_SIZE[0],
            cfg.INPUT.MULTIPLICATIVE_FACTOR * cfg.INPUT.GENERATED_VIEW_SIZE[1],
        )
        def_min, def_max = (
            cfg.INPUT.MULTIPLICATIVE_FACTOR * cfg.INPUT.GENERATED_DEF_PM[0],
            cfg.INPUT.MULTIPLICATIVE_FACTOR * cfg.INPUT.GENERATED_DEF_PM[1],
        )
        print(width, height, def_min, def_max)
        svds = GenerateSViewDS(
            output_shape=(
                width,
                height,
            ),
            def_min=def_min,
            def_max=def_max,
            is_train=is_train,
            train_on_full_dataset=cfg.DATASETS.TRAIN_ON_FULL_DATASET
        )
        SPLIT = {"val": svds.val, "test": svds.test}
        if is_train:
            kwargs = {
                "vds": svds.train,
                "transform": transforms,
            }
        else:
            kwargs = {
                "vds": SPLIT[cfg.DATASETS.EVAL_ON],
                "transform": transforms,
            }
        #if cfg.DATASETS.EVALUATION:
        kwargs["return_camera"] = True

    if cfg.DATASETS.TRAIN == "viewds":
        kwargs["num_elements"] = cfg.DATASETS.NUM_ELEMENTS
    datasets = DATASETS[cfg.DATASETS.TRAIN](**kwargs)
    return datasets


def make_data_loader(cfg: CfgNode, is_train: bool = True) -> data.DataLoader:
    """Create the data loader.

    Args:
        cfg (CfgNode): config file.
        is_train (bool, optional): _Train or eval the model. Defaults to True.

    Returns:
        data_loader (data.Dataset): torch Dataloader
    """
    if is_train:
        batch_size = cfg.SOLVER.IMS_PER_BATCH
        shuffle = True
    else:
        batch_size = cfg.TEST.IMS_PER_BATCH
        shuffle = False

    transforms = build_transforms(cfg, is_train)
    datasets = build_dataset(cfg, transforms=transforms, is_train=is_train)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return data_loader
