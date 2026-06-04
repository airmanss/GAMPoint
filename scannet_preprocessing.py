"""
ScanNet20 / ScanNet200 / ScanNet Data Efficient Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

try:
    from scipy.spatial import cKDTree
except ImportError:
    import warnings
    warnings.warn("Please install scipy for parsing curvature")


def compute_curvature(coords, k_list=(16, 32)):
    """
    PCA-based curvature over k-NN neighborhoods (Weinmann et al. ISPRS 2015).
    Returns (N, 4*len(k_list)): curvature, linearity, planarity, sphericity per scale.
    """
    tree = cKDTree(coords)
    k_max = max(k_list)
    _, all_idx = tree.query(coords, k=k_max + 1)
    all_idx = all_idx[:, 1:]  # drop self

    scale_feats = []
    for k in k_list:
        idx = all_idx[:, :k]
        N = len(coords)
        eigs = np.zeros((N, 3), dtype=np.float32)
        for start in range(0, N, 50000):
            end = min(start + 50000, N)
            neigh = coords[idx[start:end]]
            centered = neigh - neigh.mean(axis=1, keepdims=True)
            cov = np.einsum("bki,bkj->bij", centered, centered) / (k - 1)
            for i, c in enumerate(cov):
                vals = np.linalg.eigvalsh(c)
                eigs[start + i] = vals[::-1]  # descending: lam0 >= lam1 >= lam2

        lam0, lam1, lam2 = eigs[:, 0], eigs[:, 1], eigs[:, 2]
        eps = 1e-8
        curvature  = lam2 / (lam0 + lam1 + lam2 + eps)
        linearity  = (lam0 - lam1) / (lam0 + eps)
        planarity  = (lam1 - lam2) / (lam0 + eps)
        sphericity = lam2           / (lam0 + eps)
        scale_feats.append(
            np.stack([curvature, linearity, planarity, sphericity], axis=1)
        )

    return np.concatenate(scale_feats, axis=1).astype(np.float32)

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .defaults import DefaultDataset
from .transform import Compose, TRANSFORMS
from .preprocessing.scannet.meta_data.scannet200_constants import (
    VALID_CLASS_IDS_20,
    VALID_CLASS_IDS_200,
)


@DATASETS.register_module()
class ScanNetDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment20",
        "instance",
        "curvature",
    ]
    class2id = np.array(VALID_CLASS_IDS_20)

    def __init__(
        self,
        lr_file=None,
        la_file=None,
        **kwargs,
    ):
        self.lr = np.loadtxt(lr_file, dtype=str) if lr_file is not None else None
        self.la = torch.load(la_file) if la_file is not None else None
        super().__init__(**kwargs)

    def get_data_list(self):
        if self.lr is None:
            data_list = super().get_data_list()
        else:
            data_list = [
                os.path.join(self.data_root, "train", name) for name in self.lr
            ]
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)

        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name
        data_dict["coord"] = data_dict["coord"].astype(np.float32)
        data_dict["color"] = data_dict["color"].astype(np.float32)
        data_dict["normal"] = data_dict["normal"].astype(np.float32)

        # Load pre-computed curvature if available, otherwise compute on-the-fly
        if "curvature" in data_dict.keys():
            data_dict["curvature"] = data_dict["curvature"].astype(np.float32)
        else:
            data_dict["curvature"] = compute_curvature(data_dict["coord"])

        if "segment20" in data_dict.keys():
            data_dict["segment"] = (
                data_dict.pop("segment20").reshape([-1]).astype(np.int32)
            )
        elif "segment200" in data_dict.keys():
            data_dict["segment"] = (
                data_dict.pop("segment200").reshape([-1]).astype(np.int32)
            )
        else:
            data_dict["segment"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        if "instance" in data_dict.keys():
            data_dict["instance"] = (
                data_dict.pop("instance").reshape([-1]).astype(np.int32)
            )
        else:
            data_dict["instance"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )
        if self.la:
            sampled_index = self.la[self.get_data_name(idx)]
            mask = np.ones_like(data_dict["segment"], dtype=bool)
            mask[sampled_index] = False
            data_dict["segment"][mask] = self.ignore_index
            data_dict["sampled_index"] = sampled_index
        return data_dict


@DATASETS.register_module()
class ScanNet200Dataset(ScanNetDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment200",
        "instance",
        "curvature",
    ]
    class2id = np.array(VALID_CLASS_IDS_200)
