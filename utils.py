#!/usr/bin/env python3

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from pathlib import Path
from functools import partial
from multiprocessing import Pool
from contextlib import AbstractContextManager
from typing import Callable, Iterable, List, Set, Tuple, TypeVar, cast
from scipy.spatial.distance import directed_hausdorff

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch import Tensor, einsum

tqdm_ = partial(tqdm, dynamic_ncols=True,
                leave=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]')


class Dcm(AbstractContextManager):
    # Dummy Context manager
    def __exit__(self, *args, **kwargs):
        pass


# Functools
A = TypeVar("A")
B = TypeVar("B")


def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return list(map(fn, iter))


def mmap_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return Pool().map(fn, iter)


def starmmap_(fn: Callable[[Tuple[A]], B], iter: Iterable[Tuple[A]]) -> List[B]:
    return Pool().starmap(fn, iter)


# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def eq(a: Tensor, b) -> bool:
    return torch.eq(a, b).all()


def simplex(t: Tensor, axis=1) -> bool:
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


def class2one_hot(seg: Tensor, K: int) -> Tensor:
    # Breaking change but otherwise can't deal with both 2d and 3d
    # if len(seg.shape) == 3:  # Only w, h, d, used by the dataloader
    #     return class2one_hot(seg.unsqueeze(dim=0), K)[0]

    assert sset(seg, list(range(K))), (uniq(seg), K)

    b, *img_shape = seg.shape

    device = seg.device
    res = torch.zeros((b, K, *img_shape), dtype=torch.int32, device=device).scatter_(1, seg[:, None, ...], 1)

    assert res.shape == (b, K, *img_shape)
    assert one_hot(res)

    return res


def probs2class(probs: Tensor) -> Tensor:
    b, _, *img_shape = probs.shape
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, *img_shape)

    return res


def probs2one_hot(probs: Tensor) -> Tensor:
    _, K, *_ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), K)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


# Save the raw predictions
def save_images(segs: Tensor, names: Iterable[str], root: Path) -> None:
        for seg, name in zip(segs, names):
                save_path = (root / name).with_suffix(".png")
                save_path.parent.mkdir(parents=True, exist_ok=True)

                if len(seg.shape) == 2:
                        Image.fromarray(seg.detach().cpu().numpy().astype(np.uint8)).save(save_path)
                elif len(seg.shape) == 3:
                        np.save(str(save_path), seg.detach().cpu().numpy())
                else:
                        raise ValueError(seg.shape)


# Metrics
def meta_dice(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8) -> Tensor:
    assert label.shape == pred.shape
    assert one_hot(label)
    assert one_hot(pred)

    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
    sum_sizes: Tensor = (einsum(sum_str, [label]) + einsum(sum_str, [pred])).type(torch.float32)

    dices: Tensor = (2 * inter_size + smooth) / (sum_sizes + smooth)

    return dices


dice_coef = partial(meta_dice, "bk...->bk")
dice_batch = partial(meta_dice, "bk...->k")  # used for 3d dice


def intersection(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])

    res = a & b
    assert sset(res, [0, 1])

    return res


def union(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])

    res = a | b
    assert sset(res, [0, 1])

    return res

def average_hausdorff_distance(pred, gt):
    """Compute the average Hausdorff distance between the predicted and ground truth masks."""
    
    # Convert PyTorch tensors to NumPy arrays
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()

    # Ensure that we are working with boolean arrays
    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)

    # Compute Hausdorff distance using scipy
    hausdorff_distance_1 = directed_hausdorff(pred, gt)[0]
    hausdorff_distance_2 = directed_hausdorff(gt, pred)[0]

    return max(hausdorff_distance_1, hausdorff_distance_2)

def torch2D_Hausdorff_distance(x,y): # Input be like (Batch,width,height)
    x = x.float()
    y = y.float()
    distance_matrix = torch.cdist(x,y,p=2) # p=2 means Euclidean Distance
    
    value1 = distance_matrix.min(2)[0].max(1, keepdim=True)[0]
    value2 = distance_matrix.min(1)[0].max(1, keepdim=True)[0]
    
    value = torch.cat((value1, value2), dim=1)
    
    return value.max(1)[0]

def collect_patient_slices(patient_slices, img_paths, pred_seg, gt, B):
    """
    Collects and stores prediction and ground truth slices for each patient.
    
    Args:
    - patient_slices: Dictionary to store slices for each patient.
    - img_paths: List of image paths to extract patient IDs.
    - pred_seg: Predicted segmentation slices.
    - gt: Ground truth slices.
    - B: Batch size.
    """
    for b in range(B):  # Iterate over the batch
        patient_id = img_paths[b]  # Extract patient ID from the image path (e.g., 'Patient_03')

        # Initialize an entry for a new patient in the dictionary if not already present
        if patient_id not in patient_slices:
            patient_slices[patient_id] = {'pred_slices': [], 'gt_slices': []}

        # Append the current prediction and ground truth slice to the corresponding patient
        patient_slices[patient_id]['pred_slices'].append(pred_seg[b].cpu())  # Storing the prediction slice
        patient_slices[patient_id]['gt_slices'].append(gt[b].cpu())  # Storing the ground truth slice

    return patient_slices

def calculate_3d_dice(patient_slices):
    """
    Computes the 3D Dice score for each patient and logs the results.

    Args:
    - patient_slices: Dictionary containing prediction and ground truth slices for each patient.
    - log_3d_dice: List to store 3D Dice scores.

    Returns:
    - 3d_dices: Numpy array containing 3D Dice scores for each patient.
    """
    num_patients = len(patient_slices)
    dices_3d = np.zeros(num_patients)  # Initialize array to store 3D Dice scores for each patient

    for idx, (patient_id, slices) in enumerate(patient_slices.items()):
        if len(slices['pred_slices']) > 1:  # Ensure we have more than one slice to create a 3D volume
            # Convert the list of slices into a 3D tensor
            pred_volume = torch.stack(slices['pred_slices'], dim=-1)
            gt_volume = torch.stack(slices['gt_slices'], dim=-1)

            # Calculate 3D Dice
            dice_3d = dice_coef(gt_volume, pred_volume).mean().item()

            # Log 3D Dice for the patient
            dices_3d[idx] = dice_3d

    return dices_3d
