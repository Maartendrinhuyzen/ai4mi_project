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
    pred_seg_cpu = pred_seg.cpu()
    gt_cpu = gt.cpu()
    
    for b in range(B):
        patient_id = img_paths[b]

        if patient_id not in patient_slices:
            patient_slices[patient_id] = {'pred_slices': [], 'gt_slices': []}
    
        # Now we're just accessing slices from already-CPU tensors
        patient_slices[patient_id]['pred_slices'].append(pred_seg_cpu[b])
        patient_slices[patient_id]['gt_slices'].append(gt_cpu[b])

    return patient_slices

# import torch
# import numpy as np
# from scipy.spatial.distance import directed_hausdorff
# def calculate_3d_hausdorff(patient_slices):
#     """
#     Computes the 3D Hausdorff distance for each patient and logs the results.
#     Args:
#     - patient_slices: Dictionary containing prediction and ground truth slices for each patient.
#     Returns:
#     - hausdorff_distances: Numpy array containing 3D Hausdorff distances for each patient.
#     """
#     num_patients = len(patient_slices)
#     hausdorff_distances = np.zeros(num_patients)  # Initialize array to store 3D Hausdorff distances for each patient
#     for idx, (patient_id, slices) in enumerate(patient_slices.items()):
#         if len(slices['pred_slices']) > 1:  # Ensure we have more than one slice to create a 3D volume
#             # Convert the list of slices into a 3D tensor
#             pred_volume = torch.stack(slices['pred_slices'], dim=-1).cpu().numpy()  # Convert to numpy array
#             gt_volume = torch.stack(slices['gt_slices'], dim=-1).cpu().numpy()  # Convert to numpy array
#             # Get the coordinates of the boundary points (where the segmentation is positive)
#             pred_points = np.argwhere(pred_volume > 0)
#             gt_points = np.argwhere(gt_volume > 0)
#             # Calculate the directed Hausdorff distances
#             hausdorff_pred_to_gt = directed_hausdorff(pred_points, gt_points)[0]
#             hausdorff_gt_to_pred = directed_hausdorff(gt_points, pred_points)[0]
#             # Hausdorff distance is the maximum of the directed distances
#             hausdorff_distance = max(hausdorff_pred_to_gt, hausdorff_gt_to_pred)
#             # Log Hausdorff distance for the patient
#             hausdorff_distances[idx] = hausdorff_distance
#     return hausdorff_distances



def calculate_3d_dice(patient_slices):
    """
    Computes the 3D Dice score for each patient and class.

    Args:
    - patient_slices: Dictionary containing prediction and ground truth slices for each patient.

    Returns:
    - per_patient_dice: Dictionary where each key is a patient ID and the value is a list of 
      Dice scores, one for each class.
    """
    per_patient_dice = {}
    
    for patient_id, slices in patient_slices.items():
        if len(slices['pred_slices']) > 1:
            # Convert the list of slices into a 3D tensor
            pred_volume = torch.stack(slices['pred_slices'], dim=0)  # Shape: [D, K, W, H]
            gt_volume = torch.stack(slices['gt_slices'], dim=0)      # Shape: [D, K, W, H]
            
            # Calculate Dice for each class
            K = pred_volume.shape[1]  # Number of classes
            dice_scores = []
            
            for k in range(K):
                pred_k = pred_volume[:, k]  # Shape: [D, W, H]
                gt_k = gt_volume[:, k]      # Shape: [D, W, H]
                
                intersection = torch.sum(pred_k * gt_k).float()
                sum_pred_gt = torch.sum(pred_k).float() + torch.sum(gt_k).float()
                
                dice = (2.0 * intersection / (sum_pred_gt + 1e-8)).item()
                dice_scores.append(dice)
            
            per_patient_dice[patient_id] = dice_scores

    return per_patient_dice

# def calculate_directed_hausdorff(source_points: torch.Tensor, target_points: torch.Tensor, batch_size: int = 1000) -> float:
#     """
#     Calculates the directed Hausdorff distance from source points to target points.
    
#     Args:
#     - source_points: Points from which to calculate distances (N x D tensor)
#     - target_points: Points to which distances are calculated (M x D tensor)
#     - batch_size: Number of points to process at once
    
#     Returns:
#     - Maximum minimum distance from source to target points
#     """
#     max_min_distance = 0
    
#     for i in range(0, len(source_points), batch_size):
#         source_batch = source_points[i:i + batch_size]
#         # Initialize min_distances as a tensor with the same device as source_points
#         min_distances = torch.full((len(source_batch),), float('inf'), 
#                                  device=source_points.device)
        
#         for j in range(0, len(target_points), batch_size):
#             target_batch = target_points[j:j + batch_size]
#             distances = torch.cdist(source_batch, target_batch, p=2)
#             batch_min = distances.min(dim=1)[0]
#             min_distances = torch.minimum(min_distances, batch_min)
        
#         batch_max = min_distances.max().item()
#         max_min_distance = max(max_min_distance, batch_max)
    
#     return max_min_distance

# def calculate_3d_hausdorff(patient_slices):
#     """
#     Computes the 3D Hausdorff distance for each patient and class using PyTorch with batch processing.

#     Args:
#     - patient_slices: Dictionary containing prediction and ground truth slices for each patient.

#     Returns:
#     - per_patient_hausdorff: Dictionary where each key is a patient ID and the value is a list of 
#       Hausdorff distances, one for each class.
#     """
#     per_patient_hausdorff = {}
#     BATCH_SIZE = 1000000

#     for patient_id, slices in patient_slices.items():
#         if len(slices['pred_slices']) > 1:
#             pred_volume = torch.stack(slices['pred_slices'], dim=0)  # Shape: [D, K, W, H]
#             gt_volume = torch.stack(slices['gt_slices'], dim=0)      # Shape: [D, K, W, H]
            
#             K = pred_volume.shape[1]  # Number of classes
#             hausdorff_scores = []
            
#             for k in range(K):
#                 pred_k = pred_volume[:, k]  # Shape: [D, W, H]
#                 gt_k = gt_volume[:, k]      # Shape: [D, W, H]
                
#                 pred_points = torch.nonzero(pred_k).float()
#                 gt_points = torch.nonzero(gt_k).float()
                
#                 if len(pred_points) == 0 or len(gt_points) == 0:
#                     hausdorff_scores.append(0.0)
#                     continue
#                 print(len(pred_points), len(gt_points))
#                 # Calculate Hausdorff distance in both directions
#                 pred_to_gt = calculate_directed_hausdorff(pred_points, gt_points, BATCH_SIZE)
#                 gt_to_pred = calculate_directed_hausdorff(gt_points, pred_points, BATCH_SIZE)
#                 print("yes")
#                 # Hausdorff distance is the maximum of both directed distances
#                 hausdorff_distance = max(pred_to_gt, gt_to_pred)
#                 hausdorff_scores.append(hausdorff_distance)
            
#             per_patient_hausdorff[patient_id] = hausdorff_scores

#     return per_patient_hausdorff



def calculate_3d_iou(patient_slices):
    """
    Calculates the 3D Intersection over Union (IoU) for each patient and each class based on the 
    predicted and ground truth segmentation volumes.
    Args:
        patient_slices (dict): A dictionary where each key is a patient ID, and the value is a dictionary 
            containing the predicted and ground truth slices for that patient. The expected structure is:
    Returns:
        dict: A dictionary where each key is a patient ID and the value is a list of IoU values, one for 
        each class (excluding background, if applicable). The structure of the output is:
        {
            'PatientID': [iou_class_0, iou_class_1, ..., iou_class_K-1],
            ...
        }
    """
    per_patient_iou = {}
    for patient_id, slices_dict in patient_slices.items():
        pred_slices = slices_dict['pred_slices']
        gt_slices = slices_dict['gt_slices']

        # Stack slices to form 3D volumes
        pred_volume = torch.stack(pred_slices, dim=0)  # Shape: [D, K, W, H]
        gt_volume = torch.stack(gt_slices, dim=0)      # Shape: [D, K, W, H]

        # For each class, compute IoU
        K = pred_volume.shape[1]
        ious = []
        for k in range(K):
            pred_k = pred_volume[:, k].contiguous().view(-1)  # Flatten to 1D
            gt_k = gt_volume[:, k].contiguous().view(-1)      # Flatten to 1D

            intersection = torch.sum(pred_k * gt_k).float()
            union = torch.sum(pred_k).float() + torch.sum(gt_k).float() - intersection

            iou = (intersection / (union + 1e-8)).item()
            ious.append(iou)

        per_patient_iou[patient_id] = ious  # List of IoU per class

    return per_patient_iou    