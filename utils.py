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
from scipy.ndimage import distance_transform_edt, binary_erosion, binary_closing, binary_dilation, generate_binary_structure
from dataset import SliceDataset

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

# Binarize the volume based on a threshold
def binarize_volume(, threshold=0.5):
    return (volume > threshold).astype(np.uint8)
    
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

            # Binarize the volumes to ensure they are in 0-1 format
            pred_volume_binary = binarize_volume(pred_volume)
            gt_volume_binary = binarize_volume(gt_volume)
            # Apply morphological closing to the predicted segmentation
            structuring_element = generate_binary_structure(3, 1)  # 3x3x3 structure
            pred_volume_closed = binary_closing(pred_volume_binary, structure=structuring_element)
            pred_volume = pred_volume_binary
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



def compute_hausdorff_distance(pred_mask: torch.Tensor, 
                               gt_mask: torch.Tensor, 
                               voxel_spacing: tuple = (1.0, 1.0, 1.0),
                               percentile: int = 95) -> dict:
    """
    Computes both standard and 95th percentile Hausdorff distances between two binary volumes.

    Args:
        pred_mask (torch.Tensor): Binary volume of predictions (D x W x H)
        gt_mask (torch.Tensor): Binary volume of ground truth (D x W x H)
        voxel_spacing (tuple): Physical spacing between voxels (depth, height, width)
        percentile (int): Percentile for HD calculation (default 95 for HD95)

    Returns:
        dict: Contains both standard HD and HD95 measurements
    """
    
    # Convert tensors to numpy arrays
    pred_mask_np = pred_mask.cpu().numpy().astype(np.bool_)
    gt_mask_np = gt_mask.cpu().numpy().astype(np.bool_)

    # If both masks are empty or full, return 0 distance
    if (np.sum(pred_mask_np) == 0 and np.sum(gt_mask_np) == 0) or \
       (np.sum(pred_mask_np) == pred_mask_np.size and np.sum(gt_mask_np) == gt_mask_np.size):
        return {'HD': 0.0, 'HD95': 0.0}

    # Compute distance transforms with voxel spacing
    dt_gt = distance_transform_edt(~gt_mask_np, sampling=voxel_spacing)
    dt_pred = distance_transform_edt(~pred_mask_np, sampling=voxel_spacing)

    # Extract surface voxels using binary erosion
    surface_pred = pred_mask_np ^ binary_erosion(pred_mask_np)
    surface_gt = gt_mask_np ^ binary_erosion(gt_mask_np)

    # Get distances for both directions
    distances_pred_to_gt = dt_gt[surface_pred]
    distances_gt_to_pred = dt_pred[surface_gt]

    # Handle empty surfaces
    if distances_pred_to_gt.size == 0 and distances_gt_to_pred.size == 0:
        return {'HD': 0.0, 'HD95': 0.0}

    # Compute standard HD
    if distances_pred_to_gt.size == 0:
        max_distance_pred_to_gt = 0
    else:
        max_distance_pred_to_gt = distances_pred_to_gt.max()

    if distances_gt_to_pred.size == 0:
        max_distance_gt_to_pred = 0
    else:
        max_distance_gt_to_pred = distances_gt_to_pred.max()

    hd = max(max_distance_pred_to_gt, max_distance_gt_to_pred)

    # Compute HD95
    all_distances = np.concatenate([distances_pred_to_gt, distances_gt_to_pred])
    hd95 = np.percentile(all_distances, percentile) if all_distances.size > 0 else 0.0

    return {'HD': float(hd), 'HD95': float(hd95)}

def calculate_3d_hausdorff(patient_slices, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Computes the 3D Hausdorff distance for each patient and class.

    Args:
        patient_slices (dict): Dictionary containing prediction and ground truth slices for each patient.
        voxel_spacing (tuple): Physical spacing between voxels (depth, height, width)

    Returns:
        dict: Per-patient Hausdorff distances (both HD and HD95).
    """
    per_patient_hausdorff = {}

    patient_progress = tqdm(patient_slices.items(), desc="Calculating 3D Hausdorff", unit="patient")

    for patient_id, slices in patient_progress:
        if len(slices['pred_slices']) > 1:
            # Stack slices into volumes
            pred_volume = torch.stack(slices['pred_slices'], dim=0)  # Shape: [D, K, W, H]
            gt_volume = torch.stack(slices['gt_slices'], dim=0)      # Shape: [D, K, W, H]

            # Binarize the volumes to ensure they are in 0-1 format
            pred_volume_binary = binarize_volume(pred_volume)
            gt_volume_binary = binarize_volume(gt_volume)
            # Apply morphological closing to the predicted segmentation
            structuring_element = generate_binary_structure(3, 1)  # 3x3x3 structure
            pred_volume_closed = binary_closing(pred_volume_binary, structure=structuring_element)
            pred_volume = pred_volume_binary

            K = pred_volume.shape[1]  # Number of classes
            hausdorff_scores = {'HD': [], 'HD95': []}


            for k in range(K):
                pred_k = pred_volume[:, k]  # Shape: [D, W, H]
                gt_k = gt_volume[:, k]      # Shape: [D, W, H]

                # Skip if both masks are empty
                if torch.sum(pred_k) == 0 and torch.sum(gt_k) == 0:
                    hausdorff_scores['HD'].append(0.0)
                    hausdorff_scores['HD95'].append(0.0)
                    continue

                # Compute both HD and HD95
                distances = compute_hausdorff_distance(pred_k, gt_k, voxel_spacing)
                hausdorff_scores['HD'].append(distances['HD'])
                hausdorff_scores['HD95'].append(distances['HD95'])

            per_patient_hausdorff[patient_id] = hausdorff_scores

        # Update progress bar
        patient_progress.set_description(f"Calculating 3D Hausdorff - Patient: {patient_id}")

    return per_patient_hausdorff



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

        # Binarize the volumes to ensure they are in 0-1 format
        pred_volume_binary = binarize_volume(pred_volume)
        gt_volume_binary = binarize_volume(gt_volume)
        # Apply morphological closing to the predicted segmentation
        structuring_element = generate_binary_structure(3, 1)  # 3x3x3 structure
        pred_volume_closed = binary_closing(pred_volume_binary, structure=structuring_element)
        pred_volume = pred_volume_binary
        
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

def get_boundary_points(seg):
    """
    Extracts boundary points from a binary segmentation mask.
    
    Parameters:
    seg (Tensor): Binary segmentation mask, shape [1, H, W]
    
    Returns:
    Tensor: Tensor of boundary points, shape [N, 2] where N is the number of boundary points.
    """
    # Using morphological operations to find the boundary
    kernel = torch.ones((3, 3), dtype=torch.float32).to(seg.device)
    seg_dilated = torch.nn.functional.conv2d(seg.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=1)
    boundary = seg_dilated - seg.unsqueeze(0).unsqueeze(0)
    boundary = boundary.squeeze(0).squeeze(0)
    
    # Get the coordinates of the boundary points
    boundary_points = torch.nonzero(boundary).float()
    
    return boundary_points

def compute_pairwise_distances(points1, points2):
    """
    Computes pairwise Euclidean distances between two sets of points.
    
    Parameters:
    points1 (Tensor): Tensor of shape [N, 2]
    points2 (Tensor): Tensor of shape [M, 2]
    
    Returns:
    Tensor: Pairwise distances of shape [N]
    """
    # Expand points and compute Euclidean distance
    dists = torch.cdist(points1, points2, p=2)
    
    # Return the minimum distance for each point in points1 to points2
    return dists.min(dim=1)[0]


def count_class_occurrences(dataset: SliceDataset, num_classes: int) -> list[int]:
    class_occurrences = [0] * num_classes
    total_samples = len(dataset)

    print(f">> Counting class occurrences across {total_samples} samples...")

    for idx in tqdm(range(total_samples)):
        sample = dataset[idx]
        gt = sample['gts']  # Shape: [num_classes, H, W]
        
        # Check which classes are present in this sample
        for class_idx in range(num_classes):
            if gt[class_idx].any():
                class_occurrences[class_idx] += 1

    print(f">> Class occurrences: {class_occurrences}")
    return class_occurrences

def calculate_class_weights(frequencies: list[float], alpha: float = 1.0) -> list[float]:
    epsilon = 1e-8  # To prevent division by zero
    weights = [alpha / (freq + epsilon) if freq > 0 else 0.0 for freq in frequencies]
    # Normalize weights to have mean = 1
    mean_weight = sum(weights) / len(weights)
    normalized_weights = [w / mean_weight for w in weights]
    print(f">> Calculated Class Weights: {normalized_weights}")
    return normalized_weights    