import argparse
import warnings
from typing import Any
from pathlib import Path
from pprint import pprint
from operator import itemgetter
from shutil import copytree, rmtree

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import SliceDataset
from ShallowNet import shallowCNN
from ENet import ENet
from Unet import UNet
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   save_images,
                   average_hausdorff_distance,
                   intersection,
                   union)

from losses import (CrossEntropy,
                   BalancedCrossEntropy,
                   DiceLoss,
                   FocalLoss,
                   CombinedLoss)

from multiprocessing import Pool
from tqdm import tqdm

from sklearn.model_selection import KFold

def get_kfold_splits(dataset, n_splits=5):
    """
    Returns k-fold cross-validation splits.
    
    Args:
    - dataset: Your entire dataset (list or numpy array of sample identifiers).
    - n_splits: Number of folds (default=5).
    
    Returns:
    - List of (train_indices, val_indices) for each fold.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)  # Random state for reproducibility
    splits = []

    for train_index, val_index in kf.split(dataset):
        splits.append((train_index, val_index))
    
    return splits

def main():
    set_seed(42)
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--dataset', default='TOY2', choices=datasets_params.keys())
    parser.add_argument('--mode', default='full', choices=['partial', 'full', 'balanced', 'dice', 'focal', 'combined'])
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the results (predictions and weights).")
    parser.add_argument('--model', default='UNet', choices=list(models.keys()),
                        help="Choose the model architecture")
    parser.add_argument('--k_folds', default=5, type=int, help="Number of folds for cross-validation")
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logic around epochs and logging easily.")

    args = parser.parse_args()

    # Load dataset and prepare K-fold splits
    root_dir = Path("data") / args.dataset

    # Assuming you have a function to load the entire dataset (without splits yet)
    dataset = SliceDataset('full', root_dir, img_transform=img_transform, gt_transform=gt_transform, debug=args.debug)
    
    # Get K-fold splits
    splits = get_kfold_splits(dataset, args.k_folds)
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"Starting fold {fold_idx + 1}/{args.k_folds}...")
        
        # Create train/val subsets for this fold
        train_set = torch.utils.data.Subset(dataset, train_idx)
        val_set = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_set, batch_size=datasets_params[args.dataset]['B'], shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_set, batch_size=datasets_params[args.dataset]['B'], shuffle=False, num_workers=args.num_workers)
        
        # Update destination folder for this fold
        fold_dest = args.dest / f"fold_{fold_idx + 1}"
        fold_dest.mkdir(parents=True, exist_ok=True)

        # Call the training function for this fold
        runTraining(args, train_loader, val_loader, fold_dest)
    
    print("Cross-validation completed!")

if __name__ == '__main__':
    main()
