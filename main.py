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
from UnetAttention import UNetAttention
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

datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the clases with C (often used for the number of Channel)
datasets_params: dict[str, dict[str, Any]] = {}
datasets_params["TOY2"] = {'K': 2, 'B': 2}
datasets_params["SEGTHOR"] = {'K': 5, 'B': 8}
datasets_params["SEGTHOR_transformed"] = {'K': 5, 'B': 8}


models = {
    "shallowCNN": shallowCNN,
    "ENet": ENet,
    "UNet": UNet,
    "UNetAttention": UNetAttention
}

def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int]:
    # Networks and scheduler
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    K: int = datasets_params[args.dataset]['K']
    net = models[args.model](1, K)
    net.init_weights()
    net.to(device)

    # Use the hyper parameter from arguments
    lr = args.hyper_parameter
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)   
    #optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

    # Dataset part
    B: int = datasets_params[args.dataset]['B']
    root_dir = Path("data") / args.dataset

    img_transform = transforms.Compose([
        lambda img: img.convert('L'),
        lambda img: np.array(img)[np.newaxis, ...],
        lambda nd: nd / 255,  # Normalize to [0, 1]
        lambda nd: torch.tensor(nd, dtype=torch.float32)
    ])

    gt_transform = transforms.Compose([
        lambda img: np.array(img)[...],
        # Mapping classes
        lambda nd: nd / (255 / (K - 1)) if K != 5 else nd / 63,  # max <= 1
        lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add batch dimension
        lambda t: class2one_hot(t, K=K),
        itemgetter(0)
    ])

    
    # Combine train and validation sets if 'full' option is selected
    full_set = SliceDataset('full',  # Loading full dataset
                            root_dir,
                            img_transform=img_transform,
                            gt_transform=gt_transform,
                            debug=args.debug)
    full_loader = DataLoader(full_set,
                                batch_size=B,
                                num_workers=args.num_workers,
                                shuffle=True)

    # Return only the full dataset loader
    return (net, optimizer, device, full_set, full_loader, K)

def runTraining(args):
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")
    net, optimizer, device, full_set, _, K = setup(args)

    best_dice = 0
    # Initialize k-fold cross-validation
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)

    # Split dataset into folds
    kfolds_dice = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(full_set)):
        print(f"Running fold {fold + 1}/{args.k_folds}")

        # Create subsets for training and validation using the indices from KFold
        train_subset = torch.utils.data.Subset(full_set, train_idx)
        val_subset = torch.utils.data.Subset(full_set, val_idx)

        # Create data loaders for this fold
        train_loader = DataLoader(train_subset, batch_size=args.num_workers, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=args.num_workers, shuffle=False)

        if args.mode == "full":
            loss_fn = CrossEntropy(idk=list(range(K)))  # Supervise both background and foreground
        elif args.mode == "balanced":
            train_dataset = train_loader.dataset
            class_occurrences = count_class_occurrences(train_dataset, K)
            class_weights = calculate_class_weights(class_occurrences)
            loss_fn = BalancedCrossEntropy(idk=list(range(K)), class_weights=class_weights)
        elif args.mode == "dice":
            loss_fn = DiceLoss(idk=list(range(K)), smooth=1.0)
        elif args.mode == "focal":
            train_dataset = train_loader.dataset
            class_occurrences = count_class_occurrences(train_dataset, K)
            class_weights = calculate_class_weights(class_occurrences)
            loss_fn = FocalLoss(idk=list(range(K)), alpha=class_weights, gamma=2.0, reduction='mean')
        elif args.mode == "combined":
            loss_fn = CombinedLoss(idk=list(range(K)), weight_ce=1.0, weight_dice=1.0)
        elif args.mode in ["partial"] and args.dataset in ['SEGTHOR', 'SEGTHOR_STUDENTS']:
            loss_fn = CrossEntropy(idk=[0, 1, 3, 4])  # Do not supervise the heart (class 2)
        else:
            raise ValueError(args.mode, args.dataset)

        # Initialize new model, optimizer, etc. for each fold
        net, optimizer, device, _, _, K = setup(args)  # Reset model and optimizer for each fold

        # Initialize logs for losses and metrics
        log_loss_tra = torch.zeros((args.epochs, len(train_loader)))
        log_dice_tra = torch.zeros((args.epochs, len(train_loader.dataset), K))
        log_loss_val = torch.zeros((args.epochs, len(val_loader)))
        log_dice_val = torch.zeros((args.epochs, len(val_loader.dataset), K))
        # log_iou_tra = torch.zeros((args.epochs, len(train_loader)))  # IoU during training
        # log_iou_val = torch.zeros((args.epochs, len(val_loader)))    # IoU during validation
        # log_ahd_tra = torch.zeros((args.epochs, len(train_loader)))  # AHD during training
        # log_ahd_val = torch.zeros((args.epochs, len(val_loader)))    # AHD during validation

        current_fold_dice = []
        for e in range(args.epochs):
            for m in ['train', 'val']:
                match m:
                    case 'train':
                        net.train()
                        opt = optimizer
                        cm = Dcm
                        desc = f">> Training   ({e: 4d})"
                        loader = train_loader
                        log_loss = log_loss_tra
                        log_dice = log_dice_tra
                        # log_iou = log_iou_tra
                        # log_ahd = log_ahd_tra
                    case 'val':
                        net.eval()
                        opt = None
                        cm = torch.no_grad
                        desc = f">> Validation ({e: 4d})"
                        loader = val_loader
                        log_loss = log_loss_val
                        log_dice = log_dice_val
                        # log_iou = log_iou_val
                        # log_ahd = log_ahd_val

                with cm():  # Either dummy context manager or torch.no_grad for validation
                    j = 0
                    tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                    for i, data in tq_iter:
                        img = data['images'].to(device)
                        gt = data['gts'].to(device)

                        if opt:  # Only for training
                            opt.zero_grad()

                        # Ensure the data range is valid
                        assert 0 <= img.min() and img.max() <= 1
                        B, _, W, H = img.shape

                        # Get predictions
                        pred_logits = net(img)
                        pred_probs = F.softmax(1 * pred_logits, dim=1)  # Softmax across classes

                        # For each sample in the batch
                        pred_seg = probs2one_hot(pred_probs)
                        log_dice[e, j:j + B, :] = dice_coef(gt, pred_seg)  # One DSC value per sample and per class

                        # Compute IoU
                        # for b in range(B):
                        #     for k in range(K):
                        #         intersection_area = intersection(pred_seg[b, k], gt[b, k])
                        #         union_area = union(pred_seg[b, k], gt[b, k])
                        #         iou = intersection_area / (union_area + 1e-8)
                        #         log_iou[e, j + b, k] = iou
                        
                        # Compute Hausdorff Distance
                        # for b in range(B):
                        #     for k in range(K):
                        #         ahd = average_hausdorff_distance(pred_seg[b, k], gt[b, k])
                        #         log_ahd[e, j + b, k] = ahd
                        # Loss computation
                        loss = loss_fn(pred_probs, gt)
                        log_loss[e, i] = loss.item()

                        if opt:  # Only for training
                            loss.backward()
                            opt.step()

                        if m == 'val':
                            with warnings.catch_warnings():
                                warnings.filterwarnings('ignore', category=UserWarning)
                                predicted_class = probs2class(pred_probs)
                                mult = 63 if K == 5 else (255 / (K - 1))
                                save_images(predicted_class * mult, data['stems'], args.dest / f"iter{e:03d}" / m)

                        j += B

                        # Update the progress bar with metrics
                        postfix_dict = {
                            "Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                            "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"#,
                            # "IoU": f"{log_iou[e, :j, 1:].mean():05.3f}",
                            #  "AHD": f"{log_ahd[e, :j, 1:].mean():05.3f}"
                             }

                        if K > 2:  # Multi-class case
                            postfix_dict |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}" for k in range(1, K)
                                             }
                            # postfix_dict |= {f"IoU-{k}": f"{log_iou[e, :j, k].mean():05.3f}" for k in range(1, K)
                            #                  }
                            # postfix_dict |= {f"AHD-{k}": f"{log_ahd[e, :j, k].mean():05.3f}" for k in range(1, K)
                            #                  }
                        tq_iter.set_postfix(postfix_dict)

            # Save the metrics at each epoch
            np.save(args.dest / "loss_tra.npy", log_loss_tra)
            np.save(args.dest / "dice_tra.npy", log_dice_tra)
            # np.save(args.dest / "iou_tra.npy", log_iou_tra)
            # np.save(args.dest / "ahd_tra.npy", log_ahd_tra)
            np.save(args.dest / "loss_val.npy", log_loss_val)
            np.save(args.dest / "dice_val.npy", log_dice_val)
            # np.save(args.dest / "iou_val.npy", log_iou_val)
            # np.save(args.dest / "ahd_val.npy", log_ahd_val)

            # Track best Dice score
            current_dice = log_dice_val[e, :, 1:].mean().item()
            if current_dice > best_dice:
                print(f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC")
                best_dice = current_dice
                with open(args.dest / "best_epoch.txt", 'w') as f:
                    f.write(str(e))

                best_folder = args.dest / "best_epoch"
                if best_folder.exists():
                    rmtree(best_folder)
                copytree(args.dest / f"iter{e:03d}", Path(best_folder))

                torch.save(net, args.dest / "bestmodel.pkl")
                torch.save(net.state_dict(), args.dest / "bestweights.pt")
        # append the best dice from the current fold
        kfolds_dice.append(best_dice)

    # After completing all folds:
    mean_dice = np.mean(kfolds_dice)
    print(f">>> Mean Dice across all folds: {mean_dice:05.3f}")

    # Save the mean dice score to a file
    with open(args.dest / "kfold_results.txt", 'w') as f:
        f.write(f"Mean Dice across all folds: {mean_dice:05.3f}\n")

    # Optionally, save the k-fold dice values to a numpy file
    np.save(args.dest / "kfold_dice_scores.npy", np.array(kfolds_dice))
    print(f'Results over folds are stored in {args.dest}/kfold_dice_scores.npy and {args.dest}/kfold_results.txt')

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensures deterministic behavior for CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logic around epochs and logging easily.")
                             
    parser.add_argument('--hyper_parameter', type=float, default=1e-5,
                        help="The hyperparameter to tune")
    parser.add_argument('--k_folds', type=int, default=2,
                        help="specify k folds")


    args = parser.parse_args()

    pprint(vars(args))

    runTraining(args)


if __name__ == '__main__':
    main()
