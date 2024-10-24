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
import pickle
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
                   union,
                   torch2D_Hausdorff_distance, 
                   collect_patient_slices,
                   calculate_3d_dice,
                   calculate_3d_iou,
                   calculate_3d_hausdorff,
                   count_class_occurrences,
                   calculate_class_weights)

from losses import (CrossEntropy,
                   BalancedCrossEntropy,
                   DiceLoss,
                   FocalLoss,
                   CombinedLoss)

from multiprocessing import Pool
from tqdm import tqdm

from sklearn.model_selection import KFold



datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the clases with C (often used for the number of Channel)
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
    lr = args.hyper_parameter # Original learning rate: lr = 0.0005
    if args.optimizer == 'Adam':
        # Original optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)
   
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

    train_set = SliceDataset('train',
                             root_dir,
                             img_transform=img_transform,
                             gt_transform=gt_transform,
                             debug=args.debug)
    train_loader = DataLoader(train_set,
                              batch_size=B,
                              num_workers=args.num_workers,
                              shuffle=True)

    val_set = SliceDataset('val',
                           root_dir,
                           img_transform=img_transform,
                           gt_transform=gt_transform,
                           debug=args.debug)
    val_loader = DataLoader(val_set,
                            batch_size=B,
                            num_workers=args.num_workers,
                            shuffle=False)

    
    # Combine train and validation sets if 'full' option is selected
    full_set = SliceDataset('full',  # Loading full dataset
                            root_dir,
                            img_transform=img_transform,
                            gt_transform=gt_transform,
                            debug=args.debug)
    
    # Return only the full dataset loader
    return (net, optimizer, device, train_loader, val_loader, full_set, K)

def runTraining(args):
    print(f">>> Setting up to train on {args.dataset} with {args.mode} and {args.optimizer}")

    # Setup function now returns train, validation, and full datasets/loaders
    net, optimizer, device, train_loader, val_loader, full_set, K = setup(args)

    best_dice = 0

    if args.k_folds > 1:
        # K-fold cross-validation logic remains the same
        kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
        kfolds_dice = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(full_set)):
            print(f"Running fold {fold + 1}/{args.k_folds}")

            # Create subsets for training and validation using the indices from KFold
            train_subset = torch.utils.data.Subset(full_set, train_idx)
            val_subset = torch.utils.data.Subset(full_set, val_idx)

            # Create data loaders for this fold
            fold_train_loader = DataLoader(train_subset, batch_size=args.num_workers, shuffle=True)
            fold_val_loader = DataLoader(val_subset, batch_size=args.num_workers, shuffle=False)

            net, optimizer, _, _, _, _, _ = setup(args)

            current_fold_dice = train_model_fold(args, net, optimizer, device, K, fold_train_loader, fold_val_loader, fold)
            kfolds_dice.append(current_fold_dice)

        # After completing all folds
        mean_dice = np.mean(kfolds_dice)
        print(f">>> Mean Dice across all folds: {mean_dice:05.3f}")

        # Save k-fold results
        with open(args.dest / "kfold_results.txt", 'w') as f:
            f.write(f"Mean Dice across all folds: {mean_dice:05.3f}\n")
        np.save(args.dest / "kfold_dice_scores.npy", np.array(kfolds_dice))

    else:
        # Standard training when k_folds == 1 (no cross-validation)
        print("Running standard training without k-fold cross-validation.")

        # Using the predefined train_loader and val_loader from the setup
        best_dice = train_model_fold(args, net, optimizer, device, K, train_loader, val_loader, fold=None)

        print(f"Best Dice Score: {best_dice:05.3f}")

        # Save the best dice score to a file
        with open(args.dest / "best_results.txt", 'w') as f:
            f.write(f"Best Dice Score: {best_dice:05.3f}\n")

def get_loss_fn(args, train_loader, K):
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
    return loss_fn

def train_model_fold(args, net, optimizer, device, K, train_loader, val_loader, fold=None):
    # Initialize logs for losses and metrics
    best_dice = 0
    log_loss_tra = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val = torch.zeros((args.epochs, len(val_loader.dataset), K))

    loss_fn = get_loss_fn(args, train_loader, K)
    log_3d_dice_val = {}
    log_3d_dice_tra = {}
    log_3d_iou_val = {}
    log_3d_iou_tra = {}
    log_3d_ahd_val = {}
    log_3d_ahd_tra = {}
    for e in range(args.epochs):
        patient_slices_tra = {}
        patient_slices_val = {}
        
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
                case 'val':
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    log_dice = log_dice_val

            
            with cm():  # Either dummy context manager or torch.no_grad for validation
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data['images'].to(device)
                    gt = data['gts'].to(device)
                    img_paths = [image_path.split('_')[1] for image_path in data['stems']]

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

                    # Inside the batch loop
                    if m == 'val':
                        patient_slices_val = collect_patient_slices(patient_slices_val, img_paths, pred_seg, gt, B)

                    else:
                        patient_slices_tra = collect_patient_slices(patient_slices_tra, img_paths, pred_seg, gt, B)

                    log_dice[e, j:j + B, :] = dice_coef(gt, pred_seg)  # One DSC value per sample and per class

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
                        "Loss": f"{log_loss[e, :i + 1].mean():5.2e}",
                         }

                    if K > 2:  # Multi-class case
                        postfix_dict |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}" for k in range(1, K)
                                         }
                    tq_iter.set_postfix(postfix_dict)
        
        # this is what patient slices looks like:
        # patient_slices = {'Patient03': {'pred_slices': [], 'gt_slices': []},
        #                   'Patient04': {'pred_slices': [], 'gt_slices': []},
        #                   ...}
        # Calculate 3D metrics for training and validation
        log_3d_dice_tra[e] = calculate_3d_dice(patient_slices_tra)
        log_3d_iou_tra[e] = calculate_3d_iou(patient_slices_tra)

        log_3d_dice_val[e] = calculate_3d_dice(patient_slices_val)
        log_3d_iou_val[e] = calculate_3d_iou(patient_slices_val)

        # Calculate and save 3D AHD only for the last epoch
        if e == args.epochs - 1:
            log_3d_ahd_val[e], log_3d_ahd_tra[e] = (
                calculate_3d_hausdorff(patient_slices_val),
                calculate_3d_hausdorff(patient_slices_tra),
            )
            print(f"Final epoch 3D AHD training: {log_3d_ahd_tra[e]}")
            print(f"Final epoch 3D AHD validation: {log_3d_ahd_val[e]}")

            save_pickle(args.dest, {
                'final_3d_ahd_val.pkl': log_3d_ahd_val[e],
                'final_3d_ahd_tra.pkl': log_3d_ahd_tra[e],
            })

        # Save metrics at each epoch
        save_numpy(args.dest, {
            "loss_tra.npy": log_loss_tra,
            "dice_tra.npy": log_dice_tra,
            "loss_val.npy": log_loss_val,
            "dice_val.npy": log_dice_val,
        })

        save_pickle(args.dest, {
            'log_3d_dice_val.pkl': log_3d_dice_val,
            'log_3d_dice_tra.pkl': log_3d_dice_tra,
            'log_3d_iou_val.pkl': log_3d_iou_val,
            'log_3d_iou_tra.pkl': log_3d_iou_tra,
        })
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

    return best_dice

def save_pickle(dest, data_dict):
    """Helper function to save multiple pickle files."""
    for filename, data in data_dict.items():
        with open(dest / filename, 'wb') as f:
            pickle.dump(data, f)

def save_numpy(dest, data_dict):
    """Helper function to save multiple NumPy arrays."""
    for filename, data in data_dict.items():
        np.save(dest / filename, data)

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
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
    parser.add_argument('--optimizer', default='AdamW', choices=['Adam','AdamW'],
                        help="Choose the optimizer to use from Adam or AdamW")                         
    parser.add_argument('--hyper_parameter', type=float, default=1e-5,
                        help="The hyperparameter to tune")
    parser.add_argument('--k_folds', type=int, default=1,
                        help="specify k folds")


    args = parser.parse_args()

    pprint(vars(args))

    runTraining(args)


if __name__ == '__main__':
    main()