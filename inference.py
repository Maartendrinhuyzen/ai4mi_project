#!/usr/bin/env python3

import argparse
from pathlib import Path
from pprint import pprint
from typing import Any

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import SliceDatasetWithoutGT
from ENet import ENet
from Unet import UNet
from ShallowNet import shallowCNN
from UnetAttention import UNetAttention
from utils import class2one_hot, probs2class, save_images

models = {
    "shallowCNN": shallowCNN,
    "ENet": ENet,
    "UNet": UNet,
    "UNetAttention": UNetAttention
}

datasets_params = {
    "TOY2": {'K': 2, 'B': 2},
    "SEGTHOR": {'K': 5, 'B': 8},
    "SEGTHOR_transformed": {'K': 5, 'B': 8},
    "SEGTHOR_TESTSET": {'K': 5, 'B': 8}
}

def setup_inference(args) -> tuple[nn.Module, Any, DataLoader, int]:
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Using {device} for inference")

    K: int = datasets_params[args.dataset]['K']
    net = models[args.model](1, K)
    net.to(device)

    # Load weights
    net.load_state_dict(torch.load(args.weights, map_location=device))
    net.eval()

    # Dataset part
    root_dir = Path("data") / args.dataset
    B: int = datasets_params[args.dataset]['B']

    img_transform = transforms.Compose([
        lambda img: img.convert('L'),
        lambda img: np.array(img)[np.newaxis, ...],
        lambda nd: nd / 255,
        lambda nd: torch.tensor(nd, dtype=torch.float32)
    ])

    dataset = SliceDatasetWithoutGT('test',  # Using validation set for inference
                           root_dir,
                           img_transform=img_transform,
                           debug=args.debug)

    loader = DataLoader(dataset,
                        batch_size=B,
                        num_workers=args.num_workers,
                        shuffle=False)

    return net, loader, device, K

def run_inference(args):
    print(f">>> Running inference on {args.dataset} using {args.model} model")
    net, loader, device, K = setup_inference(args)

    n = len(loader)
    for i, data in enumerate(loader):
        print(f"Running inference on image {i}/{n} ...")
        img = data['images'].to(device)
        img_paths = [image_path.split('_')[1] for image_path in data['stems']]
        assert 0 <= img.min() and img.max() <= 1

        with torch.no_grad():
            pred_logits = net(img)
            pred_probs = torch.softmax(pred_logits, dim=1)
            pred_class = probs2class(pred_probs)

            mult = 63 if K == 5 else (255 / (K - 1))
            save_images(pred_class * mult, data['stems'], args.dest / f"inference")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='SEGTHOR_TESTSET', choices=datasets_params.keys())
    parser.add_argument('--model', required=True, choices=list(models.keys()),
                        help="Specify the model architecture to load")
    parser.add_argument('--weights', type=Path, required=True,
                        help="Path to the weights file (e.g., bestweights.pt)")
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the inference results")
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logic easily.")

    args = parser.parse_args()

    pprint(vars(args))

    run_inference(args)


if __name__ == '__main__':
    main()
