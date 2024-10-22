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
from typing import Callable, Union, List, Tuple

from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset


def make_dataset(root, subset) -> List[Tuple[Path, Path]]:
    """
    Creates the dataset by combining image and label pairs.

    Args:
    - root: The root directory of the dataset.
    - subset: The subset to load ('train', 'val', 'test', or 'full').

    Returns:
    - A list of tuples containing image and label paths.
    """
    root = Path(root)

    if subset in ['train', 'val', 'test']:
        img_path = root / subset / 'img'
        full_path = root / subset / 'gt'

        images = sorted(img_path.glob("*.png"))
        full_labels = sorted(full_path.glob("*.png"))

        return list(zip(images, full_labels))

    elif subset == 'full':
        # Combine the 'train' and 'val' sets for the full dataset
        train_data = make_dataset(root, 'train')
        val_data = make_dataset(root, 'val')
        return train_data + val_data

    else:
        raise ValueError(f"Unknown subset {subset}. Valid options are 'train', 'val', 'test', or 'full'.")


class SliceDataset(Dataset):
    """
    A dataset class to load image and label pairs for segmentation.

    Args:
    - subset: Which subset to use ('train', 'val', 'test', or 'full').
    - root_dir: The root directory containing the dataset.
    - img_transform: The transformation to apply to the images.
    - gt_transform: The transformation to apply to the ground truth labels.
    - augment: Whether to apply data augmentation.
    - equalize: Whether to apply histogram equalization.
    - debug: Whether to enable debug mode (using a small subset of data).
    """
    def __init__(self, subset, root_dir, img_transform=None,
                 gt_transform=None, augment=False, equalize=False, debug=False):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: bool = augment
        self.equalize: bool = equalize

        # Use the 'full' option to combine 'train' and 'val' sets
        self.files = make_dataset(root_dir, subset)
        if debug:
            self.files = self.files[:10]

        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]

        img: Tensor = self.img_transform(Image.open(img_path))
        gt: Tensor = self.gt_transform(Image.open(gt_path))

        _, W, H = img.shape
        K, _, _ = gt.shape
        assert gt.shape == (K, W, H)

        return {"images": img,
                "gts": gt,
                "stems": img_path.stem}


class SliceDatasetWithoutGT(Dataset):
    """
    A dataset class to load images for inference, without ground truth labels.

    Args:
    - subset: Which subset to use ('train', 'val', 'test', or 'full').
    - root_dir: The root directory containing the dataset.
    - img_transform: The transformation to apply to the images.
    - augment: Whether to apply data augmentation.
    - equalize: Whether to apply histogram equalization.
    - debug: Whether to enable debug mode (using a small subset of data).
    """
    def __init__(self, subset, root_dir, img_transform=None, augment=False, equalize=False, debug=False):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.augmentation: bool = augment
        self.equalize: bool = equalize

        # Only load images, no ground truth labels
        self.files = self._make_image_dataset(root_dir, subset)
        if debug:
            self.files = self.files[:10]

        print(f">> Created {subset} dataset with {len(self)} images (no ground truth)...")

    def _make_image_dataset(self, root, subset) -> List[Path]:
        """
        Creates the dataset by loading image paths only.

        Args:
        - root: The root directory of the dataset.
        - subset: The subset to load ('train', 'val', 'test', or 'full').

        Returns:
        - A list of image paths.
        """
        root = Path(root)

        if subset in ['test']:
            img_path = root / subset / 'img'
            images = sorted(img_path.glob("*.png"))
            return images
        else:
            raise ValueError(f"Unknown subset {subset}. Valid options are 'train', 'val', 'test', or 'full'.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict[str, Union[Tensor, str]]:
        img_path = self.files[index]

        img: Tensor = self.img_transform(Image.open(img_path))

        return {"images": img,
                "stems": img_path.stem}
