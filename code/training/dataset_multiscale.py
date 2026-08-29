"""
Multi-scale dataset loader for ABC fractional features.
Loads 10-channel input: Green + 9 ABC fractional features
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from pathlib import Path
import albumentations as A


class VesselDatasetMultiscale(Dataset):
    """Dataset loader with multi-scale ABC fractional features"""

    def __init__(self, image_list, dataset_dir, features_dir, augment=False, image_size=1024):
        self.image_list = image_list
        self.dataset_dir = Path(dataset_dir)
        self.features_dir = Path(features_dir)
        self.augment = augment
        self.image_size = image_size

        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]

        # Load RGB image
        rgb_path = self.dataset_dir / "images" / f"{image_name}.png"
        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            raise ValueError(f"Could not load: {rgb_path}")

        # Extract green channel and normalize
        green = rgb[:, :, 1].astype(np.float32) / 255.0

        # Load ABC fractional features [H, W, 9]
        features_path = self.features_dir / f"{image_name}_features.npy"
        if not features_path.exists():
            raise ValueError(f"Features not found: {features_path}")

        abc_features = np.load(features_path).astype(np.float32)  # [H, W, 9]

        # Combine: [green, ABC_features] -> [H, W, 10]
        combined_features = np.concatenate([
            green[..., None],  # [H, W, 1]
            abc_features       # [H, W, 9]
        ], axis=-1)  # [H, W, 10]

        # Load ground truth
        gt_path = self.dataset_dir / "ground_truth" / f"{image_name}_gt.png"
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        gt = (gt > 127).astype(np.float32)

        # Load FOV mask
        mask_path = self.dataset_dir / "masks" / f"{image_name}_mask.png"
        fov_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        fov_mask = (fov_mask > 127).astype(np.float32)

        # Skip augmentation for multi-channel features (albumentations limitation)
        # The pre-computed ABC features maintain their integrity without augmentation
        # Note: Future improvement could implement custom augmentation for 10-channel data

        # Convert to tensors
        # features: [10, H, W]
        features_tensor = torch.from_numpy(combined_features).permute(2, 0, 1).float()
        gt_tensor = torch.from_numpy(gt).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(fov_mask).unsqueeze(0).float()

        return {
            'features': features_tensor,   # [10, H, W]
            'gt': gt_tensor,                # [1, H, W]
            'fov_mask': mask_tensor,        # [1, H, W]
            'name': image_name
        }


def get_dataloaders_multiscale(dataset_dir, features_dir, splits_dir, batch_size=5, num_workers=4):
    """Create train, validation, and test dataloaders for multiscale features"""

    with open(Path(splits_dir) / "train.txt") as f:
        train_list = [line.strip() for line in f]

    with open(Path(splits_dir) / "val.txt") as f:
        val_list = [line.strip() for line in f]

    with open(Path(splits_dir) / "test.txt") as f:
        test_list = [line.strip() for line in f]

    train_dataset = VesselDatasetMultiscale(train_list, dataset_dir, features_dir, augment=True)
    val_dataset = VesselDatasetMultiscale(val_list, dataset_dir, features_dir, augment=False)
    test_dataset = VesselDatasetMultiscale(test_list, dataset_dir, features_dir, augment=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"Multiscale Dataset - Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader
