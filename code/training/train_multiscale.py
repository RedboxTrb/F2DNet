"""
Training script for multi-scale ABC fractional Attention U-Net.
Uses 10-channel input: Green + 9 ABC fractional features
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import argparse

from dataset_multiscale import get_dataloaders_multiscale
from model_multiscale import AttentionUNetMultiscale


class TverskyLoss(nn.Module):
    """Tversky loss for handling class imbalance in vessel segmentation"""
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred_logits, target, mask=None):
        pred = torch.sigmoid(pred_logits)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        if mask is not None:
            mask_flat = mask.view(-1)
            pred_flat = pred_flat * mask_flat
            target_flat = target_flat * mask_flat

        TP = (pred_flat * target_flat).sum()
        FP = (pred_flat * (1 - target_flat)).sum()
        FN = ((1 - pred_flat) * target_flat).sum()

        tversky_index = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)
        return 1 - tversky_index


class FocalLoss(nn.Module):
    """Focal loss for hard example mining"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_logits, target):
        pred = torch.sigmoid(pred_logits)
        bce = nn.functional.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')

        p_t = pred * target + (1 - pred) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_weight = alpha_t * focal_weight

        return (focal_weight * bce).mean()


class CombinedLoss(nn.Module):
    """Combined Tversky and Focal loss"""
    def __init__(self, tversky_weight=0.7, focal_weight=0.3,
                 alpha=0.7, beta=0.3, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.tversky_loss = TverskyLoss(alpha=alpha, beta=beta)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight

    def forward(self, pred_logits, target, mask=None):
        tversky = self.tversky_loss(pred_logits, target, mask)
        focal = self.focal_loss(pred_logits, target)
        return self.tversky_weight * tversky + self.focal_weight * focal


def compute_metrics(pred_logits, target, mask=None, threshold=0.5):
    """Compute evaluation metrics"""
    pred = torch.sigmoid(pred_logits)
    pred_binary = (pred > threshold).float()

    if mask is not None:
        pred_binary = pred_binary * mask
        target = target * mask

    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)

    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    fn = ((1 - pred_flat) * target_flat).sum()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)

    return {
        'sensitivity': sensitivity.item(),
        'specificity': specificity.item(),
        'precision': precision.item(),
        'dice': dice.item()
    }


def train_epoch(model, loader, criterion, optimizer, device, scaler, accumulation_steps=4):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
        features = batch['features'].to(device)  # [B, 10, H, W]
        gt = batch['gt'].to(device)
        fov_mask = batch['fov_mask'].to(device)

        with torch.amp.autocast('cuda'):
            pred = model(features)
            loss = criterion(pred, gt, fov_mask)
            loss = loss / accumulation_steps  # Normalize loss

        scaler.scale(loss).backward()

        # Update weights every accumulation_steps
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_metrics = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            features = batch['features'].to(device)
            gt = batch['gt'].to(device)
            fov_mask = batch['fov_mask'].to(device)

            with torch.amp.autocast('cuda'):
                pred = model(features)
                loss = criterion(pred, gt, fov_mask)

            total_loss += loss.item()
            metrics = compute_metrics(pred, gt, fov_mask)
            all_metrics.append(metrics)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0].keys()}
    return total_loss / len(loader), avg_metrics


def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    print(f"Multi-scale model (10-channel input)")

    train_loader, val_loader, _ = get_dataloaders_multiscale(
        dataset_dir=config['dataset_dir'],
        features_dir=config['features_dir'],
        splits_dir=config['splits_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    model = AttentionUNetMultiscale(in_channels=10, out_channels=1).to(device)

    criterion = CombinedLoss(
        tversky_weight=config['tversky_weight'],
        focal_weight=config['focal_weight'],
        alpha=config['tversky_alpha'],
        beta=config['tversky_beta'],
        focal_alpha=config['focal_alpha'],
        focal_gamma=config['focal_gamma']
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)
    scaler = torch.amp.GradScaler('cuda')

    best_dice = 0
    train_losses, val_losses, val_dices = [], [], []

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler,
                                 accumulation_steps=config.get('accumulation_steps', 4))
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_metrics['dice'])

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Dice: {val_metrics['dice']:.4f} | Sensitivity: {val_metrics['sensitivity']:.4f} | Specificity: {val_metrics['specificity']:.4f}")

        scheduler.step(val_loss)

        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice': best_dice,
                'metrics': val_metrics,
                'config': config
            }, output_dir / 'best_model.pth')
            print(f"Best model saved (Dice: {best_dice:.4f})")

    # Save training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='Train')
    ax1.plot(val_losses, label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(val_dices, color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)

    print(f"\nTraining complete. Best Dice: {best_dice:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pth'}")


def main():
    parser = argparse.ArgumentParser(description='Train multi-scale Attention U-Net')
    parser.add_argument('--dataset_dir', type=str, default='Dataset', help='Dataset directory')
    parser.add_argument('--features_dir', type=str, default='Dataset/multiscale_features', help='ABC features directory')
    parser.add_argument('--splits_dir', type=str, default='splits', help='Splits directory')
    parser.add_argument('--output_dir', type=str, default='models_multiscale', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--num_epochs', type=int, default=120, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')

    args = parser.parse_args()

    config = {
        'dataset_dir': args.dataset_dir,
        'features_dir': args.features_dir,
        'splits_dir': args.splits_dir,
        'output_dir': args.output_dir,
        'batch_size': args.batch_size,
        'accumulation_steps': args.accumulation_steps,
        'num_workers': args.num_workers,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'tversky_weight': 0.7,
        'focal_weight': 0.3,
        'tversky_alpha': 0.7,
        'tversky_beta': 0.3,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
    }

    train(config)


if __name__ == "__main__":
    main()
