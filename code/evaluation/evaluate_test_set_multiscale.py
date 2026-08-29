"""
Comprehensive evaluation on test set for multiscale Attention U-Net
Generates metrics, visualizations, ROC curves, and LaTeX tables
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns
import argparse

from model_multiscale import AttentionUNetMultiscale


def load_model(checkpoint_path, device='cuda'):
    """Load trained multiscale vessel segmentation model"""
    model = AttentionUNetMultiscale(in_channels=10, out_channels=1)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def load_multiscale_features(image_path, features_path, image_size=1024):
    """Load image and ABC features, return 10-channel tensor"""
    # Load RGB image
    rgb = cv2.imread(str(image_path))
    if rgb is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Resize if needed
    if rgb.shape[0] != image_size or rgb.shape[1] != image_size:
        rgb = cv2.resize(rgb, (image_size, image_size))

    # Extract and normalize green channel
    green = rgb[:, :, 1].astype(np.float32) / 255.0

    # Load ABC fractional features [H, W, 9]
    abc_features = np.load(features_path).astype(np.float32)

    # Resize ABC features if needed
    if abc_features.shape[0] != image_size or abc_features.shape[1] != image_size:
        # Resize each channel separately
        resized_features = []
        for i in range(abc_features.shape[2]):
            resized_channel = cv2.resize(abc_features[:, :, i], (image_size, image_size))
            resized_features.append(resized_channel)
        abc_features = np.stack(resized_features, axis=-1)

    # Combine: [green, ABC_features] -> [H, W, 10]
    combined_features = np.concatenate([
        green[..., None],  # [H, W, 1]
        abc_features       # [H, W, 9]
    ], axis=-1)

    # Convert to tensor [10, H, W]
    features_tensor = torch.from_numpy(combined_features).permute(2, 0, 1).float()

    return features_tensor


def predict_multiscale(features_tensor, model, device):
    """Run inference with multiscale model"""
    with torch.no_grad():
        features_batch = features_tensor.unsqueeze(0).to(device)  # [1, 10, H, W]

        with torch.amp.autocast('cuda'):
            output = model(features_batch)

        prob_map = torch.sigmoid(output).squeeze(0).squeeze(0).cpu().numpy()
        binary_mask = (prob_map > 0.5).astype(np.uint8)

    return binary_mask, prob_map


def compute_metrics(pred_binary, pred_prob, gt_binary, fov_mask=None):
    """Compute comprehensive evaluation metrics"""
    # Apply mask if provided
    if fov_mask is not None:
        pred_binary = pred_binary * fov_mask
        gt_binary = gt_binary * fov_mask
        mask_flat = fov_mask.flatten() > 0
    else:
        mask_flat = np.ones(pred_binary.size, dtype=bool)

    pred_flat = pred_binary.flatten()[mask_flat]
    gt_flat = gt_binary.flatten()[mask_flat]
    prob_flat = pred_prob.flatten()[mask_flat]

    # Confusion matrix
    tp = np.sum((pred_flat == 1) & (gt_flat == 1))
    fp = np.sum((pred_flat == 1) & (gt_flat == 0))
    fn = np.sum((pred_flat == 0) & (gt_flat == 1))
    tn = np.sum((pred_flat == 0) & (gt_flat == 0))

    # Metrics
    sensitivity = tp / (tp + fn + 1e-10)
    specificity = tn / (tn + fp + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-10)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-10)
    jaccard = tp / (tp + fp + fn + 1e-10)

    # ROC AUC and PR AUC
    if len(np.unique(gt_flat)) > 1:
        fpr, tpr, _ = roc_curve(gt_flat, prob_flat)
        roc_auc = auc(fpr, tpr)
        pr_auc = average_precision_score(gt_flat, prob_flat)
    else:
        roc_auc = 0.0
        pr_auc = 0.0

    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'accuracy': accuracy,
        'f1': f1,
        'dice': dice,
        'jaccard': jaccard,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }


def evaluate_test_set(dataset_dir, features_dir, splits_dir, model_path, output_dir, device='cuda'):
    """Evaluate multiscale model on test set"""
    dataset_dir = Path(dataset_dir)
    features_dir = Path(features_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test split
    with open(Path(splits_dir) / "test.txt") as f:
        test_list = [line.strip() for line in f]

    print(f"Found {len(test_list)} test images")

    # Load model
    print(f"Loading multiscale model from {model_path}...")
    model = load_model(model_path, device)
    print("✓ Model loaded (10-channel input: Green + 9 ABC features)\n")

    # Evaluate all test images
    results = []
    all_gt = []
    all_prob = []

    for image_name in tqdm(test_list, desc="Evaluating"):
        try:
            # Load images and features
            image_path = dataset_dir / "images" / f"{image_name}.png"
            features_path = features_dir / f"{image_name}_features.npy"
            gt_path = dataset_dir / "ground_truth" / f"{image_name}_gt.png"
            mask_path = dataset_dir / "masks" / f"{image_name}_mask.png"

            if not features_path.exists():
                print(f"\nWarning: Features not found for {image_name} - skipping")
                continue

            # Load ground truth and mask
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            fov_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if gt is None or fov_mask is None:
                print(f"Warning: Could not load GT/mask for {image_name}")
                continue

            gt_binary = (gt > 127).astype(np.uint8)
            fov_mask = (fov_mask > 127).astype(np.uint8)

            # Load multiscale features and predict
            features_tensor = load_multiscale_features(image_path, features_path)
            pred_binary, pred_prob = predict_multiscale(features_tensor, model, device)

            # Resize if needed
            if pred_prob.shape != gt.shape:
                pred_prob = cv2.resize(pred_prob, (gt.shape[1], gt.shape[0]))
                pred_binary = cv2.resize(pred_binary, (gt.shape[1], gt.shape[0]))

            # Compute metrics
            metrics = compute_metrics(pred_binary, pred_prob, gt_binary, fov_mask)
            metrics['image_name'] = image_name
            results.append(metrics)

            # Store for aggregate analysis
            mask_flat = fov_mask.flatten() > 0
            all_gt.extend(gt_binary.flatten()[mask_flat].tolist())
            all_prob.extend(pred_prob.flatten()[mask_flat].tolist())

        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")

    if len(results) == 0:
        print("\n❌ No test images were successfully evaluated!")
        print("Make sure ABC features are generated using generate_test_features.py")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save detailed results
    csv_path = output_dir / 'test_results_detailed.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Detailed results saved: {csv_path}")

    # Compute aggregate statistics
    metrics_keys = ['sensitivity', 'specificity', 'precision', 'accuracy', 'f1', 'dice', 'jaccard', 'roc_auc', 'pr_auc']

    print("\n" + "="*70)
    print("TEST SET EVALUATION RESULTS - MULTISCALE MODEL")
    print("="*70)
    print(f"Total test images: {len(df)}")
    print("\nMETRICS (Mean ± Std):")
    for key in metrics_keys:
        mean = df[key].mean()
        std = df[key].std()
        print(f"  {key.upper():12s}: {mean:.4f} ± {std:.4f}")

    # Generate summary report
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("TEST SET EVALUATION SUMMARY - MULTISCALE ABC FRACTIONAL MODEL\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Input: 10 channels (Green + 9 ABC fractional features)\n")
        f.write(f"Total test images: {len(df)}\n\n")
        f.write("PERFORMANCE METRICS:\n")
        for key in metrics_keys:
            mean = df[key].mean()
            std = df[key].std()
            min_val = df[key].min()
            max_val = df[key].max()
            f.write(f"  {key.upper():12s}: {mean:.4f} ± {std:.4f} (range: {min_val:.4f} - {max_val:.4f})\n")

    print(f"✓ Summary saved: {output_dir / 'summary.txt'}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. ROC Curve
    all_gt = np.array(all_gt)
    all_prob = np.array(all_prob)
    fpr, tpr, _ = roc_curve(all_gt, all_prob)
    roc_auc_val = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_val:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curve - Multiscale Model', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ ROC curve saved")

    # 2. Precision-Recall Curve
    precision_curve, recall_curve, _ = precision_recall_curve(all_gt, all_prob)
    pr_auc_val = average_precision_score(all_gt, all_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc_val:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title('Precision-Recall Curve - Multiscale Model', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'pr_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ PR curve saved")

    # 3. Metric distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    plot_metrics = ['sensitivity', 'specificity', 'dice', 'jaccard']
    for idx, metric in enumerate(plot_metrics):
        axes[idx].boxplot([df[metric]], labels=[metric.capitalize()])
        axes[idx].set_ylabel('Score', fontsize=10)
        axes[idx].set_title(f'{metric.capitalize()} Distribution', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim([0, 1.05])

        mean_val = df[metric].mean()
        axes[idx].axhline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.4f}')
        axes[idx].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'metric_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Metric distributions saved")

    # 4. Metric comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    means = [df[key].mean() for key in metrics_keys]
    stds = [df[key].std() for key in metrics_keys]
    x = np.arange(len(metrics_keys))

    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, color='green')
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Multiscale Model Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([key.upper() for key in metrics_keys], rotation=45, ha='right')
    ax.set_ylim([0, 1.1])
    ax.grid(True, axis='y', alpha=0.3)

    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.3f}\n±{std:.3f}',
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Metrics comparison saved")

    # 5. Generate LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Multiscale Model Test Set Performance}")
    latex_lines.append("\\label{tab:multiscale_results}")
    latex_lines.append("\\begin{tabular}{lc}")
    latex_lines.append("\\hline")
    latex_lines.append("Metric & Score \\\\")
    latex_lines.append("\\hline")
    for key in metrics_keys:
        mean = df[key].mean()
        std = df[key].std()
        latex_lines.append(f"{key.capitalize().replace('_', ' ')} & ${mean:.4f} \\pm {std:.4f}$ \\\\")
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    with open(output_dir / 'results_table.tex', 'w') as f:
        f.write("\n".join(latex_lines))
    print("✓ LaTeX table saved")

    print("\n" + "="*70)
    print("✓ MULTISCALE MODEL EVALUATION COMPLETE!")
    print(f"  Output directory: {output_dir}")
    print(f"  Test images: {len(df)}")
    print(f"  Mean Dice Score: {df['dice'].mean():.4f} ± {df['dice'].std():.4f}")
    print(f"  Mean ROC AUC: {roc_auc_val:.4f}")
    print("="*70)

    return df


def main():
    parser = argparse.ArgumentParser(description='Evaluate multiscale model on test set')
    parser.add_argument('--dataset_dir', type=str, default='Dataset/Vessel_maps_dataset',
                       help='Dataset directory')
    parser.add_argument('--features_dir', type=str, default='Dataset/multiscale_features',
                       help='ABC features directory')
    parser.add_argument('--splits_dir', type=str, default='Dataset/Vessel_maps_dataset/splits',
                       help='Splits directory')
    parser.add_argument('--model_path', type=str, default='msc/models/best_model.pth',
                       help='Path to trained multiscale model')
    parser.add_argument('--output_dir', type=str, default='results/multiscale_test_evaluation',
                       help='Output directory')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    evaluate_test_set(
        dataset_dir=args.dataset_dir,
        features_dir=args.features_dir,
        splits_dir=args.splits_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=device
    )


if __name__ == '__main__':
    main()
