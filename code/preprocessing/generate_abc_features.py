"""
Generate multi-scale ABC fractional derivative features for all dataset images.
This creates 9-channel feature maps (8 fractional orders + CLAHE) for each image.
"""

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import time
from scipy.ndimage import convolve
from scipy.special import gamma
from skimage.exposure import equalize_adapthist
from skimage.morphology import diamond, binary_erosion
import argparse


# Fractional orders to generate
FRACTIONAL_ORDERS = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
GAUSSIAN_SIGMA = 1.45


def create_fov_mask(image):
    """Create field of view mask to exclude black borders"""
    if len(image.shape) == 3:
        red_channel = image[:, :, 0]
    else:
        red_channel = image

    mask = (red_channel > 20).astype(np.uint8)
    se = diamond(15)
    eroded_mask = binary_erosion(mask, se).astype(np.uint8)
    return eroded_mask


def abc_fractional_kernels(v):
    """
    Create ABC fractional derivative kernels

    Args:
        v: Fractional order

    Returns:
        kernel_x, kernel_y: 3x3 convolution kernels
    """
    M = 1.0 - v + v / gamma(v)
    p = (2.0 * gamma(v) * (1.0 - v) + 1.0) / (2.0 * M * gamma(v))
    q = (2.0 ** (v - 1.0)) / (M * gamma(v))
    r = ((3.0 ** v) - 1.0) / (2.0 * M * gamma(v))

    kernel_x = np.array([
        [p,  0, -p],
        [q,  0, -q],
        [r,  0, -r]
    ], dtype=np.float64)

    kernel_y = np.array([
        [ p,  q,  r],
        [ 0,  0,  0],
        [-p, -q, -r]
    ], dtype=np.float64)

    return kernel_x, kernel_y


def compute_vesselness(gxx, gyy, gxy):
    """
    Compute vesselness measure from Hessian eigenvalues
    Vessels are tubular dark structures with one large negative eigenvalue
    """
    trace = gxx + gyy
    det = gxx * gyy - gxy * gxy
    discriminant = np.maximum(trace * trace - 4.0 * det, 0)
    sqrt_disc = np.sqrt(discriminant)

    lambda1 = (trace + sqrt_disc) / 2.0
    lambda2 = (trace - sqrt_disc) / 2.0

    lambda_abs1 = np.abs(lambda1)
    lambda_abs2 = np.abs(lambda2)

    lambda_max = np.maximum(lambda_abs1, lambda_abs2)

    # Keep only dark structures (vessels)
    is_dark = (lambda1 < 0) | (lambda2 < 0)
    vesselness = lambda_max * is_dark

    return vesselness


def principal_curvature(image, v, sigma=1.45):
    """
    Compute principal curvature using ABC fractional derivatives

    Args:
        image: Grayscale image
        v: Fractional order
        sigma: Gaussian smoothing parameter

    Returns:
        vesselness: Vessel enhancement map
    """
    img = image.astype(np.float64)

    # Gaussian blur
    img_blurred = cv2.GaussianBlur(img, (0, 0), sigma)

    # Normalize
    img_norm = (img_blurred - img_blurred.min()) / (img_blurred.max() - img_blurred.min() + 1e-8)
    img_norm = img_norm * 255.0

    # ABC kernels
    kernel_x, kernel_y = abc_fractional_kernels(v)

    # First derivatives
    grad_x = convolve(img_norm, kernel_x, mode='reflect')
    grad_y = convolve(img_norm, kernel_y, mode='reflect')

    # Hessian (second derivatives)
    gxx = convolve(grad_x, kernel_x, mode='reflect')
    gxy = convolve(grad_y, kernel_x, mode='reflect')
    gyy = convolve(grad_y, kernel_y, mode='reflect')

    # Vesselness
    vesselness = compute_vesselness(gxx, gyy, gxy)

    return vesselness


def vessel_enhancement(image, v, sigma=1.45):
    """
    Complete vessel enhancement pipeline for one fractional order

    Args:
        image: RGB fundus image
        v: Fractional order
        sigma: Gaussian smoothing parameter

    Returns:
        enhanced: CLAHE-enhanced vessel map (normalized to [0, 1])
    """
    fov_mask = create_fov_mask(image)
    green = image[:, :, 1]

    vesselness = principal_curvature(green, v, sigma)

    # Normalize to [0, 255]
    v_min = vesselness.min()
    v_max = vesselness.max()

    if v_max > v_min:
        vesselness_norm = ((vesselness - v_min) / (v_max - v_min) * 255).astype(np.uint8)
    else:
        vesselness_norm = np.zeros_like(vesselness, dtype=np.uint8)

    vesselness_masked = vesselness_norm * fov_mask

    # CLAHE enhancement
    if vesselness_masked.max() > 0:
        v_float = vesselness_masked.astype(np.float32) / 255.0

        h, w = v_float.shape
        tile_h = max(h // 4, 8)
        tile_w = max(w // 4, 8)

        enhanced = equalize_adapthist(
            v_float,
            kernel_size=(tile_h, tile_w),
            clip_limit=0.01,
            nbins=256
        )

        enhanced = (enhanced * 255).astype(np.uint8)
    else:
        enhanced = vesselness_masked

    # Normalize to [0, 1] for storage
    enhanced_norm = enhanced.astype(np.float32) / 255.0

    return enhanced_norm


def apply_clahe(image):
    """Apply CLAHE to green channel"""
    green = image[:, :, 1]
    h, w = green.shape
    tile_h = max(h // 8, 8)
    tile_w = max(w // 8, 8)

    green_norm = green.astype(np.float32) / 255.0

    clahe = equalize_adapthist(
        green_norm,
        kernel_size=(tile_h, tile_w),
        clip_limit=0.02,
        nbins=256
    )

    return clahe.astype(np.float32)


def generate_features_single_image(image_path):
    """
    Generate 9-channel feature map for one image

    Channels:
        0: CLAHE-enhanced green
        1-8: ABC fractional derivatives (v = 0.6 to 1.3)

    Args:
        image_path: Path to RGB image

    Returns:
        features: [H, W, 9] float32 array, normalized to [0, 1]
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load: {image_path}")

    h, w = image.shape[:2]
    features = np.zeros((h, w, 9), dtype=np.float32)

    # Channel 0: CLAHE-enhanced green
    features[:, :, 0] = apply_clahe(image)

    # Channels 1-8: ABC fractional derivatives
    for i, v in enumerate(FRACTIONAL_ORDERS):
        features[:, :, i + 1] = vessel_enhancement(image, v, GAUSSIAN_SIGMA)

    return features


def generate_all_features(dataset_dir, output_dir, resume=True):
    """
    Generate features for entire dataset

    Args:
        dataset_dir: Path to Dataset folder
        output_dir: Path to output folder for features
        resume: If True, skip already processed images
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted((dataset_dir / "images").glob("*.png"))

    print("ABC Fractional Feature Generation")
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_dir}")
    print(f"Images: {len(images)}")
    print(f"Fractional orders: {FRACTIONAL_ORDERS}")
    print(f"Total channels: 9 (1 CLAHE + 8 ABC)")

    successful = 0
    skipped = 0
    failed = 0
    failed_list = []

    start_time = time.time()

    for img_path in tqdm(images, desc="Generating features"):
        output_path = output_dir / f"{img_path.stem}_features.npy"

        if resume and output_path.exists():
            skipped += 1
            continue

        try:
            features = generate_features_single_image(img_path)
            np.save(output_path, features)
            successful += 1

        except Exception as e:
            failed += 1
            failed_list.append((img_path.name, str(e)))

    elapsed = time.time() - start_time

    print(f"\nComplete!")
    print(f"Successful: {successful}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Time: {elapsed:.1f}s ({(successful + skipped) / elapsed:.2f} images/s)")

    if failed > 0:
        print(f"\nFailed images:")
        for name, error in failed_list[:10]:
            print(f"  {name}: {error}")


def main():
    parser = argparse.ArgumentParser(description='Generate ABC fractional features')
    parser.add_argument('--dataset_dir', type=str, default='Dataset', help='Dataset directory')
    parser.add_argument('--output_dir', type=str, default='Dataset/multiscale_features', help='Output directory')
    parser.add_argument('--no_resume', action='store_true', help='Regenerate all features')

    args = parser.parse_args()

    generate_all_features(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        resume=not args.no_resume
    )


if __name__ == "__main__":
    main()
