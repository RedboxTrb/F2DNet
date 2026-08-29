"""X5 (R1-C14, R1-C16) — per-dataset vessel segmentation metrics with bootstrap CIs.

Evaluates the existing Attention U-Net checkpoint on:
  FIVES test  (200 imgs, official split, in-domain)
  FIVES train (600 imgs, seen during training - reported to show the train/test gap)
  HRF         ( 45 imgs, never trained on - genuine external segmentation validation)

Metrics are computed at the ground truth's native resolution, matching what the deployed
pipeline does (predict at 1024^2, resize the binary mask back with nearest-neighbour).
HRF is scored inside its supplied FOV mask, which is standard for that dataset; scoring the
black surround would inflate specificity and accuracy toward 1.
"""
import csv, os, sys, time

os.environ.setdefault('MODEL_DEVICE', 'cuda')
sys.path.insert(0, r'C:\DrBackend')

import cv2, numpy as np, torch
from models.model_loader import ModelManager
from services.preprocessing import preprocess_for_vessel

OUT = r'D:\Mukesh Delu\VDMDR\comments_report'
F5  = r'D:\Mukesh Delu\Dataset\FIVES\FIVES A Fundus Image Dataset for AI-based Vessel Segmentation'
HRF = r'D:\Mukesh Delu\Dataset\HRF'
EXT = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}


def stem(p):
    return os.path.splitext(os.path.basename(p))[0]


def pair_dir(img_dir, gt_dir):
    gts = {stem(f): os.path.join(gt_dir, f) for f in os.listdir(gt_dir)
           if os.path.splitext(f)[1].lower() in EXT}
    out = []
    for f in sorted(os.listdir(img_dir)):
        if os.path.splitext(f)[1].lower() not in EXT:
            continue
        if stem(f) in gts:
            out.append((os.path.join(img_dir, f), gts[stem(f)], None))
    return out


def pair_hrf():
    imgs = os.path.join(HRF, 'images'); gt = os.path.join(HRF, 'manual1'); mk = os.path.join(HRF, 'mask')
    gts = {stem(f): os.path.join(gt, f) for f in os.listdir(gt)}
    mks = {stem(f): os.path.join(mk, f) for f in os.listdir(mk)}
    out = []
    for f in sorted(os.listdir(imgs)):
        s = stem(f)
        if s in gts:
            out.append((os.path.join(imgs, f), gts[s], mks.get(s)))
    return out


def metrics(pred, gt, fov):
    """pred/gt boolean at native resolution; fov boolean or None."""
    if fov is not None:
        pred, gt = pred[fov], gt[fov]
    else:
        pred, gt = pred.ravel(), gt.ravel()
    tp = float(np.count_nonzero(pred & gt))
    fp = float(np.count_nonzero(pred & ~gt))
    fn = float(np.count_nonzero(~pred & gt))
    tn = float(np.count_nonzero(~pred & ~gt))
    e = 1e-9
    return dict(
        dice=2 * tp / (2 * tp + fp + fn + e),
        iou=tp / (tp + fp + fn + e),
        sens=tp / (tp + fn + e),
        spec=tn / (tn + fp + e),
        acc=(tp + tn) / (tp + tn + fp + fn + e),
        prec=tp / (tp + fp + e),
    )


def boot_ci(vals, n=2000, seed=0):
    """Percentile bootstrap over images. Fixed seed so the number is reproducible."""
    v = np.asarray(vals, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(v.mean()), float(v.std(ddof=1)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    dev = torch.device(os.environ['MODEL_DEVICE'])
    mm = ModelManager(use_fp16=False)
    mm.load_all_models()
    print('vessel model loaded on', dev, flush=True)

    sets = {
        'FIVES-test':  pair_dir(os.path.join(F5, 'test', 'Original'),  os.path.join(F5, 'test', 'Ground truth')),
        'HRF':         pair_hrf(),
        'FIVES-train': pair_dir(os.path.join(F5, 'train', 'Original'), os.path.join(F5, 'train', 'Ground truth')),
    }

    per_image, summary = [], []
    for name, items in sets.items():
        print(f'\n{name}: {len(items)} paired images', flush=True)
        acc = {k: [] for k in ('dice', 'iou', 'sens', 'spec', 'acc', 'prec')}
        t0 = time.time()
        for i, (ip, gp, mp) in enumerate(items):
            bgr = cv2.imread(ip, cv2.IMREAD_COLOR)
            gt = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
            if bgr is None or gt is None:
                print('  !! unreadable', ip, flush=True); continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = gt.shape[:2]

            vt, _ = preprocess_for_vessel(rgb)
            with torch.no_grad():
                prob = torch.sigmoid(mm.vessel_model(vt.to(dev))).cpu().numpy().squeeze()
            pred = (prob > 0.5).astype(np.uint8)
            if pred.shape != (h, w):
                pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

            fov = None
            if mp:
                m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    if m.shape != (h, w):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    fov = m > 127

            r = metrics(pred.astype(bool), gt > 127, fov)
            for k in acc:
                acc[k].append(r[k])
            per_image.append(dict(dataset=name, image=os.path.basename(ip),
                                  **{k: round(r[k], 6) for k in acc}))
            if (i + 1) % 50 == 0:
                print(f'   {i+1}/{len(items)}  {time.time()-t0:.0f}s', flush=True)

        row = {'dataset': name, 'n_images': len(acc['dice'])}
        for k in acc:
            m, sd, lo, hi = boot_ci(acc[k])
            row[f'{k}_mean'] = round(m, 4); row[f'{k}_sd'] = round(sd, 4)
            row[f'{k}_ci_lo'] = round(lo, 4); row[f'{k}_ci_hi'] = round(hi, 4)
        summary.append(row)
        print(f"  Dice {row['dice_mean']:.4f} [{row['dice_ci_lo']:.4f}, {row['dice_ci_hi']:.4f}]  "
              f"IoU {row['iou_mean']:.4f}  Sens {row['sens_mean']:.4f}  Spec {row['spec_mean']:.4f}  "
              f"Acc {row['acc_mean']:.4f}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'seg_per_image.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(per_image[0].keys())); w.writeheader(); w.writerows(per_image)
    with open(os.path.join(OUT, 'seg_summary_ci.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
    print(f'\nwrote {OUT}\seg_per_image.csv and seg_summary_ci.csv', flush=True)


if __name__ == '__main__':
    main()
