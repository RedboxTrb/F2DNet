"""Score each LODO fold on its held-out dataset (R1-C1, R1-C2, R1-C16; R2-C3, R2-C4).

For every fold we report BOTH:
  in-domain  - best validation Dice reached during training (from the checkpoint)
  out-domain - Dice on the entire held-out dataset, which the model never saw

Reporting both is what makes a reduced-epoch schedule defensible: if in-domain performance is
high while out-of-domain is low, the gap is domain shift and not undertraining. Without that
control a reviewer can attribute the whole result to a short schedule.

Metrics are computed inside the supplied FOV mask, with 2,000-sample bootstrap CIs over images.
"""
import csv
import json
import os
import sys

sys.path.insert(0, r'D:\lodo\code')

import cv2
import numpy as np
import torch

from model_multiscale import AttentionUNetMultiscale

ROOT = r'D:\lodo'
DATA = os.path.join(ROOT, 'Dataset')
FEATS = os.path.join(DATA, 'multiscale_features')
SPLITS = os.path.join(ROOT, 'splits')
RUNS = os.path.join(ROOT, 'runs')
OUT = r'D:\Mukesh Delu\VDMDR\comments_report'
FOLDS = ['FIVES', 'HRF', 'CHASE', 'DRIVE', 'STARE']


def load_input(stem):
    rgb = cv2.imread(os.path.join(DATA, 'images', stem + '.png'), cv2.IMREAD_COLOR)
    if rgb is None:
        return None, None, None
    green = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)[:, :, 1].astype(np.float32) / 255.0
    abc = np.load(os.path.join(FEATS, stem + '_features.npy')).astype(np.float32)
    x = np.concatenate([green[..., None], abc], axis=-1)          # [H,W,10]
    gt = cv2.imread(os.path.join(DATA, 'ground_truth', stem + '_gt.png'), cv2.IMREAD_GRAYSCALE)
    mk = cv2.imread(os.path.join(DATA, 'masks', stem + '_mask.png'), cv2.IMREAD_GRAYSCALE)
    return x, (gt > 127 if gt is not None else None), (mk > 127 if mk is not None else None)


def metrics(pred, gt, fov):
    if fov is not None:
        pred, gt = pred[fov], gt[fov]
    else:
        pred, gt = pred.ravel(), gt.ravel()
    tp = float(np.count_nonzero(pred & gt))
    fp = float(np.count_nonzero(pred & ~gt))
    fn = float(np.count_nonzero(~pred & gt))
    tn = float(np.count_nonzero(~pred & ~gt))
    e = 1e-9
    return dict(dice=2 * tp / (2 * tp + fp + fn + e), iou=tp / (tp + fp + fn + e),
                sens=tp / (tp + fn + e), spec=tn / (tn + fp + e),
                acc=(tp + tn) / (tp + tn + fp + fn + e))


def boot(v, n=2000, seed=0):
    v = np.asarray(v, float)
    if len(v) < 2:
        return float(v.mean()), float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows, per_image = [], []
    print('%-7s %5s %24s %8s %8s %8s   %s' %
          ('holdout', 'n', 'out-of-domain Dice [95% CI]', 'IoU', 'Sens', 'Spec', 'in-domain val Dice'))
    for fold in FOLDS:
        ck_path = os.path.join(RUNS, 'holdout_' + fold, 'best_model.pth')
        if not os.path.exists(ck_path):
            print('%-7s  (no checkpoint yet)' % fold)
            continue
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        indomain = ck.get('dice')
        model = AttentionUNetMultiscale(in_channels=10, out_channels=1).to(dev)
        model.load_state_dict(ck['model_state_dict'])
        model.eval()

        stems = [l.strip() for l in open(os.path.join(SPLITS, 'holdout_' + fold, 'test.txt'),
                                         encoding='utf-8') if l.strip()]
        acc = {k: [] for k in ('dice', 'iou', 'sens', 'spec', 'acc')}
        for s in stems:
            x, gt, fov = load_input(s)
            if x is None or gt is None:
                continue
            t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(dev)
            with torch.no_grad():
                prob = torch.sigmoid(model(t)).cpu().numpy().squeeze()
            pred = prob > 0.5
            if pred.shape != gt.shape:
                pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
            r = metrics(pred, gt, fov)
            for k in acc:
                acc[k].append(r[k])
            per_image.append(dict(holdout=fold, image=s, **{k: round(r[k], 6) for k in acc}))

        if not acc['dice']:
            continue
        d, lo, hi = boot(acc['dice'])
        row = dict(holdout=fold, n=len(acc['dice']),
                   in_domain_val_dice=round(float(indomain), 4) if indomain else '',
                   dice=round(d, 4), dice_ci_lo=round(lo, 4), dice_ci_hi=round(hi, 4))
        for k in ('iou', 'sens', 'spec', 'acc'):
            row[k] = round(float(np.mean(acc[k])), 4)
        rows.append(row)
        print('%-7s %5d   %.4f [%.4f, %.4f] %8.4f %8.4f %8.4f   %.4f'
              % (fold, len(acc['dice']), d, lo, hi, row['iou'], row['sens'], row['spec'],
                 indomain or float('nan')))
        del model
        if dev.type == 'cuda':
            torch.cuda.empty_cache()

    if not rows:
        print('\nno folds finished yet')
        return
    macro = float(np.mean([r['dice'] for r in rows]))
    print('\n  macro-average out-of-domain Dice across folds: %.4f' % macro)
    gap = [r['in_domain_val_dice'] - r['dice'] for r in rows if r['in_domain_val_dice'] != '']
    if gap:
        print('  mean in-domain minus out-of-domain gap      : %+.4f' % float(np.mean(gap)))
        print('  -> in-domain performance is the control: a large gap with high in-domain Dice')
        print('     is domain shift, not undertraining.')

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'lodo_results.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, 'lodo_per_image.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
        w.writeheader()
        w.writerows(per_image)
    print('\nwrote lodo_results.csv and lodo_per_image.csv')


if __name__ == '__main__':
    main()
