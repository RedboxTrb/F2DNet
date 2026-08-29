"""R1-C10 + R1-C11 - are the eight ABC fractional orders redundant?

R1-C11: "The use of eight closely spaced fractional orders may introduce substantial
redundancy." The orders are [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3], spaced 0.1 apart, so the
concern is reasonable and needs a quantitative answer rather than a defence.

Four measurements on FIVES test images (which carry vessel ground truth), all inside the FOV:

  1. pairwise Pearson correlation between the eight vesselness maps
  2. PCA over the 8-dimensional per-pixel feature vector - how many components carry the
     variance, i.e. the effective dimensionality of the bank
  3. per-order vessel-detection ROC-AUC against ground truth - what each order is worth alone
  4. greedy forward selection - the smallest subset whose combined AUC matches all eight,
     using a logistic combination fitted on held-out pixels

R1-C10 (reproducibility) is answered by the exact constants this script prints: the closed-form
ABC kernel coefficients per order, the Gaussian sigma, and the Hessian construction.
"""
import csv
import json
import os
import sys
import time

sys.path.insert(0, r'D:\lodo\code')

import cv2
import numpy as np
import generate_abc_features as G

OUT = r'D:\Mukesh Delu\VDMDR\comments_report'
F5 = r'D:\Mukesh Delu\Dataset\FIVES\FIVES A Fundus Image Dataset for AI-based Vessel Segmentation\test'
N_IMAGES = 200
PIX_PER_IMAGE = 6000          # random FOV pixels sampled per image for the pooled statistics
SEED = 0


def roc_auc(y, s):
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    if y.all() or not y.any():
        return float('nan')
    o = np.argsort(s)
    r = np.empty(len(s), float)
    ss = s[o]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    npos, nneg = y.sum(), (~y).sum()
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def fit_logistic(X, y, iters=250, lr=0.5):
    """Tiny logistic regression - avoids a sklearn dependency on this box."""
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    Xs = np.hstack([Xs, np.ones((len(Xs), 1))])
    w = np.zeros(Xs.shape[1])
    yv = y.astype(np.float64)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Xs @ w))
        w -= lr * (Xs.T @ (p - yv)) / len(yv)
    return w, X.mean(0), X.std(0)


def apply_logistic(X, w, mu, sd):
    Xs = (X - mu) / (sd + 1e-9)
    Xs = np.hstack([Xs, np.ones((len(Xs), 1))])
    return Xs @ w


def main():
    orders = list(G.FRACTIONAL_ORDERS)
    sigma = G.GAUSSIAN_SIGMA
    print('orders: %s   gaussian sigma: %s' % (orders, sigma))

    print('\n=== R1-C10: exact ABC kernel coefficients (closed form) ===')
    coeff = []
    for v in orders:
        kx, ky = G.abc_fractional_kernels(v)
        p, q, r = kx[0, 0], kx[1, 0], kx[2, 0]
        print('  v=%.1f   p=%+.6f  q=%+.6f  r=%+.6f' % (v, p, q, r))
        coeff.append(dict(order=v, p=round(float(p), 8), q=round(float(q), 8), r=round(float(r), 8)))

    idir, gdir = os.path.join(F5, 'Original'), os.path.join(F5, 'Ground truth')
    gts = {os.path.splitext(f)[0]: os.path.join(gdir, f) for f in os.listdir(gdir)}
    files = sorted(f for f in os.listdir(idir) if os.path.splitext(f)[0] in gts)
    files = files[::max(1, len(files) // N_IMAGES)][:N_IMAGES]
    print('\n%d FIVES-test images' % len(files))

    rng = np.random.default_rng(SEED)
    X_all, y_all = [], []
    per_image_corr = []
    t0 = time.time()
    for i, fn in enumerate(files):
        bgr = cv2.imread(os.path.join(idir, fn), cv2.IMREAD_COLOR)
        gt = cv2.imread(gts[os.path.splitext(fn)[0]], cv2.IMREAD_GRAYSCALE)
        if bgr is None or gt is None:
            continue
        bgr = cv2.resize(bgr, (1024, 1024))
        gt = cv2.resize(gt, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        fov = G.create_fov_mask(rgb) > 0
        vessel = gt > 127

        maps = np.stack([G.vessel_enhancement(rgb, v, sigma) for v in orders], axis=-1)

        f = maps[fov]                                     # [n_fov, 8]
        c = np.corrcoef(f.T)
        per_image_corr.append(c)

        idx = rng.choice(f.shape[0], size=min(PIX_PER_IMAGE, f.shape[0]), replace=False)
        X_all.append(f[idx])
        y_all.append(vessel[fov][idx])
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print('  %d/%d  %.0fs  eta %.1f min' % (i + 1, len(files), el,
                                                    (len(files) - i - 1) * el / (i + 1) / 60), flush=True)

    X = np.vstack(X_all).astype(np.float64)
    y = np.concatenate(y_all)
    C = np.mean(per_image_corr, axis=0)
    print('\npooled sample: %d pixels, %.1f%% vessel' % (len(X), 100.0 * y.mean()))

    print('\n=== 1. pairwise Pearson correlation between orders (mean over images) ===')
    print('        ' + ''.join('%8.1f' % v for v in orders))
    for a, v in enumerate(orders):
        print('  %4.1f  ' % v + ''.join('%8.4f' % C[a, b] for b in range(len(orders))))
    off = C[~np.eye(len(orders), dtype=bool)]
    adj = [C[i, i + 1] for i in range(len(orders) - 1)]
    print('\n  mean off-diagonal correlation : %.4f' % off.mean())
    print('  mean adjacent-order (0.1 apart): %.4f' % np.mean(adj))
    print('  min correlation anywhere       : %.4f  (orders %.1f vs %.1f)'
          % (C[~np.eye(len(orders), dtype=bool)].min(),
             orders[np.unravel_index(np.argmin(C + np.eye(len(orders)) * 9), C.shape)[0]],
             orders[np.unravel_index(np.argmin(C + np.eye(len(orders)) * 9), C.shape)[1]]))

    print('\n=== 2. PCA - effective dimensionality of the 8-order bank ===')
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-9)
    ev = np.linalg.eigvalsh(np.cov(Xc.T))[::-1]
    ev = np.clip(ev, 0, None)
    frac = ev / ev.sum()
    cum = np.cumsum(frac)
    for k in range(len(orders)):
        print('  PC%d  variance %6.3f%%   cumulative %7.3f%%' % (k + 1, 100 * frac[k], 100 * cum[k]))
    n95 = int(np.searchsorted(cum, 0.95) + 1)
    n99 = int(np.searchsorted(cum, 0.99) + 1)
    part = float(np.exp(-(frac * np.log(frac + 1e-12)).sum()))
    print('  components for 95%% variance: %d of 8' % n95)
    print('  components for 99%% variance: %d of 8' % n99)
    print('  participation ratio (effective rank): %.2f of 8' % part)

    print('\n=== 3. per-order vessel-detection ROC-AUC (alone) ===')
    aucs = []
    for a, v in enumerate(orders):
        s = roc_auc(y, X[:, a])
        aucs.append(s)
        print('  v=%.1f   AUC %.4f' % (v, s))
    print('  best single order: v=%.1f (AUC %.4f)' % (orders[int(np.argmax(aucs))], max(aucs)))

    print('\n=== 4. greedy forward selection (logistic combination) ===')
    n = len(X)
    tr = rng.random(n) < 0.5
    chosen, remaining, hist = [], list(range(len(orders))), []
    while remaining:
        best = None
        for c in remaining:
            cols = chosen + [c]
            w, mu, sd = fit_logistic(X[tr][:, cols], y[tr])
            a = roc_auc(y[~tr], apply_logistic(X[~tr][:, cols], w, mu, sd))
            if best is None or a > best[0]:
                best = (a, c)
        chosen.append(best[1])
        remaining.remove(best[1])
        hist.append((len(chosen), orders[best[1]], best[0]))
        print('  k=%d  add v=%.1f  ->  held-out AUC %.4f' % (len(chosen), orders[best[1]], best[0]))
    full = hist[-1][2]
    enough = next((k for k, _, a in hist if a >= full - 0.001), len(orders))
    print('\n  all 8 orders          : AUC %.4f' % full)
    print('  first k within 0.001  : k=%d' % enough)
    print('  -> %d of the 8 orders reproduce the full bank to within 0.001 AUC' % enough)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'order_redundancy_corr.csv'), 'w', newline='', encoding='utf-8') as f:
        w_ = csv.writer(f)
        w_.writerow([''] + orders)
        for a, v in enumerate(orders):
            w_.writerow([v] + [round(float(C[a, b]), 5) for b in range(len(orders))])
    with open(os.path.join(OUT, 'order_redundancy_summary.json'), 'w', encoding='utf-8') as f:
        json.dump({'orders': orders, 'sigma': sigma, 'kernel_coefficients': coeff,
                   'n_images': len(files), 'n_pixels': int(len(X)),
                   'mean_offdiag_corr': round(float(off.mean()), 5),
                   'mean_adjacent_corr': round(float(np.mean(adj)), 5),
                   'pca_var_fraction': [round(float(x), 6) for x in frac],
                   'components_95': n95, 'components_99': n99,
                   'participation_ratio': round(part, 4),
                   'per_order_auc': {str(o): round(float(a), 5) for o, a in zip(orders, aucs)},
                   'greedy': [{'k': k, 'order': o, 'auc': round(float(a), 5)} for k, o, a in hist],
                   'k_within_0.001_of_full': enough}, f, indent=2)
    print('\nwrote order_redundancy_corr.csv and order_redundancy_summary.json')


if __name__ == '__main__':
    main()
