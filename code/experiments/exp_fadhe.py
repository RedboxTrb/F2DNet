"""X4 + flux-form study (R1-C6, R1-C7, R1-C8, R2-C7).

Two questions, one pass over FIVES test images (which carry vessel ground truth):

(1) COMPONENT ABLATION - what does each FADHE stage actually contribute? Measured directly on
    the enhanced image with reference-based contrast metrics, using the GT vessel mask to
    separate vessel from background pixels. This deliberately does NOT feed the variants to the
    vessel network: that checkpoint was trained on RAW green, so scoring FADHE-enhanced input
    through it would measure distribution shift, not enhancement quality.

(2) FLUX FORM vs the implemented ONE-SIDED update (FINDINGS.md section 7). The shipped code is
        update = dt * (Fx*c + Fy*c)
    with Fx, Fy one-sided forward Grunwald-Letnikov differences - an explicit, non-conservative
    fractional analogue of the Perona-Malik GRADIENT update, not a discretisation of
    div(c grad I). Here the symmetric four-direction flux form is implemented alongside it and
    the two outputs are compared, so the revision can state what the simplification costs.

All metrics are computed inside the FOV mask only.
"""
import csv
import os
import sys
import time

sys.path.insert(0, r'D:\Mukesh Delu\VDMDR')

import cv2
import numpy as np
from preprocessing.green_channel_enhancer import GreenChannelEnhancer

OUT = r'D:\Mukesh Delu\VDMDR\comments_report'
F5 = r'D:\Mukesh Delu\Dataset\FIVES\FIVES A Fundus Image Dataset for AI-based Vessel Segmentation\test'
SIZE = 1024
N_IMAGES = 200        # full FIVES official test split


def gl_weights(alpha, mem):
    w = np.zeros(mem + 1)
    w[0] = 1.0
    for i in range(mem):
        w[i + 1] = w[i] * (1 - (alpha + 1) / (i + 1))
    return w


def diffuse(I, mask, form, alpha=0.2, K=0.05, dt=0.1, n_iter=20, mem=10):
    """form='onesided' reproduces the shipped code exactly.
    form='flux' is the symmetric four-direction conservative analogue."""
    w = gl_weights(alpha, mem)
    P = np.pad(I, mem, mode='symmetric')
    M, N = I.shape
    c0 = c1 = mem
    for _ in range(n_iter):
        core = P[c0:c0 + M, c1:c1 + N]
        if form == 'onesided':
            Fx = np.zeros((M, N))
            Fy = np.zeros((M, N))
            for k in range(mem):
                Fx += w[k + 1] * (P[c0 + k + 1:c0 + M + k + 1, c1:c1 + N] - core)
                Fy += w[k + 1] * (P[c0:c0 + M, c1 + k + 1:c1 + N + k + 1] - core)
            c = np.exp(-(Fx ** 2 + Fy ** 2) / (K ** 2))
            upd = dt * (Fx * c + Fy * c)
        else:
            # independent GL difference and diffusivity per direction, then summed
            dirs = []
            for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                F = np.zeros((M, N))
                for k in range(mem):
                    o = k + 1
                    F += w[o] * (P[c0 + sx * o:c0 + M + sx * o, c1 + sy * o:c1 + N + sy * o] - core)
                dirs.append(F)
            upd = dt * sum(F * np.exp(-(F ** 2) / (K ** 2)) for F in dirs)
        P[c0:c0 + M, c1:c1 + N] += upd * mask
    return P[c0:c0 + M, c1:c1 + N]


def clahe(I, mask):
    out = cv2.createCLAHE(clipLimit=0.01 * 255, tileGridSize=(8, 8)).apply((I * 255).astype(np.uint8))
    return (out * mask).astype(np.float32) / 255.0


def bilateral(I, mask):
    out = cv2.bilateralFilter((I * 255).astype(np.uint8), d=5, sigmaColor=12.75, sigmaSpace=3)
    return (out * mask).astype(np.float32) / 255.0


def unsharp(I, mask):
    u = (I * 255).astype(np.uint8)
    s = cv2.addWeighted(u, 1.6, cv2.GaussianBlur(u, (0, 0), sigmaX=1.0), -0.6, 0)
    return (np.clip(s * mask, 0, 255)).astype(np.float32) / 255.0


def quality(I, vessel, fov):
    """Contrast of vessel against background, inside the FOV."""
    v = I[vessel & fov]
    b = I[(~vessel) & fov]
    if v.size < 10 or b.size < 10:
        return None
    cnr = abs(v.mean() - b.mean()) / np.sqrt(0.5 * (v.var() + b.var()) + 1e-9)
    mich = abs(v.mean() - b.mean()) / (v.mean() + b.mean() + 1e-9)
    x = I[fov]
    h, _ = np.histogram((x * 255).astype(np.uint8), bins=256, range=(0, 255))
    p = h / max(h.sum(), 1)
    p = p[p > 0]
    return dict(cnr=float(cnr), michelson=float(mich),
                entropy=float(-(p * np.log2(p)).sum()), std=float(x.std()))


def ssim(a, b):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    sa = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a ** 2
    sb = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b ** 2
    sab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (sa + sb + C2) + 1e-12)
    return float(m.mean())


def boot(vals, n=2000, seed=0):
    v = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    mm = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return v.mean(), v.std(ddof=1), np.percentile(mm, 2.5), np.percentile(mm, 97.5)


VARIANT_ORDER = ['A_raw_green', 'B_clahe_only', 'C_frac_onesided', 'D_frac_flux',
                 'E_frac_clahe', 'F_frac_clahe_bilat', 'G_FULL_FADHE', 'H_FULL_FADHE_fluxform']


def main():
    enh = GreenChannelEnhancer()
    idir = os.path.join(F5, 'Original')
    gdir = os.path.join(F5, 'Ground truth')
    gts = {os.path.splitext(f)[0]: os.path.join(gdir, f) for f in os.listdir(gdir)}
    files = sorted(f for f in os.listdir(idir) if os.path.splitext(f)[0] in gts)
    files = files[::max(1, len(files) // N_IMAGES)][:N_IMAGES]
    print('%d FIVES-test images at %dx%d' % (len(files), SIZE, SIZE), flush=True)

    rows = []
    pairs = []
    t0 = time.time()
    for i, fn in enumerate(files):
        bgr = cv2.imread(os.path.join(idir, fn), cv2.IMREAD_COLOR)
        gt = cv2.imread(gts[os.path.splitext(fn)[0]], cv2.IMREAD_GRAYSCALE)
        if bgr is None or gt is None:
            continue
        bgr = cv2.resize(bgr, (SIZE, SIZE))
        gt = cv2.resize(gt, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        green = rgb[:, :, 1]
        mask = enh.create_fundus_mask(green)
        fov = mask > 0.5
        vessel = gt > 127
        g = green.astype(np.float32) / 255.0

        d_one = diffuse(g, mask, 'onesided')
        d_flx = diffuse(g, mask, 'flux')

        variants = {
            'A_raw_green': g * mask,
            'B_clahe_only': clahe(g, mask),
            'C_frac_onesided': d_one * mask,
            'D_frac_flux': d_flx * mask,
            'E_frac_clahe': clahe(d_one, mask),
            'F_frac_clahe_bilat': bilateral(clahe(d_one, mask), mask),
            'G_FULL_FADHE': unsharp(bilateral(clahe(d_one, mask), mask), mask),
            'H_FULL_FADHE_fluxform': unsharp(bilateral(clahe(d_flx, mask), mask), mask),
        }
        for name, img in variants.items():
            q = quality(img, vessel, fov)
            if q:
                rows.append(dict(image=fn, variant=name,
                                 **{k: round(v, 6) for k, v in q.items()}))

        full_a = variants['G_FULL_FADHE']
        full_b = variants['H_FULL_FADHE_fluxform']
        pairs.append(dict(
            image=fn,
            ssim_frac_onesided_vs_flux=round(ssim(variants['C_frac_onesided'], variants['D_frac_flux']), 6),
            ssim_full_onesided_vs_flux=round(ssim(full_a, full_b), 6),
            pearson_full=round(float(np.corrcoef(full_a[fov], full_b[fov])[0, 1]), 6),
            maxabs_full=round(float(np.abs(full_a - full_b)[fov].max()), 6),
        ))
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print('  %d/%d  %.0fs  eta %.1f min'
                  % (i + 1, len(files), el, (len(files) - i - 1) * el / (i + 1) / 60), flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'fadhe_ablation_per_image.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, 'fadhe_fluxform_pairs.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
        w.writeheader()
        w.writerows(pairs)

    print('\n=== FADHE component ablation (mean [95%% CI], n=%d images) ===' % len(pairs))
    summ = []
    for name in VARIANT_ORDER:
        sel = [r for r in rows if r['variant'] == name]
        if not sel:
            continue
        s = {'variant': name, 'n': len(sel)}
        line = '%-24s' % name
        for k in ('cnr', 'michelson', 'entropy'):
            m, sd, lo, hi = boot([r[k] for r in sel])
            s[k + '_mean'] = round(m, 4)
            s[k + '_sd'] = round(sd, 4)
            s[k + '_ci_lo'] = round(lo, 4)
            s[k + '_ci_hi'] = round(hi, 4)
            line += '  %s %7.4f [%7.4f,%7.4f]' % (k, m, lo, hi)
        summ.append(s)
        print(line, flush=True)

    with open(os.path.join(OUT, 'fadhe_ablation_summary.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)

    print('\n=== one-sided vs symmetric flux form ===')
    for k in ('ssim_frac_onesided_vs_flux', 'ssim_full_onesided_vs_flux', 'pearson_full', 'maxabs_full'):
        m, sd, lo, hi = boot([p[k] for p in pairs])
        print('  %-30s %.5f  sd %.5f  95%% CI [%.5f, %.5f]' % (k, m, sd, lo, hi))
    print('\nwrote fadhe_ablation_per_image.csv, fadhe_ablation_summary.csv, fadhe_fluxform_pairs.csv')


if __name__ == '__main__':
    main()
