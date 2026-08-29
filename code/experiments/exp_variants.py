"""Matched-protocol experiments for R1-C13, R1-C11 and R2-C6.

One code path, one split, one schedule - only the variable under test changes. That is the whole
point: R1-C13 asks whether the Table 6 baselines were trained identically, so here they provably
are.

  --arch    fhau | unet | resnet34 | densenet121      (R1-C13 matched baselines)
  --orders  8 | 4 | 2                                  (R1-C11 order ablation)
  --seed    N                                          (R2-C6 multiple seeds)

Feature layout on disk is [H,W,9] = 8 ABC orders (v=0.6..1.3) then CLAHE. The green channel is
prepended by the loader, so 8 orders -> 10 input channels, 4 -> 6, 2 -> 4.
Order subsets are chosen to span the range rather than cluster, since the redundancy analysis
showed the span carries more information than the density.
"""
import argparse, csv, json, os, random, sys, time

sys.path.insert(0, r'D:\lodo\code')

import cv2, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model_multiscale import AttentionUNetMultiscale, ConvBlock, AttentionGate

ROOT = r'D:\lodo'
DATA = os.path.join(ROOT, 'Dataset')
FEATS = os.path.join(DATA, 'multiscale_features')
OUT_CSV = r'D:\Mukesh Delu\VDMDR\comments_report'

ORDER_SUBSETS = {8: [0, 1, 2, 3, 4, 5, 6, 7], 4: [0, 2, 5, 7], 2: [0, 7]}   # + CLAHE (idx 8)
CLAHE_IDX = 8


# ---------------------------------------------------------------- data
class VesselDS(Dataset):
    def __init__(self, stems, order_idx):
        self.stems = stems
        self.sel = list(order_idx) + [CLAHE_IDX]

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        s = self.stems[i]
        rgb = cv2.imread(os.path.join(DATA, 'images', s + '.png'), cv2.IMREAD_COLOR)
        green = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)[:, :, 1].astype(np.float32) / 255.0
        # mmap so only the selected channels are read from disk, not all nine
        mm = np.load(os.path.join(FEATS, s + '_features.npy'), mmap_mode='r')
        abc = np.ascontiguousarray(mm[:, :, self.sel]).astype(np.float32)
        x = np.concatenate([green[..., None], abc], axis=-1)
        gt = (cv2.imread(os.path.join(DATA, 'ground_truth', s + '_gt.png'), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        mk = (cv2.imread(os.path.join(DATA, 'masks', s + '_mask.png'), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        return (torch.from_numpy(x).permute(2, 0, 1).float(),
                torch.from_numpy(gt).unsqueeze(0).float(),
                torch.from_numpy(mk).unsqueeze(0).float())


# ---------------------------------------------------------------- models
class PlainUNet(nn.Module):
    """FHAU-Net with the attention gates removed - isolates what the gates contribute."""

    def __init__(self, in_channels):
        super().__init__()
        self.e1, self.e2 = ConvBlock(in_channels, 64), ConvBlock(64, 128)
        self.e3, self.e4 = ConvBlock(128, 256), ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.b = ConvBlock(512, 1024)
        self.u4, self.d4 = nn.ConvTranspose2d(1024, 512, 2, 2), ConvBlock(1024, 512)
        self.u3, self.d3 = nn.ConvTranspose2d(512, 256, 2, 2), ConvBlock(512, 256)
        self.u2, self.d2 = nn.ConvTranspose2d(256, 128, 2, 2), ConvBlock(256, 128)
        self.u1, self.d1 = nn.ConvTranspose2d(128, 64, 2, 2), ConvBlock(128, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2)); e4 = self.e4(self.pool(e3))
        b = self.b(self.pool(e4))
        d = self.d4(torch.cat([self.u4(b), e4], 1))
        d = self.d3(torch.cat([self.u3(d), e3], 1))
        d = self.d2(torch.cat([self.u2(d), e2], 1))
        d = self.d1(torch.cat([self.u1(d), e1], 1))
        return self.out(d)


class EncoderUNet(nn.Module):
    def __init__(self, encoder, in_channels):
        super().__init__()
        import timm
        self.enc = timm.create_model(encoder, pretrained=False, features_only=True, in_chans=in_channels)
        ch = self.enc.feature_info.channels()
        self.ups, self.decs = nn.ModuleList(), nn.ModuleList()
        prev = ch[-1]
        for skip in reversed(ch[:-1]):
            self.ups.append(nn.ConvTranspose2d(prev, skip, 2, 2))
            self.decs.append(ConvBlock(skip * 2, skip)); prev = skip
        self.fu, self.fd = nn.ConvTranspose2d(prev, 64, 2, 2), ConvBlock(64, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        f = self.enc(x); y = f[-1]
        for up, dec, skip in zip(self.ups, self.decs, reversed(f[:-1])):
            y = up(y)
            if y.shape[-2:] != skip.shape[-2:]:
                y = nn.functional.interpolate(y, size=skip.shape[-2:], mode='nearest')
            y = dec(torch.cat([y, skip], 1))
        return self.out(self.fd(self.fu(y)))


def build(arch, in_ch):
    if arch == 'fhau':        return AttentionUNetMultiscale(in_channels=in_ch, out_channels=1)
    if arch == 'unet':        return PlainUNet(in_ch)
    if arch == 'resnet34':    return EncoderUNet('resnet34', in_ch)
    if arch == 'densenet121': return EncoderUNet('densenet121', in_ch)
    raise ValueError(arch)


# ---------------------------------------------------------------- loss (identical to the paper)
class CombinedLoss(nn.Module):
    def __init__(self, tw=0.7, fw=0.3, a=0.7, b=0.3, fa=0.25, fg=2.0):
        super().__init__(); self.tw, self.fw, self.a, self.b, self.fa, self.fg = tw, fw, a, b, fa, fg

    def forward(self, logits, target, mask=None):
        p = torch.sigmoid(logits)
        if mask is not None: p, target = p * mask, target * mask
        pf, tf = p.reshape(-1), target.reshape(-1)
        tp = (pf * tf).sum(); fp = (pf * (1 - tf)).sum(); fn = ((1 - pf) * tf).sum()
        tversky = 1 - (tp + 1.0) / (tp + self.a * fn + self.b * fp + 1.0)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none')
        pt = p * target + (1 - p) * (1 - target)
        at = self.fa * target + (1 - self.fa) * (1 - target)
        focal = (at * (1 - pt) ** self.fg * bce).mean()
        return self.tw * tversky + self.fw * focal


def dice_of(pred, gt, fov):
    pred, gt = pred[fov], gt[fov]
    tp = float(np.count_nonzero(pred & gt)); fp = float(np.count_nonzero(pred & ~gt))
    fn = float(np.count_nonzero(~pred & gt))
    return 2 * tp / (2 * tp + fp + fn + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='fhau', choices=['fhau', 'unet', 'resnet34', 'densenet121'])
    ap.add_argument('--orders', type=int, default=8, choices=[2, 4, 8])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--tag', default=None)
    a = ap.parse_args()

    tag = a.tag or ('%s_o%d_s%d' % (a.arch, a.orders, a.seed))
    outdir = os.path.join(ROOT, 'variants', tag); os.makedirs(outdir, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)

    sp = os.path.join(ROOT, 'splits', 'main')
    tr = [l.strip() for l in open(os.path.join(sp, 'train.txt'), encoding='utf-8') if l.strip()]
    va = [l.strip() for l in open(os.path.join(sp, 'val.txt'), encoding='utf-8') if l.strip()]
    te = [l.strip() for l in open(os.path.join(sp, 'test.txt'), encoding='utf-8') if l.strip()]

    sel = ORDER_SUBSETS[a.orders]
    in_ch = 1 + len(sel) + 1
    dev = torch.device('cuda')
    model = build(a.arch, in_ch).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print('%s | orders=%d | in_ch=%d | seed=%d | params %.3f M | train %d val %d test %d'
          % (a.arch, a.orders, in_ch, a.seed, nparam / 1e6, len(tr), len(va), len(te)), flush=True)

    dl_tr = DataLoader(VesselDS(tr, sel), batch_size=5, shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True, drop_last=True)
    dl_va = DataLoader(VesselDS(va, sel), batch_size=5, shuffle=False, num_workers=8, persistent_workers=True)
    dl_te = DataLoader(VesselDS(te, sel), batch_size=1, shuffle=False, num_workers=4)

    crit = CombinedLoss()
    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=7)
    scaler = torch.amp.GradScaler('cuda')

    best, t0 = 0.0, time.time()
    for ep in range(a.epochs):
        model.train(); opt.zero_grad()
        for i, (x, y, m) in enumerate(dl_tr):
            x, y, m = x.to(dev), y.to(dev), m.to(dev)
            with torch.amp.autocast('cuda'):
                loss = crit(model(x), y, m) / 4
            scaler.scale(loss).backward()
            if (i + 1) % 4 == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad()
        model.eval(); vl, ds = 0.0, []
        with torch.no_grad():
            for x, y, m in dl_va:
                x, y, m = x.to(dev), y.to(dev), m.to(dev)
                with torch.amp.autocast('cuda'):
                    lg = model(x); vl += crit(lg, y, m).item()
                p = (torch.sigmoid(lg) > 0.5).cpu().numpy()
                for k in range(p.shape[0]):
                    ds.append(dice_of(p[k, 0].astype(bool), y[k, 0].cpu().numpy() > 0.5,
                                      m[k, 0].cpu().numpy() > 0.5))
        vd = float(np.mean(ds)); sch.step(vl / max(len(dl_va), 1))
        print('  epoch %2d/%d  val Dice %.4f  (%.1f min)' % (ep + 1, a.epochs, vd, (time.time() - t0) / 60), flush=True)
        if vd > best:
            best = vd
            torch.save({'model_state_dict': model.state_dict(), 'val_dice': vd, 'epoch': ep + 1,
                        'arch': a.arch, 'orders': a.orders, 'seed': a.seed, 'in_ch': in_ch},
                       os.path.join(outdir, 'best.pth'))

    # held-out test
    model.load_state_dict(torch.load(os.path.join(outdir, 'best.pth'), map_location=dev,
                                     weights_only=False)['model_state_dict'])
    model.eval(); tds = []
    with torch.no_grad():
        for x, y, m in dl_te:
            x = x.to(dev)
            with torch.amp.autocast('cuda'):
                p = (torch.sigmoid(model(x)) > 0.5).cpu().numpy()
            tds.append(dice_of(p[0, 0].astype(bool), y[0, 0].numpy() > 0.5, m[0, 0].numpy() > 0.5))
    rng = np.random.default_rng(0); arr = np.asarray(tds)
    bs = arr[rng.integers(0, len(arr), size=(2000, len(arr)))].mean(1)
    row = dict(tag=tag, arch=a.arch, orders=a.orders, seed=a.seed, epochs=a.epochs,
               in_channels=in_ch, params_M=round(nparam / 1e6, 3),
               val_dice=round(best, 4), test_dice=round(float(arr.mean()), 4),
               test_ci_lo=round(float(np.percentile(bs, 2.5)), 4),
               test_ci_hi=round(float(np.percentile(bs, 97.5)), 4),
               n_test=len(arr), minutes=round((time.time() - t0) / 60, 1))
    print('RESULT %s' % json.dumps(row), flush=True)
    os.makedirs(OUT_CSV, exist_ok=True)
    f = os.path.join(OUT_CSV, 'variant_results.csv')
    new = not os.path.exists(f)
    with open(f, 'a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new: w.writeheader()
        w.writerow(row)


if __name__ == '__main__':
    main()
