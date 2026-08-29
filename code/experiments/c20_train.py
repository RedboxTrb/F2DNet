"""R1-C20 - the five matched ablation configurations the reviewer asked for.

  green_only        raw green channel, single stream
  fadhe_only        FADHE-enhanced green, single stream
  vessel_only       binary vessel map, single stream
  green_vessel      raw green + vessel map, dual stream
  clahe_vessel      conventional CLAHE enhancement + vessel map, dual stream

Everything else is held constant: same APTOS images, same stratified 70/15/15 split with seed 42,
same backbones, optimiser, schedule and loss. Only the input representation changes, which is the
whole point of the comment.

Backbones follow the published design: EfficientNet-B4 for the image stream, ResNet18 for the
vessel stream. Single-stream configurations use one backbone and the same classifier head, so
capacity differences between configurations are reported rather than hidden.
"""
import argparse, csv, json, os, random, time

import cv2, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader

C20 = r'D:\lodo\c20'
OUT_CSV = r'D:\Mukesh Delu\VDMDR\comments_report'
RUNS = r'D:\lodo\c20_runs'
GRADES = ['0', '1', '2', '3', '4']
SIZE = 512

# (image-stream folder or None, vessel stream?)
CONFIGS = {
    'green_only':   ('green', False),
    'fadhe_only':   ('fadhe', False),
    'vessel_only':  ('vessel', False),
    'green_vessel': ('green', True),
    'clahe_vessel': ('clahe', True),
    'fadhe_vessel': ('fadhe', True),   # the proposed configuration
}


def build_index():
    items = []
    for g in GRADES:
        d = os.path.join(C20, 'green', g)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith('.png'):
                items.append((g, os.path.splitext(f)[0], int(g)))
    return items


def stratified_split(items, seed=42):
    """70/15/15 stratified by grade, matching the protocol described for Table 4."""
    rng = random.Random(seed)
    by = {}
    for it in items:
        by.setdefault(it[2], []).append(it)
    tr, va, te = [], [], []
    for lbl in sorted(by):
        v = by[lbl][:]
        rng.shuffle(v)
        n = len(v); a = int(n * 0.70); b = int(n * 0.85)
        tr += v[:a]; va += v[a:b]; te += v[b:]
    return tr, va, te


class DS(Dataset):
    def __init__(self, items, img_dir, use_vessel, train=False):
        self.items, self.img_dir, self.use_vessel, self.train = items, img_dir, use_vessel, train

    def __len__(self):
        return len(self.items)

    def _load(self, folder, g, stem):
        p = os.path.join(C20, folder, g, stem + '.png')
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            im = np.zeros((SIZE, SIZE), np.uint8)
        if im.shape[:2] != (SIZE, SIZE):
            im = cv2.resize(im, (SIZE, SIZE))
        return im.astype(np.float32) / 255.0

    def __getitem__(self, i):
        g, stem, lbl = self.items[i]
        img = self._load(self.img_dir, g, stem)
        ves = self._load('vessel', g, stem) if self.use_vessel else None
        if self.train:                                   # identical augmentation for all configs
            if random.random() < 0.5:
                img = np.fliplr(img).copy()
                if ves is not None: ves = np.fliplr(ves).copy()
            if random.random() < 0.5:
                img = np.flipud(img).copy()
                if ves is not None: ves = np.flipud(ves).copy()
            k = random.randint(0, 3)
            if k:
                img = np.rot90(img, k).copy()
                if ves is not None: ves = np.rot90(ves, k).copy()
        img_t = torch.from_numpy((img - 0.5) / 0.5).unsqueeze(0).float()
        if ves is None:
            return img_t, torch.tensor(lbl)
        ves_t = torch.from_numpy((ves - 0.5) / 0.5).unsqueeze(0).float()
        return img_t, ves_t, torch.tensor(lbl)


class SingleStream(nn.Module):
    def __init__(self, n=5):
        super().__init__()
        import timm
        self.bb = timm.create_model('efficientnet_b4', pretrained=False, num_classes=0, in_chans=1)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.bb.num_features, n))

    def forward(self, x):
        return self.head(self.bb(x))


class DualStream(nn.Module):
    def __init__(self, n=5):
        super().__init__()
        import timm
        self.img = timm.create_model('efficientnet_b4', pretrained=False, num_classes=0, in_chans=1)
        self.ves = timm.create_model('resnet18', pretrained=False, num_classes=0, in_chans=1)
        d = self.img.num_features + self.ves.num_features
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(d, 512), nn.BatchNorm1d(512),
                                  nn.GELU(), nn.Dropout(0.2), nn.Linear(512, n))

    def forward(self, a, b):
        return self.head(torch.cat([self.img(a), self.ves(b)], 1))


def qwk(a, b, k=5):
    a, b = np.asarray(a, int), np.asarray(b, int)
    O = np.zeros((k, k))
    for x, y in zip(a, b): O[x, y] += 1
    W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], float) / (k - 1) ** 2
    E = np.outer(np.bincount(a, minlength=k), np.bincount(b, minlength=k)).astype(float)
    if E.sum() == 0: return float('nan')
    E *= O.sum() / E.sum()
    den = (W * E).sum()
    return float(1 - (W * O).sum() / den) if den else float('nan')


def evaluate(model, loader, dual, dev):
    model.eval(); P, Y = [], []
    with torch.no_grad():
        for batch in loader:
            if dual:
                a, b, y = batch
                with torch.amp.autocast('cuda'):
                    lg = model(a.to(dev), b.to(dev))
            else:
                a, y = batch
                with torch.amp.autocast('cuda'):
                    lg = model(a.to(dev))
            P += lg.argmax(1).cpu().tolist(); Y += y.tolist()
    P, Y = np.array(P), np.array(Y)
    return dict(acc=float((P == Y).mean()), qwk=qwk(Y, P),
                off1=float((np.abs(P - Y) <= 1).mean()),
                refer_sens=float((P[Y >= 2] >= 2).mean()) if (Y >= 2).any() else float('nan'),
                refer_spec=float((P[Y < 2] < 2).mean()) if (Y < 2).any() else float('nan')), P, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, choices=list(CONFIGS))
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--batch', type=int, default=16)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    img_dir, dual = CONFIGS[a.config]
    dev = torch.device('cuda')
    outdir = os.path.join(RUNS, a.config); os.makedirs(outdir, exist_ok=True)

    items = build_index()
    tr, va, te = stratified_split(items, a.seed)
    print('%s | image=%s | vessel=%s | train %d val %d test %d'
          % (a.config, img_dir, dual, len(tr), len(va), len(te)), flush=True)

    mk = lambda it, train: DS(it, img_dir, dual, train)
    dl_tr = DataLoader(mk(tr, True), batch_size=a.batch, shuffle=True, num_workers=6,
                       persistent_workers=True, pin_memory=True, drop_last=True)
    dl_va = DataLoader(mk(va, False), batch_size=a.batch, shuffle=False, num_workers=6, persistent_workers=True)
    dl_te = DataLoader(mk(te, False), batch_size=a.batch, shuffle=False, num_workers=6)

    model = (DualStream() if dual else SingleStream()).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    cnt = np.bincount([x[2] for x in tr], minlength=5).astype(np.float64)
    w = torch.tensor((cnt.sum() / (5 * np.maximum(cnt, 1))), dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
    scaler = torch.amp.GradScaler('cuda')
    print('  params %.2f M | class weights %s' % (nparam / 1e6, np.round(w.cpu().numpy(), 2)), flush=True)

    best, t0 = -1.0, time.time()
    for ep in range(a.epochs):
        model.train()
        for batch in dl_tr:
            opt.zero_grad()
            if dual:
                x1, x2, y = batch
                with torch.amp.autocast('cuda'):
                    loss = crit(model(x1.to(dev), x2.to(dev)), y.to(dev))
            else:
                x1, y = batch
                with torch.amp.autocast('cuda'):
                    loss = crit(model(x1.to(dev)), y.to(dev))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
        m, _, _ = evaluate(model, dl_va, dual, dev)
        print('  epoch %2d/%d  val acc %.4f  qwk %.4f  (%.1f min)'
              % (ep + 1, a.epochs, m['acc'], m['qwk'], (time.time() - t0) / 60), flush=True)
        if m['qwk'] > best:
            best = m['qwk']
            torch.save({'model_state_dict': model.state_dict(), 'val': m, 'epoch': ep + 1},
                       os.path.join(outdir, 'best.pth'))

    model.load_state_dict(torch.load(os.path.join(outdir, 'best.pth'), map_location=dev,
                                     weights_only=False)['model_state_dict'])
    tm, P, Y = evaluate(model, dl_te, dual, dev)
    row = dict(config=a.config, image_stream=img_dir, vessel_stream=dual, epochs=a.epochs,
               params_M=round(nparam / 1e6, 2), n_test=len(te),
               val_qwk=round(best, 4), **{k: round(v, 4) for k, v in tm.items()},
               minutes=round((time.time() - t0) / 60, 1))
    print('RESULT ' + json.dumps(row), flush=True)
    os.makedirs(OUT_CSV, exist_ok=True)
    f = os.path.join(OUT_CSV, 'c20_ablation.csv')
    new = not os.path.exists(f)
    with open(f, 'a', newline='', encoding='utf-8') as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new: w_.writeheader()
        w_.writerow(row)


if __name__ == '__main__':
    main()
