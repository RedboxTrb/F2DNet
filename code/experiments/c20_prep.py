"""R1-C20 data preparation - the five input representations the reviewer asked for.

Requested configurations: raw green only, FADHE only, vessel map only, raw green + vessel map,
and conventional enhancement + vessel map. Four distinct image products are needed:

    green   raw green channel of the RGB fundus            (trivial)
    fadhe   FADHE-enhanced green                            (CPU, ~2.7 s/image - the long pole)
    clahe   conventional CLAHE-enhanced green               (trivial, the "conventional" baseline)
    vessel  binary vessel map from the deployed checkpoint  (GPU, separate step)

Everything is written at 512x512, matching Table 4's classification input size.
Run with --stage cpu (green/fadhe/clahe) or --stage vessel (GPU).
"""
import argparse, os, sys, time
import cv2, numpy as np

SRC = r'D:\Mukesh Delu\VDMDR\Dataset\APTOS2019\RGB'
OUT = r'D:\lodo\c20'
SIZE = 512
GRADES = ['0', '1', '2', '3', '4']


def listing():
    out = []
    for g in GRADES:
        d = os.path.join(SRC, g)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                out.append((g, f, os.path.join(d, f)))
    return out


def stage_cpu(worker, nworkers):
    sys.path.insert(0, r'D:\Mukesh Delu\VDMDR')
    from preprocessing.green_channel_enhancer import GreenChannelEnhancer
    enh = GreenChannelEnhancer()
    cl = cv2.createCLAHE(clipLimit=2.55, tileGridSize=(8, 8))
    items = listing()[worker::nworkers]
    t0 = time.time()
    for i, (g, fn, path) in enumerate(items):
        stem = os.path.splitext(fn)[0]
        dests = {k: os.path.join(OUT, k, g, stem + '.png') for k in ('green', 'fadhe', 'clahe')}
        if all(os.path.exists(v) for v in dests.values()):
            continue
        for v in dests.values():
            os.makedirs(os.path.dirname(v), exist_ok=True)
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        bgr = cv2.resize(bgr, (SIZE, SIZE))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        green = rgb[:, :, 1]
        cv2.imwrite(dests['green'], green)
        cv2.imwrite(dests['clahe'], cl.apply(green))
        cv2.imwrite(dests['fadhe'], (enh.enhance_green_channel(rgb) * 255).astype(np.uint8))
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print('  w%d %d/%d  %.1f min  eta %.1f min'
                  % (worker, i + 1, len(items), el / 60, (len(items) - i - 1) * el / (i + 1) / 60), flush=True)
    print('worker %d finished %d' % (worker, len(items)), flush=True)


def stage_vessel():
    os.environ['MODEL_DEVICE'] = 'cuda'
    sys.path.insert(0, r'C:\DrBackend')
    import torch
    from models.model_loader import ModelManager
    from services.preprocessing import preprocess_for_vessel
    mm = ModelManager(use_fp16=False)
    mm.load_all_models()
    dev = mm.device
    items = listing()
    t0 = time.time()
    for i, (g, fn, path) in enumerate(items):
        stem = os.path.splitext(fn)[0]
        dst = os.path.join(OUT, 'vessel', g, stem + '.png')
        if os.path.exists(dst):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        vt, _ = preprocess_for_vessel(rgb)
        with torch.no_grad():
            m = (torch.sigmoid(mm.vessel_model(vt.to(dev))) > 0.5).cpu().numpy().squeeze()
        cv2.imwrite(dst, cv2.resize(m.astype(np.uint8) * 255, (SIZE, SIZE),
                                    interpolation=cv2.INTER_NEAREST))
        if (i + 1) % 250 == 0:
            el = time.time() - t0
            print('  vessel %d/%d  %.1f min  eta %.1f min'
                  % (i + 1, len(items), el / 60, (len(items) - i - 1) * el / (i + 1) / 60), flush=True)
    print('vessel maps complete', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['cpu', 'vessel', 'count'])
    ap.add_argument('--worker', type=int, default=0)
    ap.add_argument('--nworkers', type=int, default=1)
    a = ap.parse_args()
    if a.stage == 'count':
        it = listing()
        print('%d APTOS images' % len(it))
        for k in ('green', 'fadhe', 'clahe', 'vessel'):
            n = sum(len(os.listdir(os.path.join(OUT, k, g)))
                    for g in GRADES if os.path.isdir(os.path.join(OUT, k, g)))
            print('  %-7s %d' % (k, n))
    elif a.stage == 'cpu':
        stage_cpu(a.worker, a.nworkers)
    else:
        stage_vessel()
