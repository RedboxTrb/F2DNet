"""Leave-one-dataset-out for vessel segmentation (R1-C1, R1-C2, R1-C16; R2-C3, R2-C4).

The reviewers reject the published random mixed split because it measures mixed-domain
interpolation, not generalisation, and because FIVES is 120 of the 139 published test images
(86.3%) while STARE and DRIVE contribute 3 each. LODO answers both: train on four datasets,
test on the whole of the fifth, no fine-tuning.

Stages, run separately so the GPU stays free during the hospital demo:

  prep      merge the five processed datasets into one tree, build the five LODO split sets
            (CPU only, seconds)
  features  generate the 10-channel ABC input for every image
            (CPU only, ~4 s/image at 1024^2, parallelised across cores)
  train     five training runs                                    <-- GPU, ~11 h, run later
  eval      score each fold on its held-out dataset, bootstrap CIs

Epoch budget: the published checkpoint's best validation Dice was at **epoch 34** of a 120
epoch schedule (models_multiscale/best_model.pth), so 40 epochs per fold is not a materially
truncated schedule. Label it reduced-scale in the response letter regardless.
"""
import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time

ROOT = r'D:\lodo'
SRC = os.path.join(ROOT, 'src')             # extracted *_p folders live here
DATA = os.path.join(ROOT, 'Dataset')        # merged images/ ground_truth/ masks/
FEATS = os.path.join(ROOT, 'Dataset', 'multiscale_features')
SPLITS = os.path.join(ROOT, 'splits')
OUT = os.path.join(ROOT, 'runs')
CODE = os.path.join(ROOT, 'code')

SETS = {'FIVES': 'FIVE_p', 'HRF': 'HRF_p', 'CHASE': 'ChaseDB_p',
        'DRIVE': 'DRIVE_p', 'STARE': 'Stare_p'}
VAL_FRACTION = 0.12
SEED = 42


def stems_of(folder):
    d = os.path.join(SRC, folder, 'images')
    if not os.path.isdir(d):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(d) if f.lower().endswith('.png'))


def stage_prep():
    for sub in ('images', 'ground_truth', 'masks'):
        os.makedirs(os.path.join(DATA, sub), exist_ok=True)
    os.makedirs(SPLITS, exist_ok=True)

    membership = {}
    for name, folder in SETS.items():
        stems = stems_of(folder)
        if not stems:
            print('  !! %-6s missing (%s)' % (name, folder))
            continue
        membership[name] = stems
        for s in stems:
            for sub, suffix in (('images', ''), ('ground_truth', '_gt'), ('masks', '_mask')):
                src = os.path.join(SRC, folder, sub, s + suffix + '.png')
                dst = os.path.join(DATA, sub, s + suffix + '.png')
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)
        print('  %-6s %4d images' % (name, len(stems)))

    total = sum(len(v) for v in membership.values())
    print('merged %d images into %s' % (total, DATA))

    rng = random.Random(SEED)
    index = {}
    for held in membership:
        train_pool = [s for n, v in membership.items() if n != held for s in v]
        rng.shuffle(train_pool)
        n_val = max(4, int(len(train_pool) * VAL_FRACTION))
        val, train = train_pool[:n_val], train_pool[n_val:]
        test = membership[held]
        d = os.path.join(SPLITS, 'holdout_' + held)
        os.makedirs(d, exist_ok=True)
        for fn, items in (('train.txt', train), ('val.txt', val), ('test.txt', test)):
            with open(os.path.join(d, fn), 'w', encoding='utf-8') as f:
                f.write('\n'.join(items) + '\n')
        index[held] = dict(train=len(train), val=len(val), test=len(test))
        print('  holdout %-6s train %4d  val %3d  test %4d' % (held, len(train), len(val), len(test)))

    with open(os.path.join(ROOT, 'lodo_index.json'), 'w', encoding='utf-8') as f:
        json.dump({'membership': {k: len(v) for k, v in membership.items()},
                   'folds': index, 'seed': SEED}, f, indent=2)
    print('wrote ' + os.path.join(ROOT, 'lodo_index.json'))


def stage_features(workers):
    """Generate the 9 ABC channels per image (green is added by the loader -> 10)."""
    sys.path.insert(0, CODE)
    os.makedirs(FEATS, exist_ok=True)
    todo = [os.path.splitext(f)[0] for f in sorted(os.listdir(os.path.join(DATA, 'images')))
            if f.lower().endswith('.png')]
    todo = [s for s in todo if not os.path.exists(os.path.join(FEATS, s + '_features.npy'))]
    print('%d images need features (%d workers)' % (len(todo), workers))
    if not todo:
        return
    # fan out across processes: the work is pure numpy/scipy and releases no GIL benefit
    chunks = [todo[i::workers] for i in range(workers)]
    procs = []
    for i, ch in enumerate(chunks):
        if not ch:
            continue
        lst = os.path.join(ROOT, 'featlist_%d.txt' % i)
        with open(lst, 'w', encoding='utf-8') as f:
            f.write('\n'.join(ch))
        procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__),
                                       '--stage', 'features_worker', '--list', lst],
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT))
    t0 = time.time()
    for p in procs:
        out, _ = p.communicate()
        tail = (out or b'').decode('utf-8', 'replace').strip().splitlines()[-1:]
        print('  worker done rc=%s %s' % (p.returncode, tail))
    print('features complete in %.1f min' % ((time.time() - t0) / 60))


def stage_features_worker(list_path):
    sys.path.insert(0, CODE)
    import cv2
    import numpy as np
    import generate_abc_features as G
    stems = [l.strip() for l in open(list_path, encoding='utf-8') if l.strip()]
    for i, s in enumerate(stems):
        dst = os.path.join(FEATS, s + '_features.npy')
        if os.path.exists(dst):
            continue
        img = cv2.imread(os.path.join(DATA, 'images', s + '.png'), cv2.IMREAD_COLOR)
        if img is None:
            print('unreadable', s)
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        feats = [G.vessel_enhancement(rgb, v, G.GAUSSIAN_SIGMA) for v in G.FRACTIONAL_ORDERS]
        clahe = G.vessel_enhancement.__globals__['equalize_adapthist'](
            rgb[:, :, 1].astype(np.float64) / 255.0)
        stack = np.stack(feats + [clahe.astype(np.float32)], axis=-1).astype(np.float32)
        np.save(dst, stack)
        if (i + 1) % 25 == 0:
            print('  %d/%d' % (i + 1, len(stems)), flush=True)
    print('worker finished %d' % len(stems))


def stage_train(epochs, folds):
    os.makedirs(OUT, exist_ok=True)
    for held in folds:
        outdir = os.path.join(OUT, 'holdout_' + held)
        if os.path.exists(os.path.join(outdir, 'best_model.pth')):
            print('skip %s (already trained)' % held)
            continue
        cmd = [sys.executable, os.path.join(CODE, 'train_multiscale.py'),
               '--dataset_dir', DATA, '--features_dir', FEATS,
               '--splits_dir', os.path.join(SPLITS, 'holdout_' + held),
               '--output_dir', outdir, '--num_epochs', str(epochs)]
        print('\n=== training, held-out = %s (%d epochs) ===' % (held, epochs), flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=CODE)
        print('  rc=%s in %.1f min' % (rc, (time.time() - t0) / 60), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True,
                    choices=['prep', 'features', 'features_worker', 'train'])
    ap.add_argument('--list')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--folds', default='FIVES,HRF,CHASE,DRIVE,STARE')
    a = ap.parse_args()
    if a.stage == 'prep':
        stage_prep()
    elif a.stage == 'features':
        stage_features(a.workers)
    elif a.stage == 'features_worker':
        stage_features_worker(a.list)
    elif a.stage == 'train':
        stage_train(a.epochs, [f for f in a.folds.split(',') if f])


if __name__ == '__main__':
    main()
