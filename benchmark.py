"""Benchmark vanilla A*, Neural A*, and iA* across dataset families and map sizes.

Sweeps every npz under planning-datasets/{mpd,maze,matterport}/instances/<size>/ and
reports, per dataset and on average: nodes expanded (search area), path length, and
success rate. Absolute node counts make the numbers engine-independent.

Usage:
    python benchmark.py                          # all sizes: 064 128 256
    python benchmark.py --sizes 064              # one size
    python benchmark.py --skip-na                # without the Neural A* baseline
    python benchmark.py --ia-ckpt <pkl> --w 2.0  # a specific iA* checkpoint / weight
"""
import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

from data_loader import MazeDataset, TSDFDataset
from iastar import iastar
from dastar import dastar

CAPS = {"032": 100, "064": 50, "128": 30, "256": 8}     # instances per dataset
BATCH = {"032": 100, "064": 50, "128": 10, "256": 4}


def build_args():
    p = argparse.ArgumentParser(description="Benchmark vanilla A* / Neural A* / iA*")
    p.add_argument("--sizes", nargs="+", default=["064", "128", "256"],
                   choices=list(CAPS.keys()))
    p.add_argument("--ia-ckpt", type=str, default=None,
                   help="iA* checkpoint .pkl (default: newest under model/)")
    p.add_argument("--na-ckpt", type=str, default="model/nastar/nastar_unet_mazes032.pth",
                   help="Neural A* baseline checkpoint (see train_nastar.py)")
    p.add_argument("--w", type=float, default=2.0, help="heuristic weight for iA*/vanilla")
    p.add_argument("--skip-na", action="store_true", help="skip the Neural A* baseline")
    return p.parse_args()


def load_ds(path):
    """Auto-detect npz layout (15 arrays = TSDF, else maze) and pick test/train split."""
    with np.load(path) as f:
        n = len(f.files)
    cls = TSDFDataset if n == 15 else MazeDataset
    for split in ("test", "train"):
        try:
            ds = cls(path, split)
            if len(ds) > 0:
                return ds, split
        except Exception:
            pass
    raise RuntimeError("no usable split")


def main():
    args = build_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    kernel = torch.tensor([[[1.414, 1., 1.414], [1., 0., 1.], [1.414, 1., 1.414]]],
                          device=device).expand(1, 1, 3, 3)
    pad = nn.ZeroPad2d((1, 1, 1, 1)).to(device)

    def area(h): return h.float().sum((1, 2, 3))
    def plen(p):
        p = p.float(); return (F.conv2d(pad(p), kernel) * p).sum((1, 2, 3)) / 2.0
    def okc(p, s, g):
        p = p.float(); return ((p * s).sum((1, 2, 3)) > 0) & ((p * g).sum((1, 2, 3)) > 0)

    ia_ckpt = args.ia_ckpt or max(glob.glob("model/*/*/iaster1UNet.pkl"), key=os.path.getmtime)
    ia = iastar(encoder_input=3, encoder_arch="UNet", device=device, encoder_depth=4,
                is_training=False, output_path_list=False, w=args.w).to(device).eval()
    ia.encoder.load_state_dict(torch.load(ia_ckpt, map_location=device)["model_state_dict"])
    va = dastar(device=device, is_training=False, output_path_list=False, w=args.w)
    print(f"iA* checkpoint: {ia_ckpt} (w={args.w})")

    na = None
    if not args.skip_na:
        from neural_astar.planner import NeuralAstar
        na = NeuralAstar(encoder_arch="Unet", Tmax=1.0).to(device)
        na.load_state_dict(torch.load(args.na_ckpt, map_location=device))
        na.eval()
        print(f"Neural A* checkpoint: {args.na_ckpt}")

    for size in args.sizes:
        cap, bs = CAPS[size], BATCH[size]
        paths = sorted(glob.glob(f"planning-datasets/mpd/instances/{size}/*.npz")) \
            + glob.glob(f"planning-datasets/maze/instances/{size}/*.npz") \
            + glob.glob(f"planning-datasets/matterport/instances/{size}/*.npz")
        if not paths:
            print(f"\n### size {size}: no datasets found, skipping"); continue

        print(f"\n### size {size} (N<={cap}/dataset)")
        print(f"{'dataset':<20}{'N':>4}{'split':>6} | {'van':>8}{'NA':>8}{'iA*':>8} | "
              f"{'vanL':>7}{'NAL':>7}{'iA*L':>7} | {'iA*ok':>6}{'sec':>5}")
        print("-" * 100)
        acc = {x: [] for x in ["v", "n", "i", "vl", "nl", "il"]}
        for path in paths:
            name = os.path.basename(path).split("_" + size)[0].split("_moore")[0]
            try:
                ds, split = load_ds(path)
            except Exception as e:
                print(f"{name:<20}  SKIP ({e})"); continue
            sub = tud.Subset(ds, range(min(cap, len(ds))))
            A = {x: 0.0 for x in ["v", "n", "i", "vl", "nl", "il", "ok"]}; N = 0
            t0 = time.time()
            try:
                with torch.no_grad():
                    for b in tud.DataLoader(sub, batch_size=bs):
                        m, s, g = b[0].to(device), b[1].to(device), b[2].to(device)
                        ov = va(m, s, g, m); oi = ia(m, s, g)
                        A["v"] += area(ov.histories).sum().item(); A["i"] += area(oi.histories).sum().item()
                        A["vl"] += plen(ov.paths).sum().item();     A["il"] += plen(oi.paths).sum().item()
                        A["ok"] += okc(oi.paths, s, g).sum().item()
                        if na is not None:
                            on = na(m, s, g)
                            A["n"] += area(on.histories).sum().item(); A["nl"] += plen(on.paths).sum().item()
                        N += m.shape[0]
            except Exception as e:
                print(f"{name:<20}  SKIP ({type(e).__name__}: {str(e)[:40]})"); continue
            for x in acc:
                acc[x].append(A[x] / N)
            print(f"{name:<20}{N:>4}{split:>6} | {A['v']/N:>8.1f}{A['n']/N:>8.1f}{A['i']/N:>8.1f} | "
                  f"{A['vl']/N:>7.1f}{A['nl']/N:>7.1f}{A['il']/N:>7.1f} | "
                  f"{A['ok']/N:>6.0%}{time.time()-t0:>5.0f}")
        if not acc["v"]:
            continue
        print("-" * 100)
        mv, mn, mi = np.mean(acc["v"]), np.mean(acc["n"]), np.mean(acc["i"])
        print(f"{'MEAN':<20}{'':>10} | {mv:>8.1f}{mn:>8.1f}{mi:>8.1f} | "
              f"{np.mean(acc['vl']):>7.1f}{np.mean(acc['nl']):>7.1f}{np.mean(acc['il']):>7.1f}")
        na_red = f"NA {100*(1-mn/mv):.1f}%   " if na is not None else ""
        print(f"reduction vs vanilla:  {na_red}iA* {100*(1-mi/mv):.1f}%")


if __name__ == "__main__":
    main()
