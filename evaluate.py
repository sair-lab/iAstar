"""Evaluate a trained iA* model against vanilla A* on a held-out split.

Both use the same differentiable A* engine (``dastar``); they differ only in the cost map:
  * iA*     : the learned cost map produced by the trained encoder
  * vanilla : a uniform cost (the obstacle map) -> standard grid A*

Reported per-instance metrics (averaged over the split):
  * search area : number of expanded nodes (histories.sum) -- lower is better
  * path length : geometric length of the found path (diagonal = sqrt(2))
  * success     : the found path connects start -> goal
  * area reduction of iA* vs vanilla A*

Usage:
    python evaluate.py                       # newest checkpoint, test split
    python evaluate.py --model <path.pkl> --split test --data <npz>
"""
import os
import glob
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import create_dataloader
from iastar import iastar
from dastar import dastar


def build_args():
    p = argparse.ArgumentParser(description="Evaluate iA* vs vanilla A*")
    p.add_argument("--data", type=str,
                   default="planning-datasets/mpd/instances/032/mazes_032_moore_c8.npz")
    p.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    p.add_argument("--model", type=str, default=None,
                   help="path to a trained iaster1UNet.pkl (default: newest under model/)")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--encoder-arch", type=str, default="UNet")
    p.add_argument("--w", type=float, default=2.0,
                   help="heuristic weight for both iA* and the vanilla baseline")
    return p.parse_args()


def main():
    args = build_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    kernel = torch.tensor([[[1.414, 1., 1.414], [1., 0., 1.], [1.414, 1., 1.414]]],
                          device=device).expand(1, 1, 3, 3)
    pad = nn.ZeroPad2d((1, 1, 1, 1)).to(device)

    def area(h):
        return h.float().sum((1, 2, 3))

    def path_len(p):
        p = p.float()
        return (F.conv2d(pad(p), kernel) * p).sum((1, 2, 3)) / 2.0

    def connects(path, start, goal):
        p = path.float()
        return ((p * start).sum((1, 2, 3)) > 0) & ((p * goal).sum((1, 2, 3)) > 0)

    model_path = args.model or max(glob.glob("model/*/*/iaster1UNet.pkl"), key=os.path.getmtime)
    ia = iastar(encoder_input=3, encoder_arch=args.encoder_arch, device=device,
                encoder_depth=4, is_training=False, output_path_list=False, w=args.w).to(device)
    ia.encoder.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])
    ia.eval()
    va = dastar(device=device, is_training=False, output_path_list=False, w=args.w)
    print(f"iA* checkpoint : {model_path}")
    print(f"evaluating '{args.split}' split of {args.data}\n")

    agg = {k: [] for k in ["va_a", "ia_a", "va_l", "ia_l", "ia_ok", "va_ok"]}
    with torch.no_grad():
        for maps, start, goal, _ in create_dataloader(args.data, args.split, args.batch_size):
            maps, start, goal = maps.to(device), start.to(device), goal.to(device)
            o_ia = ia(maps, start, goal)             # learned cost
            o_va = va(maps, start, goal, maps)        # uniform cost
            agg["ia_a"] += area(o_ia.histories).tolist()
            agg["va_a"] += area(o_va.histories).tolist()
            agg["ia_l"] += path_len(o_ia.paths).tolist()
            agg["va_l"] += path_len(o_va.paths).tolist()
            agg["ia_ok"] += connects(o_ia.paths, start, goal).tolist()
            agg["va_ok"] += connects(o_va.paths, start, goal).tolist()

    m = {k: np.array(v) for k, v in agg.items()}
    n = len(m["ia_a"])
    print(f"N = {n} instances\n")
    print(f"{'metric':<22}{'vanilla A*':>13}{'iA* (trained)':>16}")
    print("-" * 51)
    print(f"{'search area (nodes)':<22}{m['va_a'].mean():>13.1f}{m['ia_a'].mean():>16.1f}")
    print(f"{'path length':<22}{m['va_l'].mean():>13.2f}{m['ia_l'].mean():>16.2f}")
    print(f"{'success rate':<22}{m['va_ok'].mean():>12.0%}{m['ia_ok'].mean():>16.0%}")
    print("-" * 51)
    red = 100 * (1 - m["ia_a"].mean() / m["va_a"].mean())
    ratio = m["ia_l"].mean() / m["va_l"].mean()
    print(f"\n=> iA* expands {red:.1f}% fewer nodes than vanilla A*, "
          f"paths within {100*(ratio-1):.1f}% of optimal length, {m['ia_ok'].mean():.0%} success.")


if __name__ == "__main__":
    main()
