"""Train the Neural A* baseline used for comparison with iA*.

Neural A* (Yonetani et al., ICML 2021) trains a cost-map encoder with the SUPERVISED
loss L1(search_history, ground_truth_optimal_path). This script reproduces the baseline
reported in the README: UNet encoder, 40 epochs on MP mazes 32x32.

Usage:
    python train_nastar.py                        # reproduce the README baseline
    python train_nastar.py --data <npz> --epochs 40 --encoder-arch Unet
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as tud

from data_loader import MazeDataset
from neural_astar.planner import NeuralAstar


def build_args():
    p = argparse.ArgumentParser(description="Train the Neural A* baseline (supervised)")
    p.add_argument("--data", type=str,
                   default="planning-datasets/mpd/instances/032/mazes_032_moore_c8.npz")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--w-decay", type=float, default=1e-3)
    p.add_argument("--encoder-arch", type=str, default="Unet", choices=["Unet", "CNN"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="model/nastar/nastar_unet_mazes032.pth")
    return p.parse_args()


def main():
    args = build_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    na = NeuralAstar(encoder_arch=args.encoder_arch, Tmax=1.0).to(device)
    opt = torch.optim.AdamW(na.parameters(), lr=args.lr, weight_decay=args.w_decay)
    train_set = MazeDataset(args.data, "train")

    print(f"training Neural A* ({args.encoder_arch}, supervised L1(history, opt_traj)) "
          f"on {args.data} [{len(train_set)} maps, {args.epochs} epochs, device {device}]")
    t0 = time.time()
    for ep in range(args.epochs):
        tot, n = 0.0, 0
        for maps, start, goal, opt_traj in tud.DataLoader(train_set, batch_size=args.batch_size,
                                                          shuffle=True):
            maps, start, goal, opt_traj = (maps.to(device), start.to(device),
                                           goal.to(device), opt_traj.to(device))
            opt.zero_grad()
            loss = F.l1_loss(na(maps, start, goal).histories, opt_traj)
            loss.backward()
            opt.step()
            tot += loss.item(); n += 1
        print(f"epoch {ep:3d} | loss {tot/n:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(na.state_dict(), args.out)
    print(f"done in {time.time()-t0:.0f}s | saved {args.out}")


if __name__ == "__main__":
    main()
