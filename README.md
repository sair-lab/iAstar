# iAstar — Imperative A\*

A **self-supervised, differentiable A\*** path planner. A learned cost-map encoder guides a
differentiable A\* search so that it **explores far fewer nodes while still returning
near-optimal paths** — and it is trained **without ground-truth optimal paths**.

The key idea ("imperative learning"): instead of supervising the search with a pre-computed
optimal path, we pull the differentiable **search history** toward the planner's **own found
path**. This shrinks the search area while *maintaining* the path the planner already produces.

```text
loss = L1( search_history , found_path.detach() )        # self-supervised
```

## Results

Trained on the `mazes_032_moore_c8` split (32×32 grid mazes) and evaluated on its 100 held-out
test maps. Vanilla A\* and iA\* share the same differentiable A\* engine and differ only in the
cost map (uniform vs. learned):

| metric | vanilla A\* | **iA\* (self-supervised)** |
| --- | ---: | ---: |
| search area (nodes expanded) | 77.3 | **50.0** |
| path length | 35.89 | 36.35 |
| success rate | 100% | 100% |

> **iA\* expands ~35% fewer nodes than vanilla A\*, keeps paths within ~1.3% of optimal, at 100% success.**

Reproduce with `python evaluate.py`.

### Comparison with Neural A\* (same data, encoder, and budget)

Both models: UNet encoder, 40 epochs on `mazes_032` only. Neural A\* is trained supervised
(`python train_nastar.py`); iA\* self-supervised (`python train.py --useIL`). Mean **absolute
nodes expanded** over the 8 MP test environments, then out-of-distribution sizes
(`python benchmark.py`):

| test setting | vanilla A\* | Neural A\* | **iA\*** |
| --- | ---: | ---: | ---: |
| 32×32 (training size) | 100.5 | 107.7 | **80.1** |
| 64×64 | 327 | 369 | **261** |
| 128×128 | 1360 | 1362 | **964** |
| 256×256 | 6736 | 6632 | **5069** |

iA\* expands the fewest nodes at every scale and keeps paths closer to optimal (+4–7% vs
Neural A\*'s +9–11%), with 100% success for all planners. Notably, the supervised baseline
stops helping beyond its training size, while the self-supervised objective keeps
transferring (20–29% reductions at 2–8× the training scale).

> Caveat: the baseline is our reproduction trained under the matched protocol above — the
> officially tuned per-dataset Neural A\* checkpoints may perform better on their home settings.

## Installation

Tested on Ubuntu + an NVIDIA RTX 5090 (Blackwell, `sm_120`) with a dedicated conda env.

```bash
conda create -n iastar python=3.10 -y
conda activate iastar

# 1) PyTorch — install FIRST from the CUDA 12.8 index (required for Blackwell / sm_120).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2) Core + evaluation dependencies (numpy pinned < 2 for compatibility).
pip install "numpy==1.26.4" pyyaml opencv-python wandb \
            pandas matplotlib pytorch-lightning einops scikit-image scipy

# 3) Neural A* baseline — install with --no-deps so it cannot downgrade the cu128 torch,
#    then add its two extra runtime deps.
pip install --no-deps "git+https://github.com/omron-sinicx/neural-astar.git"
pip install pqdict segmentation-models-pytorch
```

Exact pinned versions used for the reference results are in [`requirements.txt`](requirements.txt).

Sanity check the GPU build:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Dataset

Datasets follow the [Neural A\*](https://github.com/omron-sinicx/planning-datasets) `.npz` format
(`map_designs`, `goal_maps`, `opt_policies`, `opt_dists` per split). `config/config.yaml` points at:

```text
dataset      : planning-datasets/mpd/instances/032/   # training maps
val_dataset  : planning-datasets/val/                 # validation maps
```

The 32×32 MPD training file ships in this repo under
`planning-datasets-icml2021/data/mpd/mazes_032_moore_c8.npz`. Put it where the config expects it:

```bash
mkdir -p planning-datasets/mpd/instances/032
ln -s "$(pwd)/planning-datasets-icml2021/data/mpd/mazes_032_moore_c8.npz" \
      planning-datasets/mpd/instances/032/
```

To regenerate datasets from scratch, see the scripts under `planning-datasets/`.

## Usage

### Train

```bash
mkdir -p logs model

# Self-supervised imperative loss (shrink search toward the planner's own path):
python train.py --useIL

# Supervised loss (L1 between search history and the ground-truth optimal path):
python train.py
```

Useful flags (defaults come from `config/config.yaml`):
`--useIL` (self-supervised loss on/off), `--epochs`, `--map-num` (training batch size),
`--encoder-arch` (`CNN`, `UNet`, `UNetAtt`, `FCN`), `--lr`, `--gpu-id`.
Checkpoints are written to `model/<timestamp>/<epoch>/iaster1UNet.pkl`.

Weights & Biases logging is on by default; run offline with `WANDB_MODE=disabled` or `wandb login` first.

### Train the Neural A\* baseline

The comparisons in the README use a Neural A\* baseline (supervised `L1(history, optimal_path)`,
UNet encoder) trained on the **same data and budget** as iA\*:

```bash
python train_nastar.py        # 40 epochs on MP mazes 32x32 -> model/nastar/nastar_unet_mazes032.pth
```

### Evaluate

```bash
python evaluate.py                                   # newest iA* checkpoint vs vanilla A*, test split
python evaluate.py --model model/<...>/iaster1UNet.pkl --w 2.0
```

### Benchmark (multi-size, multi-family)

Sweeps every dataset under `planning-datasets/{mpd,maze,matterport}/instances/<size>/`
and reports absolute nodes expanded, path length, and success for vanilla A\*, Neural A\*, and iA\*:

```bash
python benchmark.py                    # sizes 064 128 256
python benchmark.py --sizes 064        # single size
python benchmark.py --skip-na          # without the Neural A* baseline
```

### Example notebook

[`example/example.ipynb`](example/example.ipynb) renders side-by-side visualizations of the
explored area and found path for vanilla A\*, Neural A\*, and iA\*.

## How it works

- **Encoder** ([`encoder.py`](encoder.py)) maps `(obstacle map, start, goal)` to a per-cell cost map.
- **Differentiable A\*** ([`dastar.py`](dastar.py)) runs A\* with a straight-through softmax over the
  open list, so the **search history is differentiable** w.r.t. the cost map. (The backtraced path
  is discrete/non-differentiable — training signals go through the history.)
- **iA\* wrapper** ([`iastar.py`](iastar.py)) ties the encoder and the differentiable A\* together.
- **Loss** ([`train.py`](train.py), `CostofTraj`): self-supervised `L1(history, found_path.detach())`.

## Repository layout

| path | description |
| --- | --- |
| `iastar.py` | iA\* model: cost-map encoder + differentiable A\* |
| `dastar.py` | differentiable A\* search |
| `encoder.py`, `layers.py` | cost-map encoder architectures (CNN / UNet / UNetAtt / FCN) |
| `train.py` | training loop, losses, W&B logging |
| `evaluate.py` | iA\* vs vanilla A\* evaluation |
| `data_loader.py` | dataset loaders for the `.npz` shortest-path format |
| `config/config.yaml` | training / model / data configuration |
| `jps/` | Jumping Point Search reference implementation |
| `transpath/` | vendored TransPath modules (used by the extended benchmarks) |
| `planning-datasets*/` | dataset generation and source data |

## Acknowledgments

Builds on ideas and tooling from
[Neural A\*](https://github.com/omron-sinicx/neural-astar) and
[TransPath](https://github.com/AIRI-Institute/TransPath).
