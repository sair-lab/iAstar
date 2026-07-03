# Create train/val/test splits for Tiled MP dataset
# Author: Ryo Yonetani
# Affiliation: OMRON SINIC X

python generate_spp_instances.py --input-path "data/mpd/original/*" --output-path data/mpd/ --maze-size 32 --mechanism moore --edge-ratio 0.25 --tile-size=2 --train-size 3200 --valid-size 400 --test-size 400