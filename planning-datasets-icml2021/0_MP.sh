# Create train/val/test splits for MP dataset
# Author: Ryo Yonetani
# Affiliation: OMRON SINIC X

for maze in alternating_gaps bugtrap_forest forest gaps_and_forest mazes multiple_bugtraps shifting_gaps single_bugtrap
do
python generate_spp_instances.py --input-path data/mpd/original/$maze --output-path data/mpd/ --maze-size 32 --mechanism moore --edge-ratio 0.25
done