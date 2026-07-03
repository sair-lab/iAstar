# Create instances for Matterport dataset
# Author: Xiangyu Chen
# Affiliation: SAIR Lab

python generate_instance.py --input-path matterport/maps/ --output-path matterport/instances/ --maze-size 64 --edge-ratio 0.25 --dataset matterport
python generate_instance.py --input-path maze/maps/ --output-path maze/instances/ --maze-size 64 --edge-ratio 0.25 --dataset maze
python generate_instance.py --input-path mpd/maps/ --output-path mpd/instances/ --maze-size 64 --edge-ratio 0.25 --dataset mpd
