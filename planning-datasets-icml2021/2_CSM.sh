# Create train/val/test splits for CSM dataset
# Author: Ryo Yonetani
# Affiliation: OMRON SINIC X

mkdir -p data/street/original
mkdir -p data/street/original/all
wget https://www.movingai.com/benchmarks/street/street-png.zip -O data/street/original/street-png.zip
unzip data/street/original/street-png.zip -d data/street/original/all/
mkdir -p data/street/original/mixed/train
mkdir -p data/street/original/mixed/validation
mkdir -p data/street/original/mixed/test

python preprocess_street_dataset.py
python generate_spp_instances.py --input-path data/street/original/mixed --output-path data/street/ --maze-size 64 --mechanism moore --edge-ratio 0.25 --train-size 3200 --valid-size 400 --test-size 400