# Create train/val/test splits for SDD
# Author: Ryo Yonetani
# Affiliation: OMRON SINIC X

mkdir -p data/sdd/original
wget https://www.dropbox.com/s/v9jvt4ln7t42m6m/StanfordDroneDataset.zip -O data/sdd/original/SDD.zip
unzip data/sdd/original/SDD.zip -d data/sdd
mv data/sdd/SDD/* data/sdd/original/
rm -r data/sdd/SDD
rm data/sdd/original/SDD.zip

python generate_sdd_instances.py