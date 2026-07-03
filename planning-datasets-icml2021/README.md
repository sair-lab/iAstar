# Datasets for Path Planning using Neural A* Search (ICML'21)

This repository is for generating datasets used in [our Neural A\* project](https://github.com/omron-sinicx/neural-astar):

Ryo Yonetani\*, Tatsunori Taniai\*, Mohammadamin Barekatain, Mai Nishimura, Asako Kanezaki, "Path Planning using Neural A\* Search", ICML, 2021 [[paper]](https://arxiv.org/abs/2009.07476) [[project page]](https://omron-sinicx.github.io/neural-astar/)

## Getting Started

### Prerequisites
- python3 (>=3.7.7)
- python3-venv

### Install using venv
```sh
$ git submodule update --init --recursive    # if you forget --recursive option
$ python3 -m venv venv
$ source activate venv/bin/activate
$ pip install -e .
```

### Generate datasets

```sh
$ sh 0_MP.sh		# generate shortest path problem instances for MP dataset
$ sh 1_TiledMP.sh	# for Tiled MP dataset
$ sh 2_CSM.sh		# for CSM dataset
$ sh 3_SDD.sh		# generate image+pedestrian traj instances from Stanford Drone Dataset
```

If you want to fully reproduce our result for MP, TiledMP, and CSM datasets, please use the original data included in this repository.

### Data format (c.f. https://github.com/omron-sinicx/neural-astar/issues/1#issuecomment-968063948)

The datafile `mazes_032_moore_c8.npz` was created using our data generation script in a separate repository https://github.com/omron-sinicx/planning-datasets.

In the data, `arr_0` - `arr_3` are 800 training, `arr_4` - `arr_7` are 100 validation, and `arr_8` - `arr_11` are 100 test data, which contain the following information:

- `arr_0`, `arr_4`, `arr_8`: binary input maps
- `arr_1`, `arr_5`, `arr_9`: one-hot goal maps
- `arr_2`, `arr_6`, `arr_10`: optimal directions (among eight directions) to reach the goal
- `arr_3`, `arr_7`, `arr_11`: shortest distances to the goal

https://github.com/omron-sinicx/planning-datasets/blob/68e182801fd8cbc4c25ccdc1b14b8dd99d9bbc73/generate_spp_instances.py#L48-L62

For each problem instance, the start location is generated randomly when `__getitem__` is called: https://github.com/omron-sinicx/neural-astar/blob/e6e626c4d159b0e4c58ee6ad33c7e03db33d72f4/neural_astar/utils/data.py#L114

### Acknowledgments

- This repository includes some code from [RLAgent/gated-path-planning-networks](https://github.com/RLAgent/gated-path-planning-networks) [1], with permission of the authors.
- MP and TiledMP datasets are created from [mohakbhardwaj/motion_planning_datasets](https://github.com/mohakbhardwaj/motion_planning_datasets) [2].
- CSM dataset is created using [City/Street Maps in Pathfinding Benchmarks](https://movingai.com/benchmarks/grids.html) [3].
- SDD dataset is created using Stanford Drone Dataset [4] reorganized in [crowdbotp/OpenTraj](https://github.com/crowdbotp/OpenTraj) [5].


### Reference
- [1] [Lisa Lee*, Emilio Parisotto*, Devendra Singh Chaplot, Eric Xing, Ruslan Salakhutdinov, "Gated Path Planning Networks", ICML, 2018.](https://arxiv.org/abs/1806.06408)
- [2] [Mohak Bhardwaj, Sanjiban Choudhury, Sebastian Scherer, "Learning Heuristic Search via Imitation", CoRL, 2017.](https://arxiv.org/abs/1707.03034)
- [3] [Nathan Sturtevant, "Benchmarks for Grid-Based Pathfinding", Transactions on Computational Intelligence and AI in Games, 2012.](https://ieeexplore.ieee.org/document/6194296)
- [4] [Alexandre Robicquet, Amir Sadeghian, Alexandre Alahi, Silvio Savarese, "Learning social etiquette: Human trajectory understanding in crowded scenes", ECCV, 2016.](http://svl.stanford.edu/assets/papers/ECCV16social.pdf)
- [5] [Javad Amirian, Bingqing Zhang, Francisco Valente Castro, Juan Jose Baldelomar, Jean-Bernard Hayet, Julien Pettre, "OpenTraj: Assessing Prediction Complexity in Human Trajectories Datasets", ACCV, 2020.](https://arxiv.org/abs/2010.00890 )
