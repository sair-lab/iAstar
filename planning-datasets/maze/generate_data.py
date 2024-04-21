from zanj import ZANJ
from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.generation import LatticeMazeGenerators, GENERATORS_MAP
from maze_dataset.generation.default_generators import DEFAULT_GENERATORS

LOCAL_DATA_PATH: str = "data/"
DATASETS: dict[int, list[MazeDataset]] = dict()
zanj: ZANJ = ZANJ(external_list_threshold=256)

def main():

    for grid_n in [32, 64, 128]:
        DATASETS[grid_n] = list()
        for gen_name, gen_kwargs in DEFAULT_GENERATORS:
            print(f"Generating {gen_name} for grid_n={grid_n}")
            DATASETS[grid_n].append(MazeDataset.from_config(
                MazeDatasetConfig(
                    name="demo",
                    maze_ctor=GENERATORS_MAP[gen_name],
                    grid_n=grid_n,
                    n_mazes=8,
                    maze_ctor_kwargs=gen_kwargs,
                ),
                local_base_path=LOCAL_DATA_PATH + str(grid_n) + '/',
                load_local=False,
                verbose=False,
                zanj=zanj,
            ))
