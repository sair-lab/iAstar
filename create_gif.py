import os
import yaml
import glob
import hydra

import moviepy.editor as mpy
from iastar import iastar
from neural_astar.utils.data import create_dataloader, visualize_results
from neural_astar.utils.training import load_from_ptl_checkpoint

root_folder = "/home/cxy/SAIRLAB/"
def main():
    filepath = os.path.join(os.path.dirname(root_folder),'iAstar','config', 'create_gif.yaml')
    with open(filepath,mode="r",encoding="utf-8") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    print(config)
    dataname = os.path.basename(config["dataset"])

    planner = iastar(
    encoder_input=config["encoder"]["input"],
    encoder_arch="CNN",
    device="cuda",
    encoder_depth=config["encoder"]["depth"],
    learn_obstacles=False,
    Tmax=config["Tmax"],
    is_training = True,
    output_path_list= False,
    store_intermediate_results = True,
    w=1.0).to('cuda')
    planner.load_state_dict(load_from_ptl_checkpoint("/home/cxy/iastar/iastar_planner/iastar_planner-main/scripts/iastar/model/mazes_032_moore_c8/lightning_logs/version_74/"))

    problem_id = config["problem_id"]
    savedir = config["resultdir"]+"/iastar"
    os.makedirs(savedir, exist_ok=True)

    dataloader = create_dataloader(
        config["dataset"] + ".npz",
        "test",
        100,
        shuffle=False,
        num_starts=1,
    )
    map_designs, start_maps, goal_maps, opt_trajs = next(iter(dataloader))
    outputs = planner(
        map_designs.to('cuda'), start_maps.to('cuda'), goal_maps.to('cuda')
    )

    outputs = planner(
        map_designs[problem_id : problem_id + 1].to('cuda'),
        start_maps[problem_id : problem_id + 1].to('cuda'),
        goal_maps[problem_id : problem_id + 1].to('cuda'),
        # store_intermediate_results=True,
    )
    print()
    frames = [
        visualize_results(
            map_designs[problem_id : problem_id + 1], intermediate_results, scale=4
        )
        for intermediate_results in outputs.intermediate_result
    ]
    print(len(frames))
    clip = mpy.ImageSequenceClip(frames + [frames[-1]] * 15, fps=30)
    clip.write_gif(f"{savedir}/video_{dataname}_{problem_id:04d}.gif")


if __name__ == "__main__":
    main()
