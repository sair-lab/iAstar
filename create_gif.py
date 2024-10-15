import os
import yaml
import glob
import torch
# import hydra

import moviepy.editor as mpy
from iastar_v3 import iastar
from neural_astar.utils.data import create_dataloader, visualize_results
from data_loader import create_dataloader
from PIL import Image

def load_from_ptl_checkpoint(filename):
    return torch.load(filename)["model_state_dict"]

root_folder = "/home/cxy/SAIRLAB/"
device = 1
def main():
    filepath = os.path.join(os.path.dirname(root_folder),'iAstar','config', 'create_gif.yaml')
    with open(filepath,mode="r",encoding="utf-8") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    print(config)
    dataname = os.path.basename(config["dataset"])

    ia_planner = iastar(encoder_input=3,
                    encoder_arch="UNetAtt",
                    device=device,
                    encoder_depth=4,
                    learn_obstacles=False,
                    is_training = False,
                    output_path_list= False,
                    store_intermediate_results=True,
                    w=1.0).to(device)
    ia_planner.encoder.load_state_dict(load_from_ptl_checkpoint("/home/cxy/iAstar/model/02-09-2024-15-18-38/22/iaster1UNetAtt.pkl"))

    problem_id = 2
    st = 1
    savedir = config["resultdir"]+"/iastar"
    os.makedirs(savedir, exist_ok=True)

    dataloader = create_dataloader(
        config["dataset"] + ".npz",
        "train",
        3
    )
    map_designs, start_maps, goal_maps, _ = next(iter(dataloader))
    outputs = ia_planner(
        map_designs.to(device), start_maps.to(device), goal_maps.to(device)
    )
    with torch.no_grad():
        outputs = ia_planner(
            map_designs[problem_id : problem_id + st].to(device),
            start_maps[problem_id : problem_id + st].to(device),
            goal_maps[problem_id : problem_id + st].to(device),
            # store_intermediate_results=True,
        )
    # print()
    frames = [
        visualize_results(
            map_designs[problem_id : problem_id + st], intermediate_results, scale=4
        )
        for intermediate_results in outputs.intermediate_result
    ]
    frames += [frames[-1]] * 15
    clip = mpy.ImageSequenceClip(frames, fps=30)
    
    clip.write_videofile(f"{savedir}/video_{dataname}_{problem_id:04d}.mp4")


if __name__ == "__main__":
    main()
