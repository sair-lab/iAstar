import os
import re
import sys
import xlwt
import glob
import math
import torch
print(sys.path)
import pandas as pd
from jps.jps import JPS
path = os.path.join(sys.path[0],'iastar') 
path1 = os.path.join(sys.path[0], 'TransPath')
sys.path.append(path)
sys.path.append(path1)
print(sys.path)
from neural_astar.planner import NeuralAstar, VanillaAstar
from iastar import iastar
from dastar import dastar
import matplotlib.pyplot as plt
from neural_astar.utils.data import create_dataloader
import torch.nn.functional as F
import torch.nn as nn

import numpy as np

sys.path.append('./transpath/')
def load_from_ptl_checkpoint(checkpoint_path: str, device='cuda:0') -> dict:
    """
    Load model weights from PyTorch Lightning checkpoint.

    Args:
        checkpoint_path (str): (parent) directory where .ckpt is stored.

    Returns:
        dict: model state dict
    """

    ckpt_file = sorted(glob.glob(f"{checkpoint_path}/**/*.ckpt", recursive=True))[-1]
    print(f"load {ckpt_file}")
    state_dict = torch.load(ckpt_file,map_location = device)["state_dict"]
    state_dict_extracted = dict()
    for key in state_dict:
        if "planner" in key:
            state_dict_extracted[re.split("planner.", key)[-1]] = state_dict[key]

    return state_dict_extracted

def cal_opt(opt_traj:torch.Tensor, target_traj:torch.Tensor):
    return (opt_traj*target_traj).sum()/opt_traj.sum()

def cal_exp(opt_his:torch.Tensor, target_his:torch.Tensor):
    o_area = opt_his.sum() 
    t_area = target_his.sum()
    return (o_area-t_area)/o_area

def cal_al(target_outputs):
    target_traj = target_outputs.paths
    target_length = cal_length(target_traj)
    target_his = target_outputs.histories
    target_area = target_his.sum()

    return target_length + target_area**0.5

def cal_length(traj):
    path_kernel = torch.tensor([[[1.4142,1.,1.4142,],[1.,0.,1.],[1.4142, 1.,1.4142]]], device=traj.device).expand(1,1, 3, 3)
    pad = nn.ReplicationPad2d(padding=(1,1,1,1))
    return (F.conv2d(pad(traj.float()).float(), path_kernel)*traj).sum()/2


def save_results(results:dict, filename = "results.xls"):

    myEx = xlwt.Workbook(encoding="utf-8")
    total_opt = {'neuralAstar':0,
                 'vastar':0,
                 'iAstar':0,
                 'iAstarS':0,
                 'fw':0,
                 'w':0,
                 'cf':0}
    total_exp = {'neuralAstar':0,
                 'vastar':0,
                 'iAstar':0,
                 'iAstarS':0,
                 'fw':0,
                 'w':0,
                 'cf':0}
    total_time = {'neuralAstar':0,
                 'vastar':0,
                 'iAstar':0,
                 'iAstarS':0,
                 'fw':0,
                 'w':0,
                 'cf':0}
    total_len = {'neuralAstar':0,
                 'vastar':0,
                 'iAstar':0,
                 'iAstarS':0,
                 'fw':0,
                 'w':0,
                 'cf':0}

    for i in results:
        mySheet = myEx.add_sheet(i)
        mySheet.write(0,1, label="opt")
        mySheet.write(0,2, label="exp")
        mySheet.write(0,3, label="time")
        mySheet.write(0,4, label="length")
        mySheet.write(0,5, label="al")
        mySheet.write(0,6, label="opt_std")
        mySheet.write(0,7, label="exp_std")
        mySheet.write(0,8, label="time_std")
        mySheet.write(0,9, label="length_std")
        mySheet.write(0,10, label="al_std")
        k=1
        for j in ['neuralAstar','vastar','iAstar','iAstarS',"fw",'w','cf']:
            mySheet.write(k,0, j)
            total_opt[j] += results[i][0][j]
            mySheet.write(k, 1, results[i][0][j])
            total_exp[j] += results[i][1][j]
            mySheet.write(k, 2, results[i][1][j])
            total_time[j] += results[i][2][j]
            mySheet.write(k, 3, results[i][2][j])
            total_len[j] += results[i][3][j]
            mySheet.write(k, 4, results[i][3][j])
            
            mySheet.write(k, 5, results[i][4][j])
            mySheet.write(k, 6, results[i][5][j])
            mySheet.write(k, 7, results[i][6][j])
            mySheet.write(k, 8, results[i][7][j])
            mySheet.write(k, 9, results[i][8][j])
            mySheet.write(k, 10, results[i][9][j])
            k=k+1

    mySheet = myEx.add_sheet("Total")
    mySheet.write(0,1, label="opt")
    mySheet.write(0,2, label="exp")
    mySheet.write(0,3, label="time")
    mySheet.write(0,4, label="length")
    mySheet.write(0,5, label="al")
    mySheet.write(0,6, label="opt_std")
    mySheet.write(0,7, label="exp_std")
    mySheet.write(0,8, label="time_std")
    mySheet.write(0,9, label="length_std")
    mySheet.write(0,10, label="al_std")
    mm=1
    for j in ['neuralAstar','vastar','iAstar','iAstarS',"fw",'w','cf']:
        mySheet.write(mm,0, label=j)
        mySheet.write(mm,1, total_opt[j] )
        mySheet.write(mm,2, total_exp[j] )
        mySheet.write(mm,3, total_time[j] )
        mySheet.write(mm,4, total_len[j] )
        mm+=1
    print("filename is", filename)
    myEx.save(filename)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"
print("device is", device)
with torch.no_grad():
    
    neural_astar = NeuralAstar(encoder_arch='CNN').to(device)
    neural_astar.load_state_dict(load_from_ptl_checkpoint("/home/xyc/cxy/iAstar/model/nastar/mazes_032_moore_c8/lightning_logs/version_0/",
                                                        device=device))
    # ia_star = iastar(encoder_arch='CNN',
    #                 is_training=False,
    #                 output_path_list=False).to(device)
    # ia_star.load_state_dict(load_from_ptl_checkpoint("/home/cxy/iastar/iastar_planner/iastar_planner-main/scripts/iastar/model/mazes_032_moore_c8/lightning_logs/version_74/"))

    ia_star = iastar(encoder_input=3,
                    encoder_arch="UNet",
                    device=device,
                    encoder_depth=4,
                    learn_obstacles=False,
                    is_training = False,
                    output_path_list= False,
                    w=2.0).to(device)
    state_dict = torch.load("/home/xyc/cxy/iAstar/model/iastar/iastar1UNet.pkl", map_location=device)["model_state_dict"]
    ia_star.encoder.load_state_dict(state_dict)
    vanilla_astar = dastar(device=device)

    iastarS = VanillaAstar().to(device)
    # iastarS.load_state_dict(load_from_ptl_checkpoint("/home/cxy/iastar/iastar_planner/iastar_planner-main/scripts/iastar/model/mazes_032_moore_c8/supervised/lightning_logs/version_1"))


    from transpath.models.autoencoder import Autoencoder
    from transpath.modules.planners import DifferentiableDiagAstar
    from data_loader import create_dataloader
    # cf_planner = DifferentiableDiagAstar(mode='k').to(device)
    w2_planner = DifferentiableDiagAstar(mode='default', h_w=2).to(device)
    # model_focal = Autoencoder(mode='f').to(device)
    # model_focal.load_state_dict(torch.load('/home/xyc/cxy/iAstar/transpath/weights/focal.pth'))
    # model_cf = Autoencoder(mode='k').to(device)
    # model_cf.load_state_dict(torch.load('/home/xyc/cxy/iAstar/transpath/weights/cf.pth'))
    # fw100_planner = DifferentiableDiagAstar(mode='f', f_w=100).to(device)

    from neural_astar.utils.data import visualize_results
    import time
def calculate_path_length(points):
    """Calculate the total path length for a given list of (x, y) points."""
    if len(points) < 2:
        return 0.0

    total_length = 0.0
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        distance = math.hypot(x2 - x1, y2 - y1)
        total_length += distance
    return total_length

def visualize(na_outputs, ia_outputs, va_outputs, map_designs, device):
    fig, axes = plt.subplots(5, 1, figsize=[120, 80])
    axes[0].imshow(visualize_results(map_designs, na_outputs))
    axes[0].set_title("Neural A*")
    axes[0].axis("off")
    axes[1].imshow(visualize_results(map_designs, ia_outputs))
    axes[1].set_title("iA*")
    axes[1].axis("off")
    axes[2].imshow(visualize_results(map_designs, va_outputs))
    axes[2].set_title("Vanilla A*")
    axes[2].axis("off")
    inputs = torch.cat((map_designs, start_maps + goal_maps),dim = 1)
    maps1 = neural_astar.encoder(inputs.to(device))
    axes[3].imshow(visualize_results(maps1.to("cpu"), na_outputs))
    axes[3].set_title("Neural A*")
    axes[3].axis("off")
    inputs = torch.cat((map_designs, start_maps + goal_maps),dim = 1)
    maps2 = ia_star.encoder(inputs.to(device))
    axes[4].imshow(visualize_results(maps2.to("cpu"), ia_outputs))
    axes[4].set_title("iA*")
    axes[4].axis("off")

n = 2
t_na = np.zeros(n)
t_ia = np.zeros(n)
t_va = np.zeros(n)
t_ias = np.zeros(n)
cc_na = np.zeros(n)
cc_ia = np.zeros(n)
cc_va = np.zeros(n)
cc_ias = np.zeros(n)
num_na = np.zeros(n)
num_ia = np.zeros(n)
num_va = np.zeros(n)
num_ias = np.zeros(n)
opts_na = np.zeros(n)
opts_ia = np.zeros(n)
opts_va = np.zeros(n)
opts_ias = np.zeros(n)
exps_na = np.zeros(n)
exps_ia = np.zeros(n)
exps_va = np.zeros(n)
exps_ias = np.zeros(n)
lens_na = np.zeros(n)
lens_ia = np.zeros(n)
lens_va = np.zeros(n)
lens_ias = np.zeros(n)
t_fw = np.zeros(n)
num_fw = np.zeros(n)
opts_fw = np.zeros(n)
exps_fw = np.zeros(n)
lens_fw = np.zeros(n)
t_w = np.zeros(n)
num_w = np.zeros(n)
opts_w = np.zeros(n)
exps_w = np.zeros(n)
lens_w = np.zeros(n)
t_cf = np.zeros(n)
num_cf = np.zeros(n)
opts_cf = np.zeros(n)
exps_cf = np.zeros(n)
lens_cf = np.zeros(n)

als_va = np.zeros(n)
als_ia = np.zeros(n)
als_na = np.zeros(n)
als_ias = np.zeros(n)
als_fw = np.zeros(n)
als_w = np.zeros(n)
als_cf = np.zeros(n)
# for kk in range(10,20,1):

env_list = ["maze"]
map_num = 2

for map_size in ["2566"]:
    with torch.no_grad():
        result_dict = {}
        jps_results = {}
        for env in env_list:
            files = glob.glob("/home/xyc/cxy/iAstar/planning-datasets/maze/instances/256/maze_256_dataset.npz")

            for filename in files:
                print(filename)
                jps_path_length = []
                jps_visited_count = []
                dataloader = create_dataloader(filename, "test", map_num, shuffle=True)
                for i in range(n):
                    # while True:
                    print(i)
                    map_designs, start_maps, goal_maps, _ = next(iter(dataloader))

                    vanilla_astar.eval()
                    t1 = time.time()
                    va_outputs = vanilla_astar(map_designs.to(device), start_maps.to(device), goal_maps.to(device))
                    t2 = time.time()
                    t_va[i] = t2 - t1

                    start_points = torch.nonzero(start_maps, as_tuple=False)
                    goal_points = torch.nonzero(goal_maps, as_tuple=False)
                    neural_astar.eval()
                    t1 = time.time()
                    na_outputs = neural_astar(map_designs.to(device), start_maps.to(device), goal_maps.to(device))
                    t2 = time.time()
                    t_na[i] = (t_va[i] - (t2 - t1))/t_va[i]
                    num_na[i] = na_outputs.histories.sum()


                    ia_star.eval()
                    t1 = time.time()
                    ia_outputs = ia_star(map_designs.to(device), start_maps.to(device), goal_maps.to(device))
                    t2 = time.time()
                    t_ia[i] = (t_va[i] - (t2 - t1))/t_va[i]
                    num_ia[i] = ia_outputs.histories.sum()
                    iastarS.eval()
                    t1 = time.time()
                    iaS_outputs = iastarS(map_designs.to(device), start_maps.to(device), goal_maps.to(device))
                    t2 = time.time()
                    t_ias[i] = (t_va[i] - (t2 - t1))/t_va[i]
                    num_ias[i] = iaS_outputs.histories.sum()
                    inputs_g = torch.cat([map_designs.to(device), goal_maps.to(device)], dim=1)
                    inputs_sg = torch.cat([((map_designs-1)**2).to(device), start_maps.to(device) + goal_maps.to(device)], dim=1)
                    t1 = time.time()
                    if map_size == "064":
                        model_focal.eval()
                        pred_f = (model_focal(inputs_sg) + 1) / 2
                        outputs_fw100 = fw100_planner(
                                        pred_f.to(device),
                                        start_maps.to(device),
                                        goal_maps.to(device),
                                        map_designs.to(device))
                    t2 = time.time()
                    t_fw[i] = t2 - t1

                    t1 = time.time()
                    if map_size == "064":
                        model_cf.eval()
                        cf_planner.eval()
                        pred_cf = (model_cf(inputs_g) + 1) / 2
                        outputs_cf = cf_planner(
                            pred_cf.to(device),
                            start_maps.to(device),
                            goal_maps.to(device),
                            map_designs.to(device))
                        cf_traj = outputs_cf.paths.to(device)
                    t2 = time.time()
                    t_cf[i] = t2 - t1

                    t1 = time.time()
                    w2_planner.eval()
                    outputs_w= w2_planner(map_designs.to(device),
                                        start_maps.to(device),
                                        goal_maps.to(device),
                                        map_designs.to(device))
                    t2 = time.time()
                    t_w[i] = (t_va[i] - (t2 - t1))/t_va[i]

                    
                    na_traj = na_outputs.paths.to(device)
                    ia_traj = ia_outputs.paths.to(device)
                    va_traj = va_outputs.paths.to(device)
                    iaS_traj = iaS_outputs.paths.to(device)
                    
                    w_traj = outputs_w.paths.to(device)
                    fw100_traj = outputs_fw100.paths.to(device) if map_size == "064" else None
                    cf_traj = outputs_cf.paths.to(device) if map_size == "064" else None

                    opt_ia = cal_opt(va_traj, ia_traj)
                    opt_iaS = cal_opt(va_traj, iaS_traj)
                    opt_na = cal_opt(va_traj, na_traj)
                    opt_va = cal_opt(va_traj, va_traj)
                    opt_w = cal_opt(va_traj, w_traj)
                    if map_size == "064":
                        opt_fw = cal_opt(va_traj, fw100_traj)
                        opt_cf = cal_opt(va_traj, cf_traj)

                    exp_ia = cal_exp(va_outputs.histories, ia_outputs.histories)
                    exp_iaS = cal_exp(va_outputs.histories, iaS_outputs.histories)
                    exp_na = cal_exp(va_outputs.histories, na_outputs.histories)
                    exp_va = cal_exp(va_outputs.histories, va_outputs.histories)
                    exp_w = cal_exp(va_outputs.histories, outputs_w.histories)
                    if map_size == "064":
                        exp_fw = cal_exp(va_outputs.histories, outputs_fw100.histories)
                        exp_cf = cal_exp(va_outputs.histories, outputs_cf.histories)


                    len_ia = cal_length(ia_traj)
                    len_iaS = cal_length(iaS_traj)
                    len_na = cal_length(na_traj)
                    len_va = cal_length(va_traj)
                    len_w = cal_length(w_traj)
                    if map_size == "064":
                        len_fw = cal_length(fw100_traj)
                        len_cf = cal_length(cf_traj)
                        
                    al_ia = cal_al(ia_outputs)
                    al_iaS = cal_al(iaS_outputs)
                    al_na = cal_al(na_outputs)
                    al_va = cal_al(va_outputs)
                    al_w = cal_al(outputs_w)
                    if map_size == "064":
                        al_fw = cal_al(outputs_fw100)
                        al_cf = cal_al(outputs_cf)  


                    opts_ia[i] = opt_ia 
                    opts_na[i] = opt_na
                    opts_va[i] = opt_va
                    exps_ia[i] = exp_ia 
                    exps_na[i] = exp_na
                    exps_va[i] = exp_va
                    lens_ia[i] = len_ia 
                    lens_na[i] = len_na
                    lens_va[i] = len_va
                    opts_ias[i] = opt_iaS
                    exps_ias[i] = exp_iaS
                    lens_ias[i] = len_iaS
                    opts_w[i] = opt_w
                    exps_w[i] = exp_w
                    lens_w[i] = len_w
                    als_ia[i] = al_ia
                    als_ias[i] = al_iaS
                    als_na[i] = al_na
                    als_va[i] = al_va
                    als_w[i] = al_w
                    

                    if map_size == "064":
                        opts_fw[i] = opt_fw
                        exps_fw[i] = exp_fw
                        lens_fw[i] = len_fw 
                        als_fw[i] = al_fw 
                        opts_cf[i] = opt_cf
                        exps_cf[i] = exp_cf
                        lens_cf[i] = len_cf
                        als_cf[i] = al_cf

                    # for j in range(map_num):
                        
                    #     map_data = (map_designs[j][0].to(torch.int).numpy()==0).astype(np.uint8)
                    #     # print(f"Processing map data for {filename} with shape {map_data}")
                    #     plt.imshow(map_data, cmap='gray')
                    #     start_pos = start_points[j].numpy().astype(np.uint8)
                    #     goal_pos = goal_points[j].numpy().astype(np.uint8)
                    #     print(map_data[start_pos[-2], start_pos[-1]], map_data[goal_pos[-2], goal_pos[-1]])
                    #     # Convert start_pos and goal_pos to lists for JSON serialization
                    #     # print(f"Processing {filename} with start {start_pos} and goal {goal_pos}")
                    #     jps = JPS(map_data)
                    #     start = (start_pos[-2], start_pos[-1])
                    #     goal = (goal_pos[-2], goal_pos[-1])
                    #     path, visited_count = jps.find_path((start_pos[-2], start_pos[-1]), (goal_pos[-2], goal_pos[-1]))
                    #     path_length  = calculate_path_length(path)
                    #     jps_path_length.append(path_length)
                    #     jps_visited_count.append(visited_count)
                    # torch.clear_autocast_cache()
                    # torch.cuda.empty_cache()
                
                    
                times = {"neuralAstar":t_na.mean()/map_num, 
                        "vastar":t_va.mean()/map_num,
                        "iAstar":t_ia.mean()/map_num,
                        "iAstarS":t_ias.mean()/map_num, 
                        "fw":t_fw.mean()/map_num,
                        "w":t_w.mean()/map_num,
                        "cf":t_cf.mean()/map_num}
                opts = {"neuralAstar":opts_na.mean(), 
                        "vastar":opts_va.mean(),
                        "iAstar":opts_ia.mean(), 
                        "iAstarS":opts_ias.mean(), 
                        "fw":opts_fw.mean(),
                        "w":opts_w.mean(),
                        "cf":opts_cf.mean()}
                exps = { "neuralAstar":exps_na.mean(), 
                        "vastar":exps_va.mean(),
                        "iAstar":exps_ia.mean(), 
                        "iAstarS":exps_ias.mean(),
                        "fw":exps_fw.mean(),
                        "w":exps_w.mean(),
                        "cf":exps_cf.mean()}
                lengths = {"neuralAstar":lens_na.mean()/map_num, 
                        "vastar":lens_va.mean()/map_num,
                        "iAstar":lens_ia.mean()/map_num, 
                        "iAstarS":lens_ias.mean()/map_num,
                        "fw":lens_fw.mean()/map_num,
                        "w":lens_w.mean()/map_num,
                        "cf":lens_cf.mean()/map_num}
                al = {"neuralAstar":als_na.mean()/map_num,
                        "vastar":als_va.mean()/map_num,
                        "iAstar":als_ia.mean()/map_num, 
                        "iAstarS":als_ias.mean()/map_num,
                        "fw":als_fw.mean()/map_num,
                        "w":als_w.mean()/map_num,
                        "cf":als_cf.mean()/map_num}
                times_std = { "neuralAstar":t_na.std()/map_num, 
                        "vastar":t_va.std()/map_num,
                        "iAstar":t_ia.std()/map_num,
                        "iAstarS":t_ias.std()/map_num, 
                        "fw":t_fw.std()/map_num,
                        "w":t_w.std()/map_num,
                        "cf":t_cf.std()/map_num}
                opts_std = {"neuralAstar":opts_na.std(), 
                        "vastar":opts_va.std(),
                        "iAstar":opts_ia.std(), 
                        "iAstarS":opts_ias.std(), 
                        "fw":opts_fw.std(),
                        "w":opts_w.std(),
                        "cf":opts_cf.std()}
                exps_std = { "neuralAstar":exps_na.std(), 
                        "vastar":exps_va.std(),
                        "iAstar":exps_ia.std(), 
                        "iAstarS":exps_ias.std(),
                        "fw":exps_fw.std(),
                        "w":exps_w.std(),
                        "cf":exps_cf.std()}
                lengths_std = {"neuralAstar":lens_na.std()/map_num, 
                        "vastar":lens_va.std()/map_num,
                        "iAstar":lens_ia.std()/map_num, 
                        "iAstarS":lens_ias.std()/map_num,
                        "fw":lens_fw.std()/map_num,
                        "w":lens_w.std()/map_num,
                        "cf":lens_cf.std()/map_num}
                al_std = {"neuralAstar":als_na.std()/map_num,
                        "vastar":als_va.std()/map_num,
                        "iAstar":als_ia.std()/map_num, 
                        "iAstarS":als_ias.std()/map_num,
                        "fw":als_fw.std()/map_num,
                        "w":als_w.std()/map_num,
                        "cf":als_cf.std()/map_num}
                tmpnames = filename.split("/")[-1].split("_")
                excelname = ''.join(tmpnames)
                jps_path_length_arr = np.array(jps_path_length)
                jps_visited_count_arr = np.array(jps_visited_count)
                jps_re = {"path_length": jps_path_length_arr.mean(), 
                                        "path_length_std": jps_path_length_arr.std(),
                                        "visited_count": jps_visited_count_arr.mean(),
                                        "visited_count_std": jps_visited_count_arr.std()}
                jps_results[excelname] = jps_re
                # print(jps_re)
                result_dict[excelname] = [opts, exps, times, lengths, al, 
                                        opts_std, exps_std, times_std, lengths_std, al_std]
        df = pd.DataFrame(jps_results)
        df.to_excel(f'jps0723{env}{map_size}.xlsx', index=False)
        save_results(result_dict, f"mm0725{env}{map_size}Unet.xls")

print("Time(NA):", t_na.mean())
print("Time(IA):", t_ia.mean())
print("Time(VA):", t_va.mean())
print("Search area of NA is ", exp_na.mean())
print("Search area of IA is ", exp_ia.mean())
# print("Search area of CF is ", exp_cf.mean())
print("Search area of NA is ", torch.sqrt((1-exp_na.mean())*64*64)+np.mean(lens_na))
print("Search area of IA is ",  torch.sqrt((1 - exp_ia.mean())*64*64)+np.mean(lens_ia))
# print("Search area of CF is ",  torch.sqrt((1-exp_cf.mean())*64*64)+np.mean(lens_cf))

