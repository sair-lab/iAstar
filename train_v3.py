import os
import json
import time
import wandb
import torch
import yaml
import random
import glob
import argparse
from iastar_v3 import iastar
from os.path import isdir
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from data_loader import *
from torch.utils.data import Dataset, DataLoader
from torchutil import EarlyStopScheduler

root_folder = "/home/cxy/"
class EncoderTrainer():

    def __init__(self) -> None:
        # self.root_folder = os.getenv('EXPERIMENT_DIRECTORY', os.getcwd())
        self.root_folder = root_folder
        self.load_config()
        self.parse_args()
        self.init_wandb()
        self.prepare_data()
        self.prepare_planner()
        
    
    def init_wandb(self):
        # Convert to string in the format you prefer
        date_time_str = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        n = self.args.encoder_arch + "w" + str(self.args.w)
        if self.args.useIL:
            n += 'IL'
        print("Using ", n)
        # Initialize wandb
        self.wandb_run = wandb.init(
            # set the wandb project where this run will be logged
            project="imperative-path-planning",
            # Set the run name to current date and time
            name=date_time_str + n + "adamWUP",
            config={
                "learning_rate": self.args.lr,
                "architecture": "PlannerNet",  # Replace with your actual architecture
                "epochs": self.args.epochs,
            }
        )

    def parse_args(self):  
        parser = argparse.ArgumentParser(description="Training config for iA*")
        # dataConfig
        parser.add_argument("--datapath", type=str, default=os.path.join(self.root_folder,'iAstar', self.config['dataset']), help="dataset root folder")
        parser.add_argument("--val-datapath", type=str, default=os.path.join(self.root_folder,'iAstar', self.config['val_dataset']), help="dataset root folder")

        # modelConfig
        parser.add_argument("--model-save", type=str, default=os.path.join(self.root_folder, 'iAstar', self.config['modelConfig'].get('model-save')), help="model save point")
        parser.add_argument('--resume', type=str, default=self.config['modelConfig'].get('resume'))
        parser.add_argument("--input-dim", type=int, default=self.config["encoder"].get('input'))
        parser.add_argument("--encoder-arch", type=str, default=self.config['encoder'].get('arch'), help="The archtecture of the embedded network, CNN, UNet, FCNs")
        # trainingConfig
        parser.add_argument('--training', type=str, default=self.config['params'].get('training'))
        parser.add_argument("--lr", type=float, default=self.config['params'].get('lr'), help="learning rate")
        parser.add_argument("--min-lr", type=float, default=self.config['params'].get('min_lr'), help="minimum lr for ReduceLROnPlateau")
        parser.add_argument("--patience", type=int, default=self.config['params'].get('patience'), help="patience of epochs for ReduceLROnPlateau")
        parser.add_argument("--epochs", type=int, default=self.config['params'].get('num_epochs'), help="number of training epochs")
        parser.add_argument("--batch-size", type=int, default=self.config['params'].get('batch_size'), help="number of minibatch size")
        parser.add_argument("--gpu-id", type=int, default=self.config['params'].get('gpu_id'), help="GPU id")
        parser.add_argument("--w-decay", type=float, default=self.config['params'].get('w_decay'), help="weight decay of the optimizer")
        parser.add_argument("--factor", type=float, default=self.config['params'].get('factor'), help="ReduceLROnPlateau factor")
        parser.add_argument("--map-num", type=int, default=self.config['params'].get('map_num'), help="num of maps")
        parser.add_argument("--useIL", type=bool, default=True)
        parser.add_argument("--w", type=float, default=1.0)
        # logConfig
        parser.add_argument("--log-save", type=str, default=os.path.join(self.root_folder, 'iAstar', self.config['logpath']), help="train log file")
        self.args = parser.parse_args()

    def load_config(self):
        filepath = os.path.join(os.path.dirname(self.root_folder),'iAstar','config', 'config.yaml')
        with open(filepath,mode="r",encoding="utf-8") as f:
            self.config = yaml.load(f.read(), Loader=yaml.FullLoader)

    def prepare_data(self):
        self.train_env_list = glob.glob(self.args.datapath+"*.npz") 
        self.val_env_list = glob.glob(self.args.val_datapath+"*.npz")
        print("Using for training", self.train_env_list)       
        print("Using for val", self.val_env_list) 
    def train_epoch(self, epoch):
        loss_sum = 0.0
        for env in self.train_env_list:
            print(env)
            self.dataloader = create_dataloader(env, "train", self.args.map_num)
            for i in range(5):
                maps, start, goal, opt_trajs = next(iter(self.dataloader))
                self.optimizer.zero_grad()
                planner_outputs = self.planner(maps.to(self.device), 
                                               start.to(self.device), 
                                               goal.to(self.device))
                a_loss, l_loss = self.getLoss(planner_outputs)
                if self.args.useIL:
                    loss = self.CostofTraj(planner_outputs)
                else:
                    loss = nn.L1Loss()(planner_outputs.histories.to(self.device), 
                                       opt_trajs.to(self.device))
                loss.backward()
                self.optimizer.step()
                train_loss = loss.item()
                loss_sum += train_loss
                wandb.log({"Running Loss": train_loss, 
                           "Search Area":a_loss, 
                           "Path Length":l_loss,
                           "Search Area + Path Length": (a_loss+l_loss)})
            self.evaluate()
        loss_sum = loss_sum/len(self.train_env_list)/5  
        return loss_sum

    def CostofTraj(self, outputs):
        paths = outputs.paths
        area_loss = torch.sum(outputs.histories - paths)/paths.shape[0]
        pad = nn.ReplicationPad2d(padding=(1,1,1,1))
        pad = nn.ZeroPad2d(padding=(1,1,1,1)).to(self.device)
        path_length = F.conv2d(pad(paths).float(), self.path_kernel)
        length_loss = torch.sum(path_length*paths)/2/paths.shape[0]
        return torch.sqrt(area_loss) + length_loss
    
    def getLoss(self, outputs):
        paths = outputs.paths
        # area_loss = F.l1_loss(outputs.histories,
        #                            torch.zeros(outputs.histories.shape,
        #                                        device=outputs.histories.device))
        # pad = nn.ReplicationPad2d(padding=(1,1,1,1))
        # pad = nn.ZeroPad2d(padding=(1,1,1,1)).to(self.device)
        # path_length = F.conv2d(pad(paths).float(), self.path_kernel)
        # length_loss = F.l1_loss(path_length*paths,
        #               torch.zeros(maps.shape,
        #                           device=self.device))
        area_loss = torch.sum(outputs.histories - paths)/paths.shape[0]
        pad = nn.ReplicationPad2d(padding=(1,1,1,1))
        pad = nn.ZeroPad2d(padding=(1,1,1,1)).to(self.device)
        path_length = F.conv2d(pad(paths).float(), self.path_kernel)
        length_loss = torch.sum(path_length*paths)/2/paths.shape[0]
        return torch.sqrt(area_loss), length_loss
    
    def prepare_planner(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:"+str(self.args.gpu_id))
        else:
            self.device = torch.device("cpu")
        self.path_kernel = torch.tensor([[[1.414,1.,1.414,],
                    [1.,0.,1.],
                    [1.414, 1.,1.414]]],
                device = self.device,
                requires_grad=True).expand(1,1, 3, 3)
        self.planner = iastar(
        encoder_input=self.args.input_dim,
        encoder_arch=self.args.encoder_arch,
        device=self.device,
        encoder_depth=self.config["encoder"]["depth"],
        learn_obstacles=False,
        Tmax=self.config["Tmax"],
        is_training = True,
        output_path_list= False,
        w=self.args.w).to(self.device)
        # if self.args.resume == True:
        #     self.net, self.best_loss = torch.load(self.args.model_save, map_location=torch.device("cpu"))
        #     print("Resume training from best loss: {}".format(self.best_loss))
        # else:
        self.best_loss = float('Inf')
        self.optimizer = optim.AdamW(self.planner.parameters(), 
                                     lr=self.args.lr, 
                                     weight_decay=self.args.w_decay)
        self.scheduler = EarlyStopScheduler(self.optimizer, 
                                            factor=self.args.factor, 
                                            verbose=True, 
                                            min_lr=self.args.min_lr, 
                                            patience=self.args.patience)

    def train(self):
        # Convert to string in the format you prefer
        date_time_str = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        self.args.log_save += ('/'+date_time_str + ".txt")
        open(self.args.log_save, 'w').close()
        for epoch in range(self.args.epochs):
            start_time = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate()
            duration = (time.time() - start_time) / 60 # minutes
            self.log_message("Epoch: %d | Training Loss: %f | Val Loss: %f | Duration: %f" % (epoch, train_loss, val_loss, duration))
            # Log metrics to wandb
            wandb.log({"Avg Training Loss": train_loss, "Validation Loss": val_loss, "Duration (min)": duration})
            if val_loss < self.best_loss:
                self.log_message("Save model of epoch %d" % epoch)
                save_model_dir = '{}/{}/{}'.format(self.args.model_save, date_time_str, epoch)
                if not isdir(save_model_dir):
                    os.makedirs(save_model_dir)
                save_model_name = os.path.join(save_model_dir, 'iaster1'+self.args.encoder_arch + '.pkl')
                print("model saved at :", save_model_name)
                torch.save({"epoch":epoch, 
                            "model_state_dict": self.planner.encoder.state_dict(),
                            "optimizer_state_dict":self.optimizer.state_dict(),
                            "train_loss":train_loss,
                            "val_loss":val_loss}, save_model_name)
                self.best_loss = val_loss
                self.log_message("Current val loss: %.4f" % self.best_loss)
                self.log_message("Epoch: %d model saved | Current Min Val Loss: %f" % (epoch, val_loss))
            self.log_message("------------------------------------------------------------------------")
            if self.scheduler.step(val_loss):
                self.log_message("Save model of epoch %d" % epoch)
                save_model_dir = '{}/{}/{}'.format(self.args.model_save, date_time_str, epoch)
                if not isdir(save_model_dir):
                    os.makedirs(save_model_dir)
                save_model_name = os.path.join(save_model_dir, 'iaster1'+self.args.encoder_arch + '.pkl')
                print("model saved at :", save_model_name)
                torch.save({"epoch":epoch, 
                            "model_state_dict": self.planner.encoder.state_dict(),
                            "optimizer_state_dict":self.optimizer.state_dict(),
                            "train_loss":train_loss,
                            "val_loss":val_loss}, save_model_name)
                self.log_message('Early Stopping!')
                break
        # Close wandb run at the end of training
        self.wandb_run.finish()

    def evaluate(self):
        area_l = 0
        length_l = 0
        t_loss = 0
        with torch.no_grad():
            for env in self.val_env_list:
                self.val_dataloader = create_dataloader(env, "valid", 8)
                maps, start, goal, opt_trajs = next(iter(self.val_dataloader))
                planner_outputs = self.planner(maps.to(self.device), 
                                            start.to(self.device),
                                            goal.to(self.device))
                if self.args.useIL:
                    loss = self.CostofTraj(planner_outputs)
                else:
                    loss = F.l1_loss(planner_outputs.histories.to(self.device), 
                                    opt_trajs.to(self.device))
                a_loss, l_loss = self.getLoss(planner_outputs)
                t_loss += loss.item()
                area_l += a_loss.item()
                length_l += l_loss.item()
            wandb.log({"Val Loss": t_loss/len(self.val_env_list), 
                    "Val Search Area": area_l/len(self.val_env_list), 
                    "Val Path Length": length_l/len(self.val_env_list),
                    "Val Search Area + Path Length": (area_l+length_l)/len(self.val_env_list)})
        return t_loss/len(self.val_env_list)
    def log_message(self, message):
        with open(self.args.log_save, 'a') as f:
            f.writelines(message)
            f.write('\n')
        print(message)

def main():
    trainer = EncoderTrainer()
    if trainer.args.training == True:
        trainer.train()
    trainer.evaluate()

if __name__ == "__main__":
    main()