import os
import json
import time
import wandb
import torch
import yaml
import random

import argparse
from iastar import iastar
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from data_loader import *
from torch.utils.data import Dataset, DataLoader
from torchutil import EarlyStopScheduler
class EncoderTrainer():

    def __init__(self) -> None:
        # self.root_folder = os.getenv('EXPERIMENT_DIRECTORY', os.getcwd())
        self.root_folder = "/home/cxy/SAIRLAB/"
        self.load_config()
        self.parse_args()
        self.init_wandb()
        self.prepare_data()
        self.prepare_planner()
        
    
    def init_wandb(self):
        # Convert to string in the format you prefer
        date_time_str = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        # Initialize wandb
        self.wandb_run = wandb.init(
            # set the wandb project where this run will be logged
            project="imperative-path-planning",
            # Set the run name to current date and time
            name=date_time_str + "adamW",
            config={
                "learning_rate": self.args.lr,
                "architecture": "PlannerNet",  # Replace with your actual architecture
                # "dataset": self.args.data_root,  # Assuming this holds the dataset name
                "epochs": self.args.epochs,
                # "goal_step": self.args.goal_step,
                # "max_episode": self.args.max_episode,
                # "fear_ahead_dist": self.args.fear_ahead_dist,
            }
        )

    

        
    def parse_args(self):  
        parser = argparse.ArgumentParser(description="Training config for iA*")
        # dataConfig
        parser.add_argument("--datapath", type=str, default=os.path.join(self.root_folder,'iAstar', self.config['dataset']), help="dataset root folder")
        # parser.add_argument('--env-id', type=str, default=self.config['dataConfig'].get('env-id'), help='environment id list')

        # modelConfig
        parser.add_argument("--model-save", type=str, default=os.path.join(self.root_folder, 'iAstar', self.config['modelConfig'].get('model-save')), help="model save point")
        parser.add_argument('--resume', type=str, default=self.config['modelConfig'].get('resume'))

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
        # logConfig
        parser.add_argument("--log-save", type=str, default=os.path.join(self.root_folder, 'iAstar', self.config['logpath']), help="train log file")
        # parser.add_argument('--test-env-id', type=int, default=self.config['logConfig'].get('test-env-id'), help='the test env id in the id list')

        self.args = parser.parse_args()


    def load_config(self):
        filepath = os.path.join(os.path.dirname(self.root_folder),'iAstar','config', 'config.yaml')
        with open(filepath,mode="r",encoding="utf-8") as f:
            self.config = yaml.load(f.read(), Loader=yaml.FullLoader)

    def prepare_data(self):
        self.dataloader = create_dataloader(self.args.datapath, "train", self.args.map_num)
        self.val_dataloader = create_dataloader(self.args.datapath, "valid", self.args.map_num)

    
    def train_epoch(self, epoch):
        loss_sum = 0.0
        # for env in self.env_list:
        for i in range(100):
            maps, start, goal, _ = next(iter(self.dataloader))
            self.optimizer.zero_grad()
            planner_outputs = self.planner(maps.to(self.device), 
                        start.to(self.device), 
                        goal.to(self.device))
            loss = self.CostofTraj(maps, planner_outputs)
            loss.backward()
            self.optimizer.step()
            train_loss = loss.item()
            loss_sum += train_loss
            wandb.log({"Running Loss": train_loss})
            
        return loss_sum

    def CostofTraj(self, maps, outputs,alpha=0.1, beta=0.9):
        paths = outputs.paths
        print(outputs.histories)
        area_loss = nn.L1Loss()(outputs.histories,
                                   torch.zeros(outputs.histories.shape,
                                               device=outputs.histories.device))
        pad = nn.ReplicationPad2d(padding=(1,1,1,1))
        # map_f = F.conv2d(pad((maps<0.5)*1.0), self.o_kernel)
        # oloss = nn.L1Loss()(map_f*paths,
        #                     torch.zeros(map_designs.shape,
        #                                 device=map_designs.device))
        pad = nn.ZeroPad2d(padding=(1,1,1,1)).to(self.device)
        path_length = F.conv2d(pad(paths).float(), self.path_kernel)
        length_loss = nn.L1Loss()(path_length*paths,
                      torch.zeros(maps.shape,
                                  device=self.device))

        print(alpha*area_loss + beta*length_loss)
        return alpha*area_loss + beta*length_loss
    
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
        encoder_input=self.config["encoder"]["input"],
        encoder_arch="CNN",
        device=self.device,
        encoder_depth=self.config["encoder"]["depth"],
        learn_obstacles=False,
        Tmax=self.config["Tmax"],
        is_training = True,
        output_path_list= False,
        w=1.0).to(self.device)
        self.optimizer = optim.AdamW(self.planner.parameters(), lr=self.args.lr, weight_decay=self.args.w_decay)
        self.scheduler = EarlyStopScheduler(self.optimizer, factor=self.args.factor, verbose=True, min_lr=self.args.min_lr, patience=self.args.patience)


    def train(self):

        # Convert to string in the format you prefer
        date_time_str = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        
        self.args.log_save += ('/'+date_time_str + ".txt")
        open(self.args.log_save, 'w').close()

        for epoch in range(self.args.epochs):
            start_time = time.time()
            print("Epoch is ", epoch)
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate()
            duration = (time.time() - start_time) / 60 # minutes

            self.log_message("Epoch: %d | Training Loss: %f | Val Loss: %f | Duration: %f" % (epoch, train_loss, val_loss, duration))
            # Log metrics to wandb
            wandb.log({"Avg Training Loss": train_loss, "Validation Loss": val_loss, "Duration (min)": duration})
            
            if val_loss < self.best_loss:
                self.log_message("Save model of epoch %d" % epoch)
                torch.save((self.net, val_loss), self.args.model_save)
                self.best_loss = val_loss
                self.log_message("Current val loss: %.4f" % self.best_loss)
                self.log_message("Epoch: %d model saved | Current Min Val Loss: %f" % (epoch, val_loss))

            self.log_message("------------------------------------------------------------------------")
            if self.scheduler.step(val_loss):
                self.log_message('Early Stopping!')
                break
            
        # Close wandb run at the end of training
        self.wandb_run.finish()

    def evaluate(self):
        maps, start, goal, _ = next(iter(self.val_dataloader))
        print(maps.shape)
        print(self.device)
        outputs_iastar = self.planner(maps.to(self.device), 
                        start.to(self.device), 
                        goal.to(self.device))
        return self.CostofTraj(maps, outputs_iastar)

    def cal_loss(self):
        loss = 0.0
        return loss
    
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