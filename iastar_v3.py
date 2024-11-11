import os
import sys
import torch
import pypose as pp

from dastar_v3 import *
import encoder
from encoder import VGGNet
class iastar(nn.Module):
    def __init__(self,
                 g_ratio: float = 0.5,
                 Tmax: float = 1,
                 device:str = "cpu",
                 encoder_input = 2,
                 encoder_arch:str = 'CNN',
                 encoder_depth: int = 4,
                 learn_obstacles:bool = False,
                 const: float = None,
                 is_training:bool = True,
                 store_intermediate_results: bool = False,
                 output_path_list = False,
                 w:float = 1.0,
                 dis_type = "Euc"):
        super().__init__()
        self.encoder_input = encoder_input
        self.encoder_arch = encoder_arch
        self.encoder_depth = encoder_depth
        self.learn_obstacles = learn_obstacles
        self.const = const
        self.init_encoder()
        self.dastar = dastar(g_ratio=g_ratio,
                              device=device,
                              Tmax=Tmax,
                              w = w,
                              store_intermediate_results=store_intermediate_results,
                              output_path_list=output_path_list,
                              is_training=is_training,
                              dis_type=dis_type)

    def init_encoder(self):
        print("Using %s as encoder", self.encoder_arch)
        e_arch = getattr(encoder, self.encoder_arch)
        if self.encoder_arch == "CNN":
            self.encoder = e_arch(self.encoder_input,
                                self.encoder_depth,
                                self.const)
        elif "FCN" in self.encoder_arch:
            vgg_model = VGGNet(requires_grad=True)
            self.encoder = e_arch(pretrained_net=vgg_model, n_class=1)
        elif self.encoder_arch=="UNet":
            self.encoder = e_arch(3,1)
        elif self.encoder_arch=="UNetAtt":
            self.encoder = e_arch(3)

    def init_obstacles_maps(self, maps):
        obstacle_maps = (
            maps if not self.learn_obstacles else torch.ones_like(maps)
            )
        return obstacle_maps

    def encode(self, maps, start_maps, goal_maps):
        if self.encoder_input==3:
            inputs = torch.cat((maps, start_maps, goal_maps), dim=1)
        elif self.encoder_input==2:
            inputs = torch.cat((maps, start_maps + goal_maps), dim=1)
        cost_maps = self.encoder(inputs)
        return cost_maps

    def forward(
        self,
        maps:torch.tensor,
        start_maps:torch.tensor,
        goal_maps:torch.tensor
    ):
        return self.dastar(
            self.encode(maps, start_maps, goal_maps),
            start_maps,
            goal_maps,
            self.init_obstacles_maps(maps)
        )

