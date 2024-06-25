import os
import sys
import torch
import rospkg
import pypose as pp

from dastar import *
import encoder

class iastar(nn.Module):
    def __init__(self,
                 g_ratio: float = 0.5,
                 Tmax: float = 1,
                 device:str = "cpu",
                 encoder_input:str = 'm+',
                 encoder_arch:str = 'CNN',
                 encoder_depth: int = 4,
                 learn_obstacles:bool = False,
                 const: float = None,
                 is_training:bool = True,
                 store_intermediate_results: bool = False,
                 output_path_list = False,
                 w:float = 1.0):
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
                              is_training=is_training)

    def init_encoder(self):
        e_arch = getattr(encoder, self.encoder_arch)
        self.encoder = e_arch(len(self.encoder_input),
                              self.encoder_depth,
                              self.const)

    def init_obstacles_maps(self, maps):
        obstacle_maps = (
            maps if not self.learn_obstacles else torch.ones_like(maps)
            )
        return obstacle_maps

    def encode(self, maps, start_maps, goal_maps):
        inputs = maps
        if "+" in self.encoder_input:
            if maps.shape[-1] == start_maps.shape[-1]:
                inputs = torch.cat((inputs, start_maps + goal_maps), dim=1)
            else:
                upsampler = nn.UpsamplingNearest2d(maps.shape[-2:])
                inputs = torch.cat(
                    (inputs, upsampler(start_maps + goal_maps)),
                    dim=1)
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

