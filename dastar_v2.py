import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _st_softmax_noexp(val: torch.tensor) -> torch.tensor:
    """
    Softmax + discretized activation
    Used a detach() trick as done in straight-through softmax

    Args:
        val (torch.tensor): exponential of inputs.

    Returns:
        torch.tensor: one-hot matrices for input argmax.
    """

    val_ = val.reshape(val.shape[0], -1)
    y = val_ / (val_.sum(dim=-1, keepdim=True))
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y)
    y_hard[range(len(y_hard)), ind] = 1
    y_hard = y_hard.reshape_as(val)
    y = y.reshape_as(val)
    return (y_hard - y).detach() + y

class AstarOutput():
    def __init__(self,
                 histories,
                 paths,
                 intermediate_result:dict=None,
                 path_list:list=None):
        self.histories = histories
        self.paths = paths
        self.intermediate_result = intermediate_result
        self.path_list = path_list

class dastar(nn.Module):
    def __init__(self,
                 maps:torch.tensor = None,
                 device:str = "cuda",
                 start_maps:torch.tensor = None,
                 goal_maps:torch.tensor = None,
                 g_ratio:float = 0.5,
                 w:float = 1.0,
                 Tmax:float = 1.0,
                 store_intermediate_results: bool = False,
                 output_path_list: list = False,
                 is_training = False,
                 dis_type = "Euc"):
        super().__init__()
        self.maps = maps
        self.start_maps = start_maps
        self.goal_maps = goal_maps
        self.cost_maps = None
        self.obstacles_maps = None
        self.g_ratio = g_ratio
        self.Tmax = Tmax
        self.store_intermediate_results = store_intermediate_results
        self.output_path_list = output_path_list
        self.is_training = is_training
        self.device = device
        self.w = w
        self.dis_type = dis_type
        self.init_matrix()

    def init_matrix(self):
        n_c = torch.ones(1, 1, 3, 3, device = self.device)
        n_c[0, 0, 1, 1] = 0
        n_c[0, 0, 0, 0] = 1.4142
        n_c[0, 0, 0, 2] = 1.4142
        n_c[0, 0, 2, 0] = 1.4142
        n_c[0, 0, 2, 2] = 1.4142
        n_c = nn.Parameter(n_c, requires_grad=False)
        self.n_c = n_c
        n_s = torch.ones(1, 1, 3, 3, device = self.device)
        n_s[0, 0, 1, 1] = 0
        n_s = nn.Parameter(n_s, requires_grad=False)
        self.n_s = n_s


    def ind2Map(self, indices):
        if self.obstacle_maps == None:
            print("Don't receive the obstacle_maps!")
            return None
        else:
            temp_map = torch.zeros_like(self.obstacle_maps)
            temp_map[torch.arange[temp_map.shape[0]],0, indices[:,0], indices[:,1]] = 1.0
            return temp_map

    def set_start_map(self, indices):
        self.start_maps = self.ind2Map(indices)
        return self.start_maps!=None

    def set_goal_map(self, indices):
        self.goal_maps = self.ind2Map(indices)
        return self.goal_maps!=None

    def set_cost_maps(self, cost_maps):
        self.cost_maps = cost_maps

    def get_heuristic(self,
                      goal_maps:torch.tensor):
        num_samples, H, W = goal_maps.shape[0], goal_maps.shape[-2], goal_maps.shape[-1]
        grid = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        loc = torch.stack(grid, dim=0).type_as(goal_maps)
        loc_expand = loc.reshape(2, -1).unsqueeze(0).expand(num_samples, 2, -1)
        goal_loc = torch.einsum("kij, bij -> bk", loc, goal_maps)
        goal_loc_expand = goal_loc.unsqueeze(-1).expand(num_samples, 2, -1)
        # Euclid
        if self.dis_type == 'Euc':
            # print('Euc distance')
            euc = torch.sqrt(((loc_expand - goal_loc_expand) ** 2).sum(1))
            h = euc.reshape_as(goal_maps)
        # chebyshev distance
        elif self.dis_type == 'Che':
            # print('Che distance')
            dxdy = torch.abs(loc_expand - goal_loc_expand)
            che_dis = dxdy.sum(dim=1) - dxdy.min(dim=1)[0]
            h = (che_dis).reshape_as(goal_maps)
        elif self.dis_type == 'Diag':
            # print('Diag distance')
            dxdy = torch.abs(loc_expand - goal_loc_expand)
            h = dxdy.min(dim=1)[0] * (2**0.5) + torch.abs(dxdy[:, 0] - dxdy[:, 1])
            h = h.reshape_as(goal_maps)
        return h

    def expand(self, x: torch.tensor, neighbor_filter: torch.tensor) -> torch.tensor:
        """
        Expand neighboring node

        Args:
            x (torch.tensor): selected nodes
            neighbor_filter (torch.tensor): 3x3 filter to indicate 8 neighbors

        Returns:
            torch.tensor: neighboring nodes of x
        """

        x = x.unsqueeze(0)
        y = F.conv2d(x, neighbor_filter, padding=1, groups=self.num_samples).squeeze()
        y = y.squeeze(0)
        return y

    def forward(self,
                cost_maps:torch.tensor,
                start_maps:torch.tensor,
                goal_maps:torch.tensor,
                obstacles_maps:torch.tensor = None
                ):
        cost_maps = cost_maps[:, 0]
        start_maps = start_maps[:, 0]
        goal_maps = goal_maps[:, 0]
        obstacles_maps = obstacles_maps[:, 0] if obstacles_maps!=None else cost_maps
        self.num_samples = cost_maps.shape[0]
        nc = torch.repeat_interleave(self.n_c, self.num_samples, 0)
        ns = torch.repeat_interleave(self.n_s, self.num_samples, 0)

        open_maps = start_maps
        histories = torch.zeros_like(start_maps,
                                     device= self.device)
        intermediate_results = []
        # # 71(Diag) 73(Diag) 74(Euc)
        h = self.get_heuristic(goal_maps)*cost_maps*self.w
        # 72(Diag)
        g = torch.zeros_like(start_maps)
        parents = (
            torch.ones_like(start_maps,
                            device=self.device).reshape(self.num_samples, -1)
            * goal_maps.reshape(self.num_samples, -1).max(-1, keepdim=True)[-1]
        )
        size = start_maps.shape
        Tmax = self.Tmax if self.training else 1.0
        Tmax = int(size[-2]*size[-1]*Tmax)
        for t in range(Tmax):
            # 71
            # f = self.g_ratio * g +(1 - self.g_ratio)*h
            # 72 73
            f = g + h
            f_exp = torch.exp(-1 * f/math.sqrt(size[-2]*size[-1]))
            f_exp = f_exp * open_maps
            node_selection = _st_softmax_noexp(f_exp)
            dist_to_goal = (node_selection*goal_maps).sum((1, 2), 
                                                          keepdim=True)
            is_unsolved = (dist_to_goal < 1e-8).float()

            histories = histories + node_selection
            histories = torch.clamp(histories, 0, 1)
            open_maps = open_maps - is_unsolved * node_selection
            open_maps = torch.clamp(open_maps, 0, 1)

            # open neighboring nodes, add them to the openlist if they satisfy certain requirements
            neighbor_nodes = self.expand(node_selection, ns)
            neighbor_nodes = neighbor_nodes * obstacles_maps

            # update g if one of the following conditions is met
            # 1) neighbor is not in the close list (1 - histories) nor in the open list (1 - open_maps)
            # 2) neighbor is in the open list but g < g2
            g2 = (g*node_selection).sum((1, 2), keepdim=True)
            g2 = g2 + self.expand(node_selection, nc)
            idx = (1 - open_maps) * (1 - histories) + open_maps * (g > g2)
            idx = idx * neighbor_nodes
            idx = idx.detach()
            g = g2 * idx + g * (1 - idx)
            g = g.detach()

            # update open maps
            open_maps = torch.clamp(open_maps + idx, 0, 1)
            open_maps = open_maps.detach()

            # for backtracking
            idx = idx.reshape(self.num_samples, -1)
            snm = node_selection.reshape(self.num_samples, -1)
            new_parents = snm.max(-1, keepdim=True)[1]
            parents = new_parents * idx + parents * (1 - idx)
            if torch.all(is_unsolved.flatten()==0):
                break
        
        
        path_maps, path_list = self.backtrack(start_maps, goal_maps, parents, t)
        if self.store_intermediate_results:
            intermediate_results.append(
                {
                    "histories": histories.unsqueeze(1).detach(),
                    "paths": path_maps.unsqueeze(1).detach(),
                }
            )
            # if t == Tmax - 1:
            #     print("Fail to find paths!!!")
            #     return -1
    

        return AstarOutput(
            histories.unsqueeze(1),
            path_maps.unsqueeze(1),
            intermediate_results,
            path_list
        )

    def backtrack(self,
        start_maps: torch.tensor,
        goal_maps: torch.tensor,
        parents: torch.tensor,
        current_t: int,
    ) -> torch.tensor:
        """
        Backtrack the search results to obtain paths

        Args:
            start_maps (torch.tensor): one-hot matrices for start locations
            goal_maps (torch.tensor): one-hot matrices for goal locations
            parents (torch.tensor): parent nodes
            current_t (int): current time step

        Returns:
            torch.tensor: solution paths
        """
        path_list = []
        num_samples = start_maps.shape[0]
        parents = parents.type(torch.long)
        goal_maps = goal_maps.type(torch.long)
        start_maps = start_maps.type(torch.long)
        path_maps = goal_maps.type(torch.long)
        num_samples = len(parents)
        loc = (parents * goal_maps.view(num_samples, -1)).sum(-1)
        map_shape = start_maps.shape

        if self.output_path_list:
            row = loc//map_shape[-1]
            col = loc%map_shape[-1]
            path_list.append([row, col])
        # if self.output_path_list:
        for _ in range(current_t):
            path_maps.view(num_samples, -1)[range(num_samples), loc] = 1
            loc = parents[range(num_samples), loc]
            if self.output_path_list:
                row = loc//map_shape[-1]
                col = loc%map_shape[-1]
                path_list.append([row, col])

        return path_maps, path_list
