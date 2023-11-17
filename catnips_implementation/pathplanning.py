import torch
import numpy as np
import matplotlib.pyplot as plt
import dijkstra3d
import cc3d
from PURR import ProbUnsafeRobotRegion, visualize
from utils import cfg
from nerf import load__nerf_model


class CATNIPS:
    def __init__(self):
        model = load__nerf_model()
        self.purr = ProbUnsafeRobotRegion(cfg, model)
        self.start = np.asarray(cfg.start)
        self.end = np.asarray(cfg.end)
    
    def get_a_star_path(self):
        field = self.purr.purr.squeeze().detach().cpu().numpy()
        field = field.astype(np.int64)
        field = field * 999
        field = field + 1

        
        print("max value now: ", np.max(field))
        print("min value now: ", np.min(field))

        connectivity_graph = cc3d.voxel_connectivity_graph(field,
                                                       connectivity = 6)
        
        print("graph shape: ", connectivity_graph.shape)
        self.astar_path = dijkstra3d.dijkstra(field, 
                                         self.start, 
                                         self.end,
                                         connectivity = 6,
                                         compass = True)#,
                                         #voxel_graph = connectivity_graph)
        
    
    def split_path_to_segments(self):
        lines = []
        direction = None
        start = 0
        end = 0

        for i in range(1,self.astar_path.shape[0]):
            diff = self.astar_path[i,:] - self.astar_path[i-1,:]
            dir = np.nonzero(diff)
            if direction is None:
                direction = dir
                end = i
            elif dir == direction:
                end = i
            else:
                lines.append([start,end])
                start = i
                end = i
                direction = dir
        lines.append([start,end])
        return lines


            


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
    else:
        device = 'cpu'

    torch.set_default_device(device)

    cat = CATNIPS()
    cat.get_a_star_path()
    print("PATH shape: ", cat.astar_path.shape)
    lines = cat.split_path_to_segments()
    print(lines)
    # visualize(cat.purr.purr, 0, path)
