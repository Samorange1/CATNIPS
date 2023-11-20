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
        x = np.array([1,0,0])
        y = np.array([0,1,0])
        z = np.array([0,0,1])
        self.normals = {0:[y,z],
                        1:[x,z],
                        2:[x,y]}
        
    
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

    def get_nearest_obstacle_distance(self,voxel, direction):
        i = 0
        while i <= self.max_obstacle_distance :
            next_voxel = voxel + direction
            if self.purr.purr[next_voxel] == 1:
                if i == 0:
                    print("SPECIAL CASE")
                    i = self.purr.scale/2
                return i
            else:
                i = i+1
                voxel = next_voxel
        return i
    
    def get_bounding_box_dims(self, line):
        min_dists = 10 * self.max_obstacle_distance * np.ones((4))

        diff = self.astar_path[line[0],:] - self.astar_path[line[1],:]

        if np.sum(diff) == 0:
            # TODO: think of a solution for one voxel length lines and one voxel only solutions
            pass

        dir = np.nonzero(diff)
        normals = self.normals[dir]

        for i in range(line[0],line[1]+1):
            voxel = self.astar_path[i,:].copy()

            for idx, normal in enumerate(normals):

                dist1 = self.get_nearest_obstacle_distance(voxel.copy(), normal)
                
                if dist1 < min_dists[2*idx]:
                    min_dists[2*idx] = dist1.copy()
                
                dist2 = self.get_nearest_obstacle_distance(voxel.copy(),-1*normal)

                if dist2 < min_dists[2*idx + 1]:
                    min_dists[2*idx + 1] = dist2.copy()
        
        normals_2_stack = np.vstack((normals[1],-normals[1], normals[1], -normals[1]))
        normals_1_stack = np.vstack((normals[0], normals[0], -normals[0], -normals[0]))
        dist_1_stack = np.vstack((min_dists[0],min_dists[0],min_dists[1],min_dists[1]))
        dist_2_stack = np.vstack((min_dists[2], min_dists[3], min_dists[2], min_dists[3]))


        corners1 = self.astar_path[line[0],:] + (dist_1_stack * normals_1_stack) + (dist_2_stack * normals_2_stack)
        corners2 = self.astar_path[line[1],:] + (dist_1_stack * normals_1_stack) + (dist_2_stack * normals_2_stack)
        corners = np.vstack((corners1,corners2))

        return corners

            


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
    # print(lines)
    # visualize(cat.purr.purr, 0, cat.astar_path)
