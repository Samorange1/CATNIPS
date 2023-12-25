import torch
import numpy as np
import matplotlib.pyplot as plt
import dijkstra3d
import cc3d
from PURR import ProbUnsafeRobotRegion, visualize
from utils import cfg
from nerf import load__nerf_model
from scipy.spatial import KDTree
# from optimizer import QuadProg


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
        self.max_obstacle_distance = 10
        self.debug = True
        self.obstacles = None
        self.kd_tree = self.convert_to_kd_tree()

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
                start = i-1
                end = i
                direction = dir
        lines.append([start,end])
        return lines

    def convert_to_kd_tree(self):
        obstacles = np.where(self.purr.purr.squeeze().detach().cpu().numpy() > 0.5)
        self.obstacles = np.array([obstacles[0], obstacles[1], obstacles[2]]).T
        
        kd_tree = KDTree(self.obstacles, leafsize = 50)
        return kd_tree

    def check_for_obstacles(self,voxel, direction):
        if self.purr.purr[voxel[0], voxel[1],voxel[2]] == 1:
            return True
        else:
            _ , indices = self.kd_tree.query(voxel.reshape(1,-1), k=8, workers =3)
            # print(self.obstacles.shape)
            nearest_neighbours = self.obstacles[indices].squeeze()
            # print(nearest_neighbours.shape)
            dir = np.nonzero(direction)
            dir = int(dir[0].item())
            print(dir)
            return np.any((nearest_neighbours[dir]-voxel[dir])==0)

    def get_nearest_obstacle_distance(self,voxel, direction):
        i = 0
        while i <= self.max_obstacle_distance :
            next_voxel = voxel + direction
            if self.check_for_obstacles(next_voxel,direction):
                return i
            else:
                i = i+1
                voxel = next_voxel
        return i
    

    def get_dimensions_of_box(self, corners):
        '''
        Returns the width height and depth of the bounding box in local dimensions
        '''
        min_x, min_y, min_z = np.min(corners, axis = 0)
        max_x, max_y, max_z = np.max(corners, axis = 0)
        
        width = self.purr.scale * (max_x - min_x) + self.purr.scale

        height = self.purr.scale * (max_y - min_y) + self.purr.scale

        depth = self.purr.scale * (max_z - min_z) + self.purr.scale
    
        
        return width, height, depth


    def get_cuboid_coords(self, corner, w, h, d):
        '''
        Returns the cuboid corner coordinates in local coordinates
        '''
        corner = self.purr.scale * corner
        x,y,z = corner
        return np.array([[x,y,z],
                         [x,   y,   z+d],
                         [x,   y+h, z+d],
                         [x,   y+h, z],
                         [x+w, y,   z],
                         [x+w, y,   z+d],
                         [x+w, y+h, z],
                         [x+w, y+h, z+d]
                        ])
  

    def get_bounding_box_coords(self, line):
        '''
        Returns the local coordinates of the bounding box  containing the line segment
        '''
        min_dists = 10 * self.max_obstacle_distance * np.ones((4))

        diff = self.astar_path[line[0],:] - self.astar_path[line[1],:]

        dir = np.nonzero(diff)
        normals = self.normals[int(dir[0].item())]

        for i in range(line[0],line[1]+1):
            voxel = self.astar_path[i,:].copy()

            for idx, normal in enumerate(normals):

                dist1 = self.get_nearest_obstacle_distance(voxel.copy(), normal)
                
                if dist1 < min_dists[2*idx]:
                    min_dists[2*idx] = dist1
                
                dist2 = self.get_nearest_obstacle_distance(voxel.copy(),-1*normal)

                if dist2 < min_dists[2*idx + 1]:
                    min_dists[2*idx + 1] = dist2
        
        normals_2_stack = np.vstack((normals[1],-normals[1], normals[1], -normals[1]))
        normals_1_stack = np.vstack((normals[0], normals[0], -normals[0], -normals[0]))
        dist_1_stack = np.vstack((min_dists[0],min_dists[0],min_dists[1],min_dists[1]))
        dist_2_stack = np.vstack((min_dists[2], min_dists[3], min_dists[2], min_dists[3]))
        
        corners1 = self.astar_path[line[0],:] + (dist_1_stack * normals_1_stack) + (dist_2_stack * normals_2_stack)
       

        corners2 = self.astar_path[line[1],:] + (dist_1_stack * normals_1_stack) + (dist_2_stack * normals_2_stack)
        # if self.debug:
        #     print("D1N1: ", dist_1_stack*normals_1_stack)
        #     print("D2N2: ", dist_2_stack*normals_2_stack)
        #     print(corners1)
        #     print(corners2)
        #     self.debug =  False
        voxel_corners = np.vstack((corners1,corners2))
        # print(voxel_corners)

        min_corner_index = np.argmin(np.sqrt(np.sum(voxel_corners**2, axis = 1)))

        top_left_corner = voxel_corners[min_corner_index] 
        # print(top_left_corner)
        
        width, height, depth = self.get_dimensions_of_box(voxel_corners)

        corners = self.get_cuboid_coords(top_left_corner, width, height, depth)
        return corners
    

    def generate_bounding_box(self):
        lines = self.split_path_to_segments()
        bounding_boxes = []

        for line in lines:
            bounding_boxes.append(self.get_bounding_box_coords(line))
        return bounding_boxes
    
    def get_smooth_path(self, deg, derivative_order):
        self.get_a_star_path()
        box_coords  = self.generate_bounding_box()
        opt = QuadProg(deg=deg, derivative_order=derivative_order, start=self.astar_path[0], end = self.astar_path[-1], box_coords=box_coords)
        optimized_control_points = opt.optimize()
        
        return optimized_control_points


    





if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
    else:
        device = 'cpu'

    torch.set_default_device(device)

    cat = CATNIPS()
    cat.get_a_star_path()
    # print("PATH shape: ", cat.astar_path.shape)
    # lines = cat.split_path_to_segments()
    # # cat.generate_bounding_box()
    # print(cat.generate_bounding_box())
    # print(cat.astar_path)
    # visualize(cat.purr.purr, 0, cat.astar_path)
    print(cat.generate_bounding_box())
    # cat.get_smooth_path()
