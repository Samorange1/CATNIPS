import torch
import numpy as np
import matplotlib.pyplot as plt
import mcubes
from mpl_toolkits.mplot3d import Axes3D
from utils import cfg
from nerf import load__nerf_model


class ProbUnsafeRobotRegion:
    def __init__(self, cfg, model): #TODO: add NERF model as input arg
        
        self.world_size = cfg.world_size
        self.scale = cfg.scale
        self.robot_size = cfg.robot_bounding_sphere_diameter
        self.purr_threshold = cfg.purr_threshold
        self.grid_size = [int(x / self.scale) for x in self.world_size]
        self.grid = torch.zeros(self.grid_size)
        self.model = model
        kernel_size = int(self.robot_size/self.scale)
        self.convolve = torch.nn.Conv3d(1,1,kernel_size=kernel_size, padding = kernel_size//2,  bias = False)
        self.cell_intensity_grid = None
        self.robot_kernel = None
        self.purr = None

    def get_local_coordinates(self):
        x,y,z = torch.meshgrid(torch.arange(self.grid_size[0]), torch.arange(self.grid_size[1]), torch.arange(self.grid_size[2]), indexing = 'ij')

        vertices_x = (x * self.scale)
        vertices_x_1 = (x * self.scale + self.scale)
        vertices_y = (y * self.scale)
        vertices_y_1 = (y *self.scale + self.scale)
        vertices_z = (z * self.scale)
        vertices_z_1 = (z * self.scale + self.scale)

        vertices_tensor = torch.stack(( torch.stack((vertices_x, vertices_y, vertices_z),dim=-1),
                               torch.stack((vertices_x_1, vertices_y, vertices_z),dim=-1),
                               torch.stack((vertices_x, vertices_y_1, vertices_z),dim=-1),
                               torch.stack((vertices_x, vertices_y, vertices_z_1),dim=-1),
                               torch.stack((vertices_x_1, vertices_y_1, vertices_z),dim=-1),
                               torch.stack((vertices_x_1, vertices_y, vertices_z_1),dim=-1),
                               torch.stack((vertices_x, vertices_y_1, vertices_z_1),dim=-1),
                               torch.stack((vertices_x_1, vertices_y_1, vertices_z_1),dim=-1) ), dim=-2)
    
        return vertices_tensor #(GS x GS x GS x 8 x3)
    
    def compute_density(self,world_coordinates):
        world_coords = world_coordinates.clone().reshape(-1,3)
        x_max = torch.max(world_coords[:,0])
        y_max = torch.max(world_coords[:,1])
        z_max = torch.max(world_coords[:,2])

        nerfed_coords = torch.zeros_like(world_coords)
        nerfed_coords[:,0] = ((world_coords[:,0] * (self.model.bound * 2))/x_max) - self.model.bound
        nerfed_coords[:,1] = ((world_coords[:,1] * (self.model.bound * 2))/y_max) - self.model.bound
        nerfed_coords[:,2] = ((world_coords[:,2] * (self.model.bound * 2))/z_max) - self.model.bound

        densities = self.model.density(nerfed_coords.to('cuda'))

        return densities['sigma'].reshape(self.grid_size[0],self.grid_size[1],self.grid_size[2],8,1)
    
    def get_coefficients(self,vertices_tensor,densities):
        data = vertices_tensor.clone()
        data = data.reshape(-1,3)
        A = torch.empty((data.shape[0], 8))
        A[:,0] = torch.ones((A.shape[0]))
        A[:,2] = data[:,1]
        A[:,1] = data[:,0]
        A[:,3] = data[:,2]
        A[:,4] = (data[:,0]*data[:,1])
        A[:,5] = (data[:,1]*data[:,2])
        A[:,6] = (data[:,0]*data[:,2])
        A[:,7] = (data[:,0]*data[:,1]*data[:,2])
        A = A.reshape(self.grid_size[0],self.grid_size[1],self.grid_size[2],8,8)
        solution = torch.linalg.lstsq(A.to('cuda'),densities)
        return solution.solution
    

    def get_analytical_solution(self, vertices_tensor, coeffs):
        b = torch.max(vertices_tensor,dim=3, keepdims = True).values
        a = torch.min(vertices_tensor,dim=3, keepdims = True).values
        coeffs = torch.transpose(coeffs,3,4)
        a = a.reshape(-1,3)
        b = b.reshape(-1,3)
        coeffs = coeffs.reshape(-1,8)
        x_d  = b[:,0] - a[:,0]
        y_d = b[:,1] - a[:,1]
        z_d = b[:,2] - a[:,2]
        x_2_d = b[:,0]**2 - a[:,0]**2
        y_2_d = b[:,1]**2 - a[:,1]**2
        z_2_d = b[:,2]**2 - a[:,2]**2
        analytical_soln = (x_d * y_d * z_d) + \
                    (coeffs[:,1] * y_d * z_d * x_2_d) + \
                    (coeffs[:,2] * x_d * z_d * y_2_d) + \
                    (coeffs[:,3] * x_d * y_d * z_2_d) + \
                    (coeffs[:,4] * z_d * x_2_d * y_2_d) + \
                    (coeffs[:,5] * x_d * y_2_d * z_2_d) + \
                    (coeffs[:,6] * y_d * x_2_d * z_2_d) + \
                    (coeffs[:,7] * x_2_d * y_2_d * z_2_d)
        analytical_soln = analytical_soln.reshape(self.grid_size[0],self.grid_size[1],self.grid_size[2],1)
        return analytical_soln


    def get_cell_intensity(self):
        vertices_tensor = self.get_local_coordinates()
        densities = self.compute_density(vertices_tensor)
        coefficients = self.get_coefficients(vertices_tensor, densities)
        self.cell_intensity_grid = self.get_analytical_solution(vertices_tensor, coefficients)
        self.cell_intensity_grid = self.cell_intensity_grid.squeeze()

    
    def get_robot_kernel(self):
        # Taking an arbitrary bounding sphere
        grid_size = self.robot_size/self.scale   #voxels
        sphere_radius = self.robot_size/2
        x, y, z = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), torch.arange(grid_size))

        # Calculate the distance from each voxel to the sphere center
        distance = torch.sqrt((x - grid_size // 2) ** 2 + (y - grid_size // 2) ** 2 + (z - grid_size // 2) ** 2)

        # Create a mask for voxels inside the sphere (distance <= sphere_radius)
        robot_kernel = (distance <= sphere_radius).float()
        robot_kernel = torch.nn.Parameter(robot_kernel.unsqueeze(dim=0).unsqueeze(dim=1), requires_grad=False)
        self.convolve.weight = robot_kernel

        

         
    def get_purr(self):
        self.get_cell_intensity()
        self.get_robot_kernel()
        self.purr = self.convolve(self.cell_intensity_grid.unsqueeze(dim=0).unsqueeze(dim=0))
        self.purr = self.purr.squeeze()

        # Threshold:
        self.purr = (self.purr<=self.purr_threshold).float()

     

        
def visualize(grid):
    occupied_voxels = grid>=10000
    occupied_voxels = occupied_voxels.squeeze().detach().cpu().numpy()
    x, y, z = np.where(occupied_voxels)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print("Generating plot")
    # Plot solid cubes for occupied voxels
    for i in range(len(x)):
        ax.bar3d(x[i], y[i], z[i], dx=1, dy=1, dz=1, shade=True)
    # ax.scatter(x, y, z, color='b', marker='o', label='Occupied Voxels')
    print("Plot gen Finished")
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Occupied Voxels in 3D Grid')
    plt.show()


def convert_to_mesh(grid):
    voxel_grid = grid.cpu().detach().numpy()
    vert, faces = mcubes.marching_cubes(mcubes.smooth(voxel_grid),0)
    print("Vertices: ", vert.shape)
    mcubes.export_obj(vert, faces, 'map.obj')


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
    else:
        device = 'cpu'

    torch.set_default_device(device)
    model = load__nerf_model()
    print("Model Loaded Successfully")
    purr = ProbUnsafeRobotRegion(cfg, model)
    print("Initialized PURR class")
    purr.get_purr()
    print("Calcuated PURR")
    print("PURR shape:", purr.cell_intensity_grid.shape)
    print("Max_value: ", torch.max(purr.purr))
    convert_to_mesh(purr.purr)


    



        