# Implementation of Collision Avoidance Through Neural Implicit Probabilistic Scenes 

This repo contains the implementation of the CATNIPS paper. 

## NeRF Model:
For this particular implementation, the Stonehenge NeRF model was used. 
<img width="582" alt="nerf" src="https://github.com/Samorange1/CATNIPS/assets/71136335/0cf3fa35-3508-4445-9c5f-23323b51eb46">


Results:

## Cell Intensity Grid:

<img width="440" alt="cellintensity" src="https://github.com/Samorange1/CATNIPS/assets/71136335/79dea92a-c6ea-4d81-9981-95c88ac573cd">



## Probabilistic Unsafe Robot Region:
Voxelized representation:

<img width="307" alt="purr (1)" src="https://github.com/Samorange1/CATNIPS/assets/71136335/19ecc0df-bef0-4494-acce-632c557b9e06">

Mesh Representation of PURR:
This is a rough mesh created by python marching cubes library. The jagged mesh is a result of the occupancy grid having 1-0 values for smoother meshes the library requires a smooth decay function.
![purr](https://github.com/Samorange1/CATNIPS/assets/71136335/d16a61ec-a693-48be-a74d-0003bd01bca6)


## Bounding Box Generation Example:
Notice how the bounding box occupies the free space in the map
<img width="693" alt="bounding box" src="https://github.com/Samorange1/CATNIPS/assets/71136335/2ed9f854-f1cd-4e46-abb8-9e8cf2ee90ab">

## Path Planning:
Astar path is in orange
The final optimized trajectory is in red.
<img width="677" alt="bezier_geometry" src="https://github.com/Samorange1/CATNIPS/assets/71136335/7f104138-2f3e-4d24-9811-02067cff3c10">


In mesh representation:
![astar with bezier2](https://github.com/Samorange1/CATNIPS/assets/71136335/05c319ca-af70-40a9-9575-6a59ce2deb4c)



