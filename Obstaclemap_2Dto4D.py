import numpy as np
data=np.load("/local-scratch/tara/project/WayPtNav-reachability/reachability/data_tmp/avoid_map_4d/v1/ttr_avoid_map_4d_whole_area3.npy")
# obstacle_2d=np.load("/local-scratch/tara/project/WayPtNav-reachability/obstacle_grid_2d.npy")
# obstacle_2dT=obstacle_2d.T
# obstacle_4dT = np.empty((9,31) + obstacle_2dT.shape,dtype=obstacle_2d.dtype)
# obstacle_4dT [:] = obstacle_2dT
# obstacle_4d=obstacle_4dT.T
# np.save('obstacle_grid_4d_ver2.npy', obstacle_4d)