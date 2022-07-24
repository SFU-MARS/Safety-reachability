# import numpy as np
# import sys
# np.set_printoptions(threshold=sys.maxsize)
# data=np.load("/local-scratch/tara/project/WayPtNav-reachability/reachability/data_tmp/avoid_map_4d/v1/ttr_avoid_map_4d_whole_area3.npy")
# obstacle_2d=np.load("/local-scratch/tara/project/WayPtNav-reachability/obstacle_grid_2d.npy")
# obstacle_2dT=obstacle_2d.T
# obstacle_4dT = np.empty((9,31) + obstacle_2dT.shape,dtype=obstacle_2d.dtype)
# obstacle_4dT [:] = obstacle_2dT
# obstacle_4d=obstacle_4dT.T
# np.save('obstacle_grid_4d_ver2.npy', obstacle_4d)
# # TTC=data[140:200,340:400,28,8]
# # TTC=data[7,20,2,4]
# #TTC=data[180:210,330:360,25,6]
# TTC=data[420:440,305:325,30,6].T
# # TTC=data[:,:,30,8].T
# print(TTC)
# #print(np.sort(TTC))
# TTCm=np.mean(np.mean(TTC))
# print(TTCm)
# import matplotlib
# import matplotlib.pyplot as plt
#
# import random
#
# # Creating dataset
# plt.rcParams["axes.grid"] = False
# plt.imshow(TTC, interpolation='none')
# plt.show()

import numpy as np
data=np.load("/local-scratch/tara/project/optimized_dp/TTR_grid_4d_20_limit.npy")
TTC=data[:,:,30,8].T
print(TTC)
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = False
plt.imshow(TTC, interpolation='none')
plt.show()

import math
import numpy as np
# reso=2
#for spatial
# grid_x, grid_y, grid_v, grid_theta = np.mgrid[0:30:reso*600j, 0:26.05:reso*521j,-1:3:31j ,-math.pi:math.pi:9j ]
points=np.mgrid[0:30:600j, 0:26.05:521j,-1:3:31j ,-math.pi:math.pi:9j]
values=np.load("/local-scratch/tara/project/WayPtNav-reachability/obstacle_grid_4d.npy")
np.newaxis=4
image = values[np.newaxis,...]
from scipy.interpolate import griddata
grid_z0 = griddata(points, values, (grid_x, grid_y, grid_v, grid_theta), method='nearest')
TTC2=grid_z0[:,:,5,8]
plt.imshow(values, interpolation='none')
plt.show()
plt.title('Original')
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = False
plt.imshow(grid_z0, interpolation='none')
plt.show()