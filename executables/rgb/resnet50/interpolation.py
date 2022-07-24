import math
import numpy as np
reso=2
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
#for spatial

# grid_x, grid_y, grid_v, grid_theta = np.mgrid[0:30:reso*600j, 0:26.05:reso*521j,-1:3:31j ,-math.pi:-math.pi:9j ]
# points=(np.mgrid[0:30:600j, 0:26.05:521j,-1:3:31j ,-math.pi:-math.pi:9j],
# obstaclemap_4D=np.load("/local-scratch/tara/project/WayPtNav-reachability/obstacle_grid_4d.npy")
# obstaclemap_2D=obstaclemap_4D[:,:,30,0]
# from scipy.interpolate import griddata
# grid_z0 = griddata(points, values, (grid_x, grid_y, grid_v, grid_theta), method='nearest')
data=np.load("/home/ttoufigh/optimized_dp/TTR_grid_4d_20_test3.npy")
TTC=data[:,:,9,:]
TTCm=np.max(np.max(TTC))
print(TTCm)
# import matplotlib
# n_cols = ['-180', '-135', '-90', '-45', '0', '45', '90', '135', '180']
fig, ax = plt.subplots(3, 3)
pos0=ax[0,0].matshow(TTC[:,:,1], label='-135')
# p0, = plt.plot([1, 2, 3], label='-135')
# plt.legend(handles=[p0], bbox_to_anchor=(1.05, 1), loc='upper left')
fig.colorbar(pos0, ax=ax[0,0])
pos01=ax[0,1].matshow(TTC[:,:,0],label='180')
fig.colorbar(pos01, ax=ax[0,1])
pos02=ax[0,2].matshow(TTC[:,:,7], label='135')
fig.colorbar(pos02, ax=ax[0,2])
ax[1,0].matshow(TTC[:,:,2], label='-90')
fig.colorbar(pos02, ax=ax[1,0])
# ax[1,1].matshow(obstaclemap_2D, label='obstacle map')
ax[1,2].matshow(TTC[:,:,6], label='90')
fig.colorbar(pos02, ax=ax[1,2])
ax[2,0].matshow(TTC[:,:,3], label='-45')
fig.colorbar(pos02, ax=ax[2,0])
ax[2,1].matshow(TTC[:,:,4], label='0')
fig.colorbar(pos02, ax=ax[2,1])
ax[2,2].matshow(TTC[:,:,5], label='45')
fig.colorbar(pos02, ax=ax[2,2])
plt.show()
# fig.tight_layout()
for i in range(3):
    for j in range(3):
        col_name = i*3+j
        ax[i,j].matshow(TTC[:,:,col_name])
        plt.show()
fig.tight_layout()
# plt.show()

import matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = False
plt.matshow(TTC)
plt.show()
x = np.linspace(0, 30, 600)
y = np.linspace(0, 26.05, 521)
v = np.linspace(-1, 3, 31)
heading = np.linspace(-math.pi,math.pi,9)*180/(math.pi)
data=np.load("/local-scratch/tara/project/WayPtNav-reachability/obstacle_grid_4d.npy")
my_interpolating_function = RegularGridInterpolator((x, y, v,heading), data)
pts = np.mgrid[0:30:reso*600j, 0:26.05:reso*521j,-1:3:31j ,-math.pi:math.pi:9j ]
(x1,y1,v1,h1)=my_interpolating_function(pts)