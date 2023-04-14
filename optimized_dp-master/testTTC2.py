import math
import PIL
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import scipy.io
import matplotlib
from odp.Grid import Grid
import plotly.graph_objects as go





# Speed range = [0.0, 0.7]

g = Grid(np.array([0, 0, -math.pi, 0.]), np.array([30, 26.05, math.pi, 0.7]),
         4, np.array([600, 521, 31, 9]), [2])

TTC = np.load("ttr_avoid_map_4d_whole_area3_no_dist.npy")
print("Array shape is {}".format(TTC.shape))


speed = 0.0
angle = math.pi/2
s_idx = np.searchsorted(g.grid_points[3], speed)
a_idx = np.searchsorted(g.grid_points[2], angle)

# let's visualize a slice
slice1= TTC[:, :, a_idx, s_idx]
print("slice shape is {}".format(slice1.shape))

fig, ax = plt.subplots()
pos0 = ax.matshow(slice1)
plt.show()


