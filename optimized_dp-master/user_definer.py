import imp
import numpy as np
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *

# Specify the  file that includes dynamic systems
from DubinsCar4D_new2 import *
# Plot options
from odp.Plots import PlotOptions
# Solver core
from odp.solver import HJSolver, computeSpatDerivArray
import math
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib

""" USER INTERFACES
- Define grid

- Generate initial values for grid using shape functions

- Time length for computations

- Run
"""

g = Grid(np.array([0, 0, -math.pi, 0.0]), np.array([30, 26.05, math.pi, 0.7+0.1]),
         4, np.array([600, 521, 31, 31]), [2])
my_car = DubinsCar4D_new2()

# Load value from my map
obstacle_2d=np.load("/local-scratch/tara/project/WayPtNav-reachability/obstacle_grid_2d.npy")
# obstacles = np.load("obstacle_grid_4d_ver2.npy")
obstacles = np.tile(
    np.expand_dims(obstacle_2d, (-2, -1)),
    (1,1,31,31)
)


# Velocity constraint - negative level set [0., 0.7]
velocity_constr = Intersection(Lower_Half_Space(g, 3, 0.6), Upper_Half_Space(g, 3, 0.0))

# Combine it with the original obstacle map
Initial_value_f = Union(-velocity_constr, obstacles)

# Look-back lenght and time step
lookback_length = 10.0
t_step = 0.05

tau = np.arange(start = 0, stop = lookback_length + t_step, step = t_step)

po2 = PlotOptions(do_plot=False, plot_type="3d_plot", plotDims=[0,1,2],
                  slicesCut=[2])

compMethods = { "TargetSetMode": "minVWithV0"}

# Solve the HJ pde
result = HJSolver(my_car, g, Initial_value_f, tau, compMethods, po2, saveAllTimeSteps=False )
np.save("V_safe1.npy", result)

