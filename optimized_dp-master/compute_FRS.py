import imp
import os
import numpy as np
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *

# Specify the  file that includes dynamic systems
from DubinsCar4D_new2 import *
from odp.dynamics.DubinsCar4D import DubinsCar4D
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

g = Grid(np.array([0, -5, 0.0, -math.pi]), np.array([5, 5, 0.7, math.pi]),
         4, np.array([100, 100, 61, 36]), [3])
         # 4, np.array([600, 521, 31, 9]), [2])
# my_car = DubinsCar4D_new2()
my_car = DubinsCar4D(uMin=[-1.1, -0.4], uMax=[1.1, 0.4],
                     dMin=[0,0], dMax=[0,0], uMode="max", dMode="min")

# Look-back lenght and time step
horizon = 6
t_step = 0.05

v_init = np.linspace(0.0, 0.6, 61)
tau = np.arange(start= 0, stop= horizon + t_step, step = t_step)

compMethods = { "TargetSetMode": "minVWithVInit"}

for idx, v in enumerate(v_init):
    # Initial position always have (x, y, theta) as (0, 0, 0)
    Initial_value_f = Rect_Around_Point(g, [0.0, 0.0, v, 0.])
    po2 = PlotOptions(do_plot=False, plot_type="3d_plot", plotDims=[0,1,3],
                      slicesCut=[idx])
    # Solve the HJ pde
    result = HJSolver(my_car, g, Initial_value_f, tau, compMethods, po2, saveAllTimeSteps=False,
                      get_FRS=True)
    result = np.swapaxes(result, 2,3)
    # base_dir = os.path.join('FRS_result/FRS_v{}_H{}'.format(v, horizon))
    # if not os.path.exists(base_dir):
    #     os.makedirs(base_dir)
    dir = "FRS_result4_wdis" +"/"+"FRS_v{:.2f}_H{}".format(v, horizon)
    # utils.mkdir_if_missing(base_dir)
    np.save(dir, result)

