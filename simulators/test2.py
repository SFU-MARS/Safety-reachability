from obstacles.obstacle_map import ObstacleMap
import numpy as np
import tensorflow as tf
import os
from sbpd.sbpd_renderer import SBPDRenderer
from utils.fmm_map import FmmMap
from systems.dubins_car import DubinsCar
import tensorflow.contrib.eager as tfe
import matplotlib.pyplot as plt

# from Simulator import reset
# from Simulator import _iterate
# from trajectory import SystemConfig
##
import tensorflow as tf
import numpy as np
from objectives.objective_function import ObjectiveFunction
from objectives.angle_distance import AngleDistance
from objectives.goal_distance import GoalDistance
from objectives.obstacle_avoidance import ObstacleAvoidance
from objectives.reach_avoid_4d import ReachAvoid4d
from objectives.reach_avoid_3d import ReachAvoid3d
from objectives.avoid_4d import Avoid4d
from trajectory.trajectory import SystemConfig, Trajectory
from simulators.simulator_helper import SimulatorHelper
from utils.fmm_map import FmmMap
from reachability.reachability_map import ReachabilityMap
import matplotlib
import copy

import tensorflow as tf
import numpy as np
from objectives.objective_function import ObjectiveFunction
from objectives.angle_distance import AngleDistance
from objectives.goal_distance import GoalDistance
from objectives.obstacle_avoidance import ObstacleAvoidance
from objectives.reach_avoid_4d import ReachAvoid4d
from objectives.reach_avoid_3d import ReachAvoid3d
from objectives.avoid_4d import Avoid4d
from trajectory.trajectory import SystemConfig, Trajectory
from simulators.simulator_helper import SimulatorHelper
from utils.fmm_map import FmmMap
from reachability.reachability_map import ReachabilityMap
import matplotlib
import copy

from scipy.stats import beta
from scipy.spatial import KDTree


import tensorflow as tf
import numpy as np
import argparse
import importlib
import pickle
import os
import time

from utils import utils, log_utils


import scipy

from sbpd import sbpd_renderer
from sbpd.sbpd_renderer import SBPDRenderer