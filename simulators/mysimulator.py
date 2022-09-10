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
#from reachability.reachability_map1 import ReachabilityMap
import matplotlib
import copy

import matplotlib.pyplot as plt
import control as ct
import control.flatsys as fs
import control.optimal as opt

###my lib

import tensorflow as tf
# from matplotlib.pyplot import hold

from utils import utils

tf.enable_eager_execution(**utils.tf_session_config())

##
from systems.dubins_car import DubinsCar

# from Simulator import reset
# from Simulator import _iterate
# from trajectory import SystemConfig
##

from trajectory.trajectory import SystemConfig

import tensorflow as tf
import numpy as np
import argparse
import importlib
import os
import time

from utils import utils

tf.enable_eager_execution(**utils.tf_session_config())

import scipy

from sbpd.sbpd_renderer import SBPDRenderer

from mp_env.mp_env import Building

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from trajectory.spline.spline_3rd_order import Spline3rdOrder
from trajectory.trajectory import SystemConfig
from dotmap import DotMap
from systems.dynamics import Dynamics
from systems.dubins_v4 import DubinsV4

from data_sources.visual_navigation_data_source import VisualNavigationDataSource

import os
import matplotlib.pyplot as plt
import control as ct
import control.flatsys as fs


##end of my lib

class Simulator(SimulatorHelper):

    def __init__(self, params):
        self.params = params.simulator.parse_params(params)
        self.rng = np.random.RandomState(params.seed)  # Sample some random states, used for initalizing map
        self.obstacle_map = self._init_obstacle_map(self.rng)
        self.obj_fn = self._init_obj_fn()
        self.planner = self._init_planner()
        self.system_dynamics = self._init_system_dynamics()
        self.Q=[]
        self.labels=[]


    @staticmethod
    def parse_params(p):
        """
        Parse the parameters to add some additional helpful parameters.
        """
        # Parse the dependencies
        p.planner_params.planner.parse_params(p.planner_params)
        p.obstacle_map_params.obstacle_map.parse_params(p.obstacle_map_params)

        dt = p.planner_params.control_pipeline_params.system_dynamics_params.dt

        p.episode_horizon = int(np.ceil(p.episode_horizon_s / dt))
        p.control_horizon = int(np.ceil(p.control_horizon_s / dt))
        p.dt = dt

        return p

    # TODO: Varun. Dont clip the vehicle trajectory object,
    # but store the time index of when
    # the episode ends. Use this when plotting stuff
    def simulate(self):
        """ A function that simulates an entire episode. The agent starts at self.start_config, repeatedly
        calling _iterate to generate subtrajectories. Generates a vehicle_trajectory for the episode, calculates its
        objective value, and sets the episode_type (timeout, collision, success)"""
        config = self.start_config
        vehicle_trajectory = self.vehicle_trajectory
        vehicle_data = self.planner.empty_data_dict()
        end_episode = False
        commanded_actions_nkf = []#random actions can be added
        k=20
        dV = []
        goal_states=[]

        while not end_episode:

##my simulate code


        # with tf.device('/cpu:0'):
            # simulator,p = self.get_simulator()

            #num = 10000
            num_samples = 1
            for action in range(num_samples):
                start_time = time.time()
                #simulator.reset()

                # self.acceleration_nk1 = np.random.uniform(-0.4, 0.4, 1)
                # self.angular_speed_nk1= np.random.uniform(-1.1, 1.1, 1)
                # a=0.5
                # b=0.5
                # loc=0.5
                # scale_a=0.5/0.4
                # scale_w=0.5/1.1
                # y = (x - loc) / scale ,  beta.pdf(x, a, b, loc, scale)

                # self.acceleration_nk1= beta.rvs(x, a, b, loc, scale_a)
                # self.angular_speed_nk1= beta.rvs(x, a, b, loc, scale_w)

                # self.speed_nk1 = [0,0.6]
                # num_actions=5
                # self.acceleration_nk1 = np.arange(0, 1.1, 1.1/(num_actions))# num_actions-1?
                # self.angular_speed_nk1 = np.arange(0, .7, 1.1/(num_actions))
                #
                # self.acceleration_nk1 = [0,.2,0.3]
                # self.acceleration_nk1=np.linspace(0, 0.4, 5)
                # self.angular_speed_nk1 = [0, 0,0]#,-np.pi/8, np.pi/8]
                # self.angular_speed_nk1 = np.linspace(0, 1.1, 12)
                # self.TTC=[]
                #
                # self.actions = np.array((self.acceleration_nk1, self.angular_speed_nk1)).T
                # self.actions=[0,0]


                # dt = 0.05
                # dt = 1
                # TTC0=0
                # TTC0=simulator.reachability_map.avoid_4d_map.compute_voxel_function(simulator.start_config.trainable_variables[0][0][0], simulator.start_config.trainable_variables[3][0], simulator.start_config.trainable_variables[1])
                # print(TTC0)
                # start_state_list=[]
                # start_state=np.zeros(4)

                # start_state=np.array(start_state_list)
                speed=np.float16(np.linspace(0.1,0.6,num=6))
                # speed = np.float16(np.random.uniform(0.1, 0.7, 5))
                # v0 = np.random.uniform(0, 0.7, 1)[0]
                # v0=simulator.start_config.speed_nk1()[0][0][0]
                # vf = [0.2, 0.5]

# ###
#                 start_posx_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * simulator.start_config.position_nk2()[0][0][0]
#                 start_posy_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * simulator.start_config.position_nk2()[0][0][1]
#                 start_pos_nk2 = tf.concat([start_posx_nk1, start_posy_nk1], axis=2)
#                 start_heading_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * simulator.start_config.heading_nk1()[0][0][0]
#                 # Initial SystemConfig is [0, 0, 0, v0, 0]
#                 start_speed_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * simulator.start_config.speed_nk1()
#                 Transformation=[[np.cos(simulator.start_config.heading_nk1()[0][0][0]), -np.sin(simulator.start_config.heading_nk1()[0][0][0]), simulator.start_config.position_nk2()[0][0][0]],
#                 [np.sin(simulator.start_config.heading_nk1()[0][0][0]), np.cos(simulator.start_config.heading_nk1()[0][0][0]), simulator.start_config.position_nk2()[0][0][1]],
#                 [0, 0, 1]]
#
# ####

                def create_params():
                    p = DotMap()
                    p.seed = 1
                    p.n = 1
                    p.k = 20
                    p.dt = .05

                    p.quad_coeffs = [1.0, 1.0, 1.0, 1e-10, 1e-10]
                    p.linear_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

                    p.system_dynamics_params = DotMap(system=DubinsV4,
                                                      dt=.05,
                                                      v_bounds=[0.0, .6],
                                                      w_bounds=[-1.1, 1.1],
                                                      a_bounds = [-0.4, 0.4]
                                                      )
                    p.system_dynamics_params.simulation_params = DotMap(simulation_mode='ideal',
                                                                        noise_params=DotMap(is_noisy=False,
                                                                                            noise_type='uniform',
                                                                                            noise_lb=[-0.02, -0.02, 0.],
                                                                                            noise_ub=[0.02, 0.02, 0.],
                                                                                            noise_mean=[0., 0., 0.],
                                                                                            noise_std=[0.02, 0.02, 0.]))
                    return p


                # actions_waypoint=[[0.25,0,0,0.05],[0.3,0.1,0.3,-0.05],[0.227,0.2225,0,0.2],[0.113,0.15,0,0.05],[0.15,-0.1,-0.2,0.05]]
                # actions_waypoint = [[0.25, 0, 0], [0.3, 0.1, 0.3], [0.227, 0.2225, 0],
                #                     [0.113, 0.15, 0], [0.15, -0.1, -0.2]]

                # actions_waypoints = [[.5, 0, 0], [0.35, -0.05, -0.2], [0.35, 0.05, 0.2],
                #                      [0.3, 0.1, 0.55], [0.3, -0.1, -0.55],
                #                     [0.4, 0.25, 0.7], [0.4, -0.25, -.7],[0.4, 0.2, 0.55], [0.4, -0.2, -0.55],*[0.1, 0.2, 0.15], [0.1, -0.2, -0.15],[0.25, -0.05, -0.3], [0.25, 0.05, 0.3], *[0.2, 0.2, 0.5], [0.2, -0.2, -0.5]
                # *[0.35, -0.2, -0.7],[0.35, 0.2, 0.7], [0.25, -0.2, -0.65], [0.25, 0.2, 0.65], [0.15, 0.3, 0.25], [0.15, -0.3, -0.25]
                #[0.2, 0.1, 0.75], [0.2, -0.1, -.75], [0.1, 0.1, 0.4], [0.1, -0.1, -0.4], [0.45, 0.2, 0.9], [0.45, -0.2, -0.9], [0.45, -0.1, -0.2], [0.45, 0.1, 0.2]
                actions_waypoints_0 =[[.5, 0, 0], [0.35, -0.05, -0.2], [0.35, 0.05, 0.2],
                                     [0.3, 0.1, 0.55], [0.3, -0.1, -0.55],
                                    [0.4, 0.25, 0.7], [0.4, -0.25, -.7],[0.4, 0.2, 0.55], [0.4, -0.2, -0.55],[0.1, 0.2, 0.15], [0.1, -0.2, -0.15],[0.25, -0.05, -0.3], [0.25, 0.05, 0.3], [0.2, 0.2, 0.5], [0.2, -0.2, -0.5],
                [0.35, -0.2, -0.7],[0.35, 0.2, 0.7], [0.25, -0.2, -0.65], [0.25, 0.2, 0.65], [0.15, 0.3, 0.25], [0.15, -0.3, -0.25],
                [0.2, 0.1, 0.75], [0.2, -0.1, -.75], [0.1, 0.1, 0.4], [0.1, -0.1, -0.4], [0.45, 0.2, 0.9], [0.45, -0.2, -0.9], [0.45, -0.1, -0.2], [0.45, 0.1, 0.2]]

                actions_waypoints_1 =[[.5, 0, 0], [0.35, -0.05, -0.2], [0.35, 0.05, 0.2],
                                     [0.3, 0.1, 0.55], [0.3, -0.1, -0.55],
                                    [0.4, 0.1, 0.7], [0.4, -0.1, -.7],[0.4, 0.2, 0.55], [0.4, -0.2, -0.55],[0.1, 0.2, 0.95], [0.1, -0.2, -0.95],[0.25, -0.05, -0.3], [0.25, 0.05, 0.3], [0.2, 0.35, 0.9], [0.2, -0.35, -0.9],
                [0.35, -0.1, -0.6],[0.35, 0.1, 0.6], [0.25, -0.35, -0.85], [0.25, 0.35, 0.85], [0.15, 0.25, 0.65], [0.15, -0.25, -0.65],
                [0.2, 0.1, 0.25], [0.2, -0.1, -.25], [0.1, 0.1, 0.6], [0.1, -0.1, -0.6], [0.45, 0.2, 0.9], [0.45, -0.2, -0.9], [0.45, -0.1, -0.2], [0.45, 0.1, 0.2]]

                actions_waypoints_unfilter =[[.5, 0, 0.1],[.5, 0, -0.1],
                                    [0.35, -0.05, -0.2], [0.35, 0.05, 0.2],
                                     # [0.3, 0.1, 0.55], [0.3, -0.1, -0.55],
                                    [0.4, 0.1, 0.7], [0.4, -0.1, -.7],
                                    # [0.4, 0.2, 0.55], [0.4, -0.2, -0.55],
                                    # [0.1, 0.2, 0.95], [0.1, -0.2, -0.95],
                                    [0.25, -0.05, -0.3], [0.25, 0.05, 0.3],
                                    # [0.2, 0.35, 0.9], [0.2, -0.35, -0.9],
                [0.35, -0.1, -0.6],[0.35, 0.1, 0.6],
                                    # [0.25, -0.35, -0.85], [0.25, 0.35, 0.85],
                                    # [0.15, 0.25, 0.65], [0.15, -0.25, -0.65],
                # [0.2, 0.1, 0.25], [0.2, -0.1, -.25],
                #                     [0.1, 0.1, 0.6], [0.1, -0.1, -0.6],

                #                     [0.45, 0.2, 0.9], [0.45, -0.2, -0.9],
                                    [0.45, -0.1, -0.2], [0.45, 0.1, 0.2],
                                    [0.1, 0.05, .8],[0.1, -0.05, -.8],
                                    [0.15, -0.1, -0.7],[0.15, 0.1, 0.7],
                                    [0.2, 0.1, 0.55],[0.2, -0.1, -0.55],
                                    [.55, 0.1, 0.05], [.55, -0.1, -0.05],
                                    [.6, 0.15, 0.15],[.6, -0.15, -0.15],
                                    [.65, 0.1, .4], [.65, -0.1, -.4],
                                    [.7, 0, 0]]

                actions_waypoints =[[0.1, 0.05, .8],[0.1, -0.05, -.8],
                                    [0.15, -0.1, -0.7],[0.15, 0.1, 0.7],[0.2, 0.1, 0.55],[0.2, -0.1, -0.55],
                                             [0.25, -0.05, -0.3], [0.25, 0.05, 0.3],
                                             [0.35, -0.1, -0.6],[0.35, 0.1, 0.6],[0.35, -0.05, -0.2], [0.35, 0.05, 0.2],
                                             [0.4, 0.1, 0.7], [0.4, -0.1, -.7],[0.45, -0.1, -0.2], [0.45, 0.1, 0.2],[.5, 0, 0.1],[.5, 0, -0.1],
                                             [.55, 0.1, 0.05], [.55, -0.1, -0.05]]

                # actions_waypoints = [[.25, 0, 0],[0.3, 0.1, 0.3],[0.227, 0.2225, 0], [0.113, 0.15, 0],[0.15, -0.1, -0.2], [0.45, -0.05, 0.1],[0.4, -0.25, -0.1]]#based on email 4.12
                actions_waypoint = [[.7, 0, 0.05]]
                # actions_waypoint =[0.227,0.2225,0,0.2]


                # actions_waypoint =[.5, 0, 0]#,[0.25, -0.5, -1.1], [0.25, 0.5, 1.1]
                x=[]
                x1 = np.zeros((1,4,100))
                # x1 = np.zeros(( 100, 4,len(actions_waypoint)))
                state_traj = []
                # x0 = [0., 0,0, 0.25]
                # x0 = [0., 0, 0, 0.35]
                u0 = [0, 0.]
                # xf = [0.2, 0.12, 1,0.25]
                # xf = [0.4, 0.2, 0.2, 0.4]

                # x0 = start_state


                # xf = start_state+actions_waypoint
                # start_config=x0
                # goal_config=xf

###
                # simulator.reset()
                # start_state = np.array(
                #     [simulator.start_config.position_nk2()[0][0][0], simulator.start_config.position_nk2()[0][0][1],
                #      simulator.start_config.heading_nk1()[0][0][0], simulator.start_config.speed_nk1()[0][0][0]])
                # start_posx_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * simulator.start_config.position_nk2()[0][0][0]
                # start_posy_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * simulator.start_config.position_nk2()[0][0][1]
                # start_pos_nk2 = tf.concat([start_posx_nk1, start_posy_nk1], axis=2)
                # start_heading_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * simulator.start_config.heading_nk1()[0][0][0]
                # # Initial SystemConfig is [0, 0, 0, v0, 0]
                # start_speed_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * simulator.start_config.speed_nk1()
                # Transformation=[[np.cos(simulator.start_config.heading_nk1()[0][0][0]), -np.sin(simulator.start_config.heading_nk1()[0][0][0]), simulator.start_config.position_nk2()[0][0][0]],
                # [np.sin(simulator.start_config.heading_nk1()[0][0][0]), np.cos(simulator.start_config.heading_nk1()[0][0][0]), simulator.start_config.position_nk2()[0][0][1]],
                # [0, 0, 1]]
                # goal_states=[]
                # for actions_waypoint in actions_waypoints:
                #     target_state=np.array(Transformation).dot([actions_waypoint[0],actions_waypoint[1],1])
                #     goal_heading_nk1 = actions_waypoint[2] + simulator.start_config.heading_nk1()[0][0][0]
                #     goal_state=np.array([target_state[0], target_state[1],goal_heading_nk1,vf[1]])
                #     goal_states.append(goal_state)
                # goal_posx_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * target_state[0]
                # goal_posy_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * target_state[1]
                # goal_pos_nk2 = tf.concat([goal_posx_nk1, goal_posy_nk1], axis=2)
                # # goal_heading_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * target_state[2]
                #
                # goal_speed_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * vf[0]

###

                def get_simulator(self):

                    parser = argparse.ArgumentParser(description='Process the command line inputs')
                    parser.add_argument("-p", "--params", required=True, help='the path to the parameter file')
                    parser.add_argument("-d", "--device", type=int, default=1,
                                        help='the device to run the training/test on')
                    args = parser.parse_args()

                    p = self.create_params(args.params)

                    p.simulator_params = p.data_creation.simulator_params  # ! change param to use simulator_2 and different branch in github
                    p.simulator_params.simulator.parse_params(p.simulator_params)

                    simulator = p.simulator_params.simulator(p.simulator_params)

                    return simulator, p


                # Function to take states, inputs and return the flat flag
                def vehicle_flat_forward(x, u):
                    # Get the parameter values

                    # Create a list of arrays to store the flat output and its derivatives
                    zflag = [np.zeros(3), np.zeros(3)]

                    # Flat output is the x, y position of the rear wheels

                    ##https: // github.com / python - control / python - control / labels / enhancement
                    # assert x[0].shape == zflag[0][0].shape
                    zflag[0][0] = x[0]
                    zflag[1][0] = x[1]
                    theta = x[2]
                    vel = x[3]
                    # zflag[3][0] = x[3]

                    # First derivatives of the flat output
                    zflag[0][1] = vel * np.cos(theta)  # dx/dt
                    zflag[1][1] = vel * np.sin(theta)  # dy/dt
                    # zflag[2][1] = u[0]
                    # zflag[3][1] = u[1]
                    # First derivative of the angle
                    thdot = u[0]
                    vdot = u[1]

                    # Second derivatives of the flat output (setting vdot = 0)
                    zflag[0][2] = - vel * thdot * np.sin(theta) + vdot * np.cos(theta)
                    zflag[1][2] = vel * thdot * np.cos(theta) + vdot * np.sin(theta)

                    return zflag

                # Function to take the flat flag and return states, inputs
                def vehicle_flat_reverse(zflag):
                    # Get the parameter values

                    # Create a vector to store the state and inputs
                    x = np.zeros(4)
                    u = np.zeros(2)

                    # Given the flat variables, solve for the state

                    x[0] = zflag[0][0]  # x position
                    x[1] = zflag[1][0]  # y position
                    x[2] = np.arctan2(zflag[1][1], zflag[0][1])  # tan(theta) = ydot/xdot
                    x[3] = np.linalg.norm([zflag[1][1], zflag[0][1]])

                    # And next solve for the inputs
                    u[0] = 1 / (1 + (zflag[0][1] / zflag[0][1]) ** 2) * (
                                (zflag[1][2] * zflag[0][1]) - (zflag[0][2] * zflag[1][1])) / (zflag[0][1] ** 2)
                    u[1] = 0.5 * (1 / x[3]) * (2 * zflag[1][2] * zflag[1][1] + 2 * zflag[0][2] * zflag[0][1])

                    return x, u

                def plot_results(t, x, u):

                    plt.figure(figsize=[9, 4.5])
                    # Plot the trajectory in xy coordinate
                    plt.subplot(1, 4, 1)
                    plt.plot(x[1], x[0])
                    plt.xlabel('y [m]')
                    plt.ylabel('x [m]')

                    # Time traces of the state and input

                    plt.subplot(2, 4, 2)
                    plt.plot(t, x[0])
                    plt.ylabel('x [m]')
                    plt.tight_layout()

                    plt.subplot(2, 4, 3)
                    plt.plot(t, x[1])
                    plt.ylabel('y [m]')
                    plt.tight_layout()

                    plt.subplot(2, 4, 4)
                    plt.plot(t, x[2])
                    plt.ylabel('theta [rad]')
                    plt.tight_layout()

                    plt.subplot(2, 4, 6)
                    plt.plot(t, x[3])
                    plt.ylabel('v [m/s]')
                    plt.tight_layout()

                    plt.subplot(2, 4, 7)
                    plt.plot(t, u[0])
                    plt.xlabel('Time t [sec]')
                    plt.ylabel('w [rad/s]')
                    plt.tight_layout()

                    # plt.axis([0, t[-1], u0[0] - 1, uf[0] + 1])

                    plt.subplot(2, 4, 8)
                    plt.plot(t, u[1]);
                    plt.xlabel('Time t [sec]')
                    plt.ylabel('a [m2/s]')
                    plt.tight_layout()
                    plt.show()

                vehicle_flat = fs.FlatSystem(forward=vehicle_flat_forward, reverse=vehicle_flat_reverse, inputs=2,
                                             states=4)
                # x0 = [0., 0,0, 0.25]
                # x0 = [0., 0, 0, 0.35]
                u0 = [0, 0.]
                # xf = [0.2, 0.12, 1,0.25]
                # xf = [0.4, 0.2, 0.2, 0.4]
                # v0 = [0.2, 0.5]
                # vf = [0.2, 0.5]
                # x0 = [0., 0, 0, v0[1]]

                # actions_waypoint = [[0.25, 0, 0], [0.3, 0.1, 0.3], [0.227, 0.2225, 0],
                #                     [0.113, 0.15, 0], [0.15, -0.1, -0.2]]

                # actions_waypoint = [[0.25, 0, 0],[.5, 0, 0],[0.25, 0.4, 0.6],[0.25, -0.4, -0.6], [0.3, 0.3, 0.9],[0.3, -0.3, -0.9], [0.4, 0.2, 0.1],[0.4, 0.2, -0.1]
                #                     ,[0.10, 0.15, 0.4], [0.10, -0.15, -0.4] ,[0.15, 0.1, 0.2],  [0.15, -0.1, -0.2], [0.2,0.4,1],[0.2,-0.4,-1] ]
                # actions_waypoint = [[.5, 0, 0], [0.2, -0.4, -1], [0.25, 0.4, 1], [0.3, 0.25, 0.7], [0.3, -0.25, -0.7],
                #
                #                     [0.25, 0.3, 0.9], [0.25, -0.3, -.9], [0.4, 0.2, 0.4], [0.4, -0.2, -0.4],
                #                     [0.1, 0.4, 1.1], [0.1, -0.4, -1.1],
                #
                #                     [0.4, 0.1, 0.3], [0.4, -0.1, -0.3], [0.45, 0.05, 0.1], [0.35, -0.2, -.5],
                #                     [0.35, 0.2, .5], [0.45, -0.05, -0.1], [0.45, -0.1, -0.2], [0.45, 0.1, 0.2],
                #                     [0.1, 0.2, 1], [0.1, -0.2, -1], [0.15, -0.15, -0.8], [0.15, 0.15, 0.8],
                #                     [0.2, 0.35, 1], [0.2, -0.35, -1], [0.35, -0.1, -0.1], [0.35, 0.1, 0.3]]

                # xf = np.concatenate((actions_waypoint, vf[1] * np.ones((len(actions_waypoint), 1))), axis=1)
                # xf=goal_states
                uf = [0, 0]
                Tf = 1
                dt = 0.05
                t = np.linspace(0, Tf, 20)
                scores=[]
                # Define a set of basis functions to use for the trajectories
                poly = fs.PolyFamily(8)
                fig = plt.figure()

                # cost_fcn = opt.state_poly_constraint(vehicle_flat, np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]), [0,0,0,0.7])

                lb, ub = [-1.1, -.4], [1.1, 0.4]
                constraints = [opt.input_range_constraint(vehicle_flat, lb, ub)]
                out_u0=[]
                out_u1=[]
                out_x3=[]

                count = 0

                for v0 in start_speed_nk1:

                    start_state = [0, 0, 0, v0]
                    x0 = start_state
                    if v0 >= 0.1 and v0 <= 0.5:
                        speedf = [v0 - .1, v0, v0 + 0.1]
                    elif v0 < 0.10:
                        speedf = [0, v0, v0 + 0.1]
                    elif v0 > 0.5:
                        speedf = [v0 - .1, v0, 0.6]

                    for xf in actions_waypoints:

                        for vf in speedf:

                            vf1=[vf]
                            goal_state = [y for x in [xf, vf1] for y in x]


                            #  goal_states = np.concatenate((actions_waypoints, vf * np.ones((len(actions_waypoints), 1))),
                            #                          axis=1)
                        # X = []
                        # Yf = []
                        # Zf = []

                            # Find a trajectory between the initial condition and the final condition
                            #     traj = fs.point_to_point(vehicle_flat,Tf, x0, u0, xf[i], uf, basis=poly)#, constraints =[( -1.1, 1.1) ,(  -0.4, 0.4)])
                            traj_const = fs.point_to_point(vehicle_flat, t, x0, u0, goal_state, uf,basis=fs.PolyFamily(8))# constraints=constraints,

                            # ,cost=cost_fcn)
                            # Create the trajectory

                            x, u = traj_const.eval(t)
                            # plot_results(t, x, u)
                            from mpl_toolkits import mplot3d
                            # ax = plt.axes(projection ="3d")
                            # ax.scatter3D(Xf.append(xf[0]), Yf.append(xf[1]), Zf.append(xf[2]))
                            max_val_w = 1.1
                            max_val_a = 0.4
                            max_val_v = 0.6
                            if abs(u[0]).max()<=max_val_w and abs(u[1]).max()<=max_val_a and abs(x[3]).max()<=max_val_v and x[3].min()>=0 :


                                print('from',x0,'can reach to', goal_state)

                                local_point=x
                                for k in range(len(t)):
                                    target_state = np.array(Transformation).dot(
                                        [local_point[0][k], local_point[1][k], 1])
                                    goal_heading_nk1 = local_point[2][k] + \
                                                       config.heading_nk1()[0][0][0]
                                    global_point = np.array(
                                        [target_state[0], target_state[1], goal_heading_nk1, (local_point[3][k]) .astype (np.float16)])
                                    ttc.append(my_interpolating_function(global_point))

                                print("final state in world: ", global_point)
                                self.TTCmin = min(ttc)
                                print("min of TTC is: ", self.TTCmin)
                                self.Q0=self.gamma * (  #
                                        dt + self.discount * (1 - pow(self.discount, self.TTCmin + 1)) / (
                                        1 - self.discount))
                                print("Q of this action-waypoint is: ", self.Q0)


                                print("label of this action-waypoint is: ", self.label0)

                                self.Q.append(self.Q0)

                                count += 1
                                print("num samples collected: ", count)
                                print(" ")

                                # print("Q values: ", self.Q)
                                # fig = plt.figure()
                                ax = fig.gca(projection='3d')
                                ax.plot3D(x[0], x[1], x[2])
                                from matplotlib.cm import ScalarMappable
                                # cmap.set_array([])
                                # cmap = plt.cm.get_cmap('RdYlBu')
                                scales=np.linspace(speed[0], speed[5], 6)
                                norm = plt.Normalize(scales.min(), scales.max())
                                # sm = ScalarMappable(norm=norm, cmap=cmap)
                                # sm.set_array([])
                                # I=ax.scatter3D(xf[0], xf[1], xf[2],c = speedf, cmap = cmap)

                                X=np.array(speed)
                                img=ax.scatter3D(xf[0], xf[1], xf[2], c=speedf, cmap=plt.hot())

                                from matplotlib import cm
                                m = cm.ScalarMappable(cmap=plt.hot())
                                # cbar = fig.colorbar(m)
                                m.set_array(speed)
                                # plt.show()
                                # plt.colorbar(I)
                                # cbar.ax.set_title("velocity")
                                ax.set_xlabel('X Label')
                                ax.set_ylabel('Y Label')
                                ax.set_zlabel('Theta Label')
                                # elev = 90
                                # azim = -90
                                # ax.view_init(elev, azim)
                                #         # ax.text2D("v0=0.2->vf=0.2")
                                #         #plt.hold(True)
                                #
                                #

                            else:
                                out_x3 = np.clip(x[3], a_min=0, a_max=max_val_v)
                                x[3] = out_x3
                                out_u0 = np.clip(u[0], a_min = -max_val_w, a_max = max_val_w)
                                out_u1 = np.clip(u[1], a_min=np.maximum(-max_val_a*np.ones((len(x[3]),1)),-1 * x[3] / dt), a_max=np.minimum(max_val_a*np.ones((len(x[3]),1)),(max_val_v-x[3])/dt))

                                u[0]=out_u0
                                u[1]=out_u0
                                # sys = control.flatsys.LinearFlatSystem(linsys)

                                # zflag = vehicle_flat.forward(x, u)
                                # simulate_T(self, x_n1d, u_nkf, T, pad_mode='zero',
                                #            mode='ideal')

                                # zflag=vehicle_flat.forward(x0,u)
                                # print(zflag[0])

                                # dubins_car = DubinsV1(dt=dt)

                                p = create_params()
                                n, k = p.n, p.k
                                dubins = p.system_dynamics_params.system(p.dt, params=p.system_dynamics_params)
                                # dubins_car=Dynamics(dt=0.05, x_dim=4,  u_dim=2,ctrlBounds=None)

                                # trajectory_world = dubins_car.simulate_T(start_n13, u_nk2, T=k - 1)
                                # trajectory_world = dubins_car.simulate_T(x0, u, T=20-1)

                                # start_pos_n13 = tf.constant(np.array([[[0.0, 0.0, 0.0]], dtype=np.float32))
                                # speed_nk1 = np.ones((n, k - 1, 1), dtype=np.float32) * 2.0
                                # angular_speed_nk1 = np.linspace(1.5, 1.3, k - 1, dtype=np.float32)[None, :, None]
                                # u_nk2 = tf.constant(np.concatenate([speed_nk1, angular_speed_nk1], axis=2))
                                # traj_ref_egocentric = dubins.simulate_T(start_pos_n13, u_nk2, k)
                                start_time_sim = time.time()
                                start_pos_n13 = tf.constant(np.array([[[0.0, 0.0, 0.0, v0]]], dtype=np.float32))
                                acceleration = (np.ones((n, k ), dtype=np.float32) * u[1]).reshape(1,k,1)
                                angular_speed_nk1 = np.float32(u[0][None, :, None])
                                u_nk2 = tf.constant(np.concatenate([angular_speed_nk1, acceleration], axis=2))
                                u_nk2=np.float32(u_nk2)
                                # start_pos_n13 = tf.constant(np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32))
                                # speed_nk1 = np.ones((n, k - 1, 1), dtype=np.float32) * 2.0
                                # angular_speed_nk1 = np.linspace(1.5, 1.3, k - 1, dtype=np.float32)[None, :, None]
                                # u_nk2 = tf.constant(np.concatenate([speed_nk1, angular_speed_nk1], axis=2))
                                # traj_ref_egocentric = dubins.simulate_T(start_pos_n13, u_nk2, k)
                                # u_nk2 = tf.constant(u)
                                traj_ref_egocentric = dubins.simulate_T(start_pos_n13, u_nk2, k)
                                end_time_sim = time.time()
                                time_sim=end_time_sim - start_time_sim
                                # print("simulation takes: ",time_sim)
                                #goal plot(traj_ref_egocentric)


                                # traj=Dynamics.simulate_T( obj, x_n1d=x, u_nkf=u, T=20, pad_mode='zero',
                                #            mode='realistic')
                                # os.getcwd()

                                # simulator, p = self.get_simulator()



                                    # num = 10000
                                # num_samples += 1
                                data0 = VisualNavigationDataSource.reset_data_dictionary(p)
                                # simulator.reset()

                                for j in range(k):
                                    start_time = time.time()
                                    # simulator.reset()

                                    # self.acceleration_nk1 = np.random.uniform(-0.4, 0.4, 1)
                                    # self.angular_speed_nk1= np.random.uniform(-1.1, 1.1, 1)
                                    # a=0.5
                                    # b=0.5
                                    # loc=0.5
                                    # scale_a=0.5/0.4
                                    # scale_w=0.5/1.1
                                    # y = (x - loc) / scale ,  beta.pdf(x, a, b, loc, scale)

                                    # self.acceleration_nk1= beta.rvs(x, a, b, loc, scale_a)
                                    # self.angular_speed_nk1= beta.rvs(x, a, b, loc, scale_w)

                                    # self.speed_nk1 = [0,0.6]
                                    # num_actions=5
                                    # self.acceleration_nk1 = np.arange(0, 1.1, 1.1/(num_actions))# num_actions-1?
                                    # self.angular_speed_nk1 = np.arange(0, .7, 1.1/(num_actions))
                                    #
                                    # self.acceleration_nk1 = [0,.2,0.3]
                                    # self.acceleration_nk1=np.linspace(0, 0.4, 5)
                                    # self.angular_speed_nk1 = [0, 0,0]#,-np.pi/8, np.pi/8]
                                    # self.angular_speed_nk1 = np.linspace(0, 1.1, 12)
                                    # self.TTC=[]
                                    #
                                    # self.actions = np.array((self.acceleration_nk1, self.angular_speed_nk1)).T
                                    # self.actions=[0,0]

                                    # ###
                                    start_posx_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * config.position_nk2()[0][0][0]
                                    start_posy_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * config.position_nk2()[0][0][1]
                                    start_pos_nk2 = tf.concat([start_posx_nk1, start_posy_nk1], axis=2)
                                    start_heading_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * config.heading_nk1()[0][0][0]
                                    # Initial SystemConfig is [0, 0, 0, v0, 0]
                                    start_speed_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * config.speed_nk1()
                                    Transformation = [[np.cos(config.heading_nk1()[0][0][0]),
                                                       -np.sin(config.heading_nk1()[0][0][0]),
                                                       config.position_nk2()[0][0][0]],
                                                      [np.sin(config.heading_nk1()[0][0][0]),
                                                       np.cos(config.heading_nk1()[0][0][0]),
                                                       config.position_nk2()[0][0][1]],
                                                      [0, 0, 1]]

                                    # ####
                                    import math
                                    x = np.linspace(0, 30, 600)

                                    y = np.linspace(0, 26.05, 521)

                                    # v = np.linspace(0, .6, 31)
                                    v = np.linspace(-0.1, 0.7, 9)
                                    # theta = np.linspace(-math.pi, math.pi, 9)
                                    theta = np.linspace(-math.pi, math.pi, 31)

                                    xg, yg, vg, thetag = np.meshgrid(x, y, v, theta, indexing='ij', sparse=True)
                                    #
                                    # data=np.load("/home/ttoufigh/optimized_dp/TTR_grid_biggergrid_3lookback_wDisturbance_wObstalceMap_speedlimit3reverse_5.npy")
                                    data = np.load(
                                        "/local-scratch/tara/project/WayPtNav-reachability/reachability/data_tmp/avoid_map_4d/v1/ttr_avoid_map_4d_whole_area3_no_dist.npy")
                                    dataV = scipy.io.loadmat(
                                        "/local-scratch/tara/project/WayPtNav-reachability/reachability/data_tmp/avoid_map_4d/v1/dataV.mat")
                                    from scipy.interpolate import RegularGridInterpolator
                                    my_interpolating_function = RegularGridInterpolator((x, y, theta, v), data)
                                    my_interpolating_functionV = RegularGridInterpolator((x, y, theta, v), dataV['dataV'])
                                    ttc = []
                                    V = []
                                    global_pts = []
                                    # for j in range(k):  # trajectory

                                    # state_traj = [point[0][j], point[1][j], point[2][j], point[3][j]]  # j
                                    # ax.plot(start_2[0], start_2[1], 'bo', markersize=14)
                                    # pos_nk3 = spline_traj.position_and_heading_nk3()[:, j]
                                    # v_nk1 = spline_traj.speed_nk1()[:, j]
                                    local_point = traj_ref_egocentric[j]

                                    # convert to global

                                    target_state = np.array(Transformation).dot(
                                        [local_point[0][0][0], local_point[0][0][1], 1])
                                    goal_heading_nk1 = local_point[0][0][2] + \
                                                       config.heading_nk1()[0][0][0]
                                    global_point = np.array(
                                        [target_state[0], target_state[1], goal_heading_nk1,
                                         local_point[0][0][3].numpy().astype(np.float16)])

                                    ttc.append(my_interpolating_function(global_point))
                                    V.append(my_interpolating_functionV(global_point))
                                    global_pts.append(global_point)

                                    #     target_state = np.array(Transformation).dot(
                                    #         [actions_waypoint[0], actions_waypoint[1], 1])
                                    #     goal_heading_nk1 = actions_waypoint[2] + \
                                    #                        simulator.start_config.heading_nk1()[0][0][0]
                                    #     goal_state = np.array(
                                    #         [target_state[0], target_state[1], goal_heading_nk1, vf[1]])
                                    #     goal_states.append(goal_state)
                                    # goal_posx_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * target_state[0]
                                    # goal_posy_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * target_state[1]
                                    # goal_pos_nk2 = tf.concat([goal_posx_nk1, goal_posy_nk1], axis=2)
                                    # # goal_heading_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * target_state[2]
                                    #
                                    # goal_speed_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * vf[0]

                                    # camera_pos_13 = simulator.start_config.heading_nk1()[0]
                                    # camera_grid_world_pos_12 = simulator.start_config.position_nk2()[0] / dx_m
                                    # rgb_image_1mk3 = r._get_rgb_image(camera_grid_world_pos_12, camera_pos_13)
                                    # pts = np.array([position_nk1_next0, position_nk1_next1, self.speed_nk1_next, self.heading_nk1_next])
                                    # ttc.append(my_interpolating_function(pts))
                                    #
                                    # self.TTC.append(simulator.reachability_map.avoid_4d_map.compute_voxel_function(position_nk1_next,self.heading_nk1_next,self.speed_nk1_next))#update with new file
                                    # ttc.append(
                                    #     simulator.reachability_map.avoid_4d_map.compute_voxel_function([point[0][j],point[1][j]],point[3][j],point[2][j]))
                                    self.discount = 0.90
                                    self.gamma = 1.0
                                    # self.theta = 1e-10
                                # print(global_pts)
                                    print("It reaches ", global_point)
                                self.TTCmin = min(ttc)
                                self.V = min(V)
                                self.label0 = np.sign(self.V)

                                print("min of TTC is: ", self.TTCmin)

                                self.Q0 = self.gamma * (  #
                                            dt + self.discount * (1 - pow(self.discount, self.TTCmin + 1)) / (
                                            1 - self.discount))
                                print("Q of this action-waypoint is: ", self.Q0)
                                print("label of this action-waypoint is: ", self.label0)

                                self.Q.append(self.Q0)
                                self.labels.append(self.label0)

                                count += 1
                                print("num samples collected: ", count)
                                print(" ")

                                    # print ("Q values: ", self.Q)


                                r = SBPDRenderer.get_renderer_already_exists()
                                dx_cm, traversible = r.get_config()
                                dx_m = dx_cm / 100.
                                    # print(type(simulator.start_config.trainable_variables[0]))
                                    # camera_pos_13 = self.heading_nk1_next[0]
                                    # camera_grid_world_pos_12 = position_nk1_next[0] / dx_m
                                    # rgb_image_1mk3 = r._get_rgb_image(camera_grid_world_pos_12, camera_pos_13)

                                camera_pos_13 = config.heading_nk1()[0]
                                camera_grid_world_pos_12 = config.position_nk2()[0] / dx_m

                                    # image of current state
                                rgb_image_1mk3 = r._get_rgb_image(camera_grid_world_pos_12, camera_pos_13)

                                img1 = r._get_topview(camera_grid_world_pos_12, camera_pos_13)
                                    #        # In the topview the positive x axis points to the right and
                                    # the positive y axis points up. The robot is located at
                                    # (0, (crop_size[0]-1)/2) (in pixel coordinates) facing directly to the right
                                crop_size = [64, 64]
                                robot = [0, (crop_size[0] - 1) / 2]
                                import matplotlib.pyplot as plt

                                fig = plt.figure(figsize=(30, 10))
                                ax1 = fig.add_subplot(1, 3, 1)
                                ax2 = fig.add_subplot(1, 3, 2)
                                ax1.imshow(rgb_image_1mk3[0].astype(np.uint8))
                                ax2.imshow(img1[0][:, :, 0].astype(np.uint8))
                                ax2.imshow(img1[0][:, :, 0], extent=[0, 64, 0, 64])
                                # plt.show()

                                    # navDataSource = VisualNavigationDataSource(p)
                                    #
                                    # navDataSource.append_data_to_dictionary(data0, simulator)


                                    # navDataSource = VisualNavigationDataSource(p)

                                    # navDataSource.append_data_to_dictionary(self, data0)

                                d = {'image': rgb_image_1mk3, 'action': actions_waypoint, 'q values': self.Q}
                            goal_states.append(goal_state)
                        dV={'start_pose':[start_posx_nk1,start_posy_nk1, start_speed_nk1, start_heading_nk1],
                                      'image': rgb_image_1mk3, 'action': goal_states, 'labels': self.labels}


                                # dV = {'image': rgb_image_1mk3, 'action': actions_waypoint, 'labels': self.label0}
                                # save to file
                        scipy.io.savemat('myGeneratedData.mat',dV)


                                    #
                                    # import pickle
                                    # with open('file.pkl', 'wb') as f:
                                    #     pickle.dump(d, f)
                                    #
                                    # with open('fileV.pkl', 'wb') as f:
                                    #     pickle.dump(dV, f)


## end of my simulate


            trajectory_segment, next_config, data, commanded_actions_1kf = self._iterate(
                config)  # while not get to the goal, or not collide? Keep iterating

            # Append to Vehicle Data
            for key in vehicle_data.keys():
                vehicle_data[key].append(data[key])

            vehicle_trajectory.append_along_time_axis(trajectory_segment)
            commanded_actions_nkf.append(commanded_actions_1kf)
            config = next_config
            end_episode, episode_data = self._enforce_episode_termination_conditions(vehicle_trajectory,
                                                                                     vehicle_data,
                                                                                     commanded_actions_nkf)
        self.vehicle_trajectory = episode_data['vehicle_trajectory']
        self.vehicle_data = episode_data['vehicle_data']
        self.vehicle_data_last_step = episode_data['vehicle_data_last_step']
        self.last_step_data_valid = episode_data['last_step_data_valid']
        self.episode_type = episode_data['episode_type']
        self.valid_episode = episode_data['valid_episode']
        self.commanded_actions_1kf = episode_data['commanded_actions_1kf']
        self.obj_val = self._compute_objective_value(self.vehicle_trajectory)

    def _iterate(self, config):
        """ Runs the planner for one step from config to generate a
        subtrajectory, the resulting robot config after the robot executes
        the subtrajectory, and relevant planner data"""

        planner_data = self.planner.optimize(config)  # Given config (start configuration) #change objectives here
        trajectory_segment, trajectory_data, commanded_actions_nkf = self._process_planner_data(config, planner_data)
        next_config = SystemConfig.init_config_from_trajectory_time_index(trajectory_segment, t=-1)
        return trajectory_segment, next_config, trajectory_data, commanded_actions_nkf

    def _process_planner_data(self, start_config, planner_data):
        """
        Process the planners current plan. This could mean applying
        open loop control or LQR feedback control on a system.
        """

        # The 'plan' is open loop control
        if 'trajectory' not in planner_data.keys():
            trajectory, commanded_actions_nkf = self.apply_control_open_loop(start_config,
                                                                             planner_data['optimal_control_nk2'],
                                                                             T=self.params.control_horizon - 1,
                                                                             sim_mode=self.system_dynamics.simulation_params.simulation_mode)
        # The 'plan' is LQR feedback control
        else:
            # If we are using ideal system dynamics the planned trajectory
            # is already dynamically feasible. Clip it to the control horizon
            if self.system_dynamics.simulation_params.simulation_mode == 'ideal':
                trajectory = Trajectory.new_traj_clip_along_time_axis(planner_data['trajectory'],
                                                                      self.params.control_horizon,
                                                                      repeat_second_to_last_speed=True)
                _, commanded_actions_nkf = self.system_dynamics.parse_trajectory(trajectory)
            elif self.system_dynamics.simulation_params.simulation_mode == 'realistic':
                trajectory, commanded_actions_nkf = self.apply_control_closed_loop(start_config,
                                                                                   planner_data['spline_trajectory'],
                                                                                   planner_data['k_nkf1'],
                                                                                   planner_data['K_nkfd'],
                                                                                   T=self.params.control_horizon - 1,
                                                                                   sim_mode='realistic')
            else:
                assert (False)

        self.planner.clip_data_along_time_axis(planner_data, self.params.control_horizon)
        return trajectory, planner_data, commanded_actions_nkf

    def get_observation(self, config=None, pos_n3=None, **kwargs):
        """
        Return the robot's observation from configuration config or
        pos_nk3.
        """
        return [None] * config.n

    def get_observation_from_data_dict_and_model(self, data_dict, model):
        """
        Returns the robot's observation from the data inside data_dict,
        using parameters specified by the model.
        """
        raise NotImplementedError

    def get_simulator_data_numpy_repr(self):
        """
        Convert the vehicle trajectory, vehicle data,
        and vehicle data last step to numpy representations
        and return them.
        """
        vehicle_trajectory = self.vehicle_trajectory.to_numpy_repr()
        vehicle_data = self.planner.convert_planner_data_to_numpy_repr(self.vehicle_data)
        vehicle_data_last_step = self.planner.convert_planner_data_to_numpy_repr(self.vehicle_data_last_step)
        vehicle_commanded_actions_1kf = self.commanded_actions_1kf.numpy()
        return vehicle_trajectory, vehicle_data, vehicle_data_last_step, vehicle_commanded_actions_1kf

    def _enforce_episode_termination_conditions(self, vehicle_trajectory, planner_data,
                                                commanded_actions_nkf):
        p = self.params
        time_idxs = []
        for condition in p.episode_termination_reasons:
            time_idxs.append(self._compute_time_idx_for_termination_condition(vehicle_trajectory,
                                                                              condition))
        try:
            idx = np.argmin(time_idxs)
        except ValueError:
            idx = np.argmin([time_idx.numpy() for time_idx in time_idxs])

        try:
            termination_time = time_idxs[idx].numpy()
        except ValueError:
            termination_time = time_idxs[idx]

        if termination_time != np.inf:
            end_episode = True
            vehicle_trajectory.clip_along_time_axis(termination_time)
            planner_data, planner_data_last_step, last_step_data_valid = self.planner.mask_and_concat_data_along_batch_dim(
                planner_data,
                k=termination_time)
            commanded_actions_1kf = tf.concat(commanded_actions_nkf, axis=1)[:, :termination_time]

            # If all of the data was masked then
            # the episode simulated is not valid
            valid_episode = True
            if planner_data['system_config'] is None:
                valid_episode = False
            episode_data = {'vehicle_trajectory': vehicle_trajectory,
                            'vehicle_data': planner_data,
                            'vehicle_data_last_step': planner_data_last_step,
                            'last_step_data_valid': last_step_data_valid,
                            'episode_type': idx,
                            'valid_episode': valid_episode,
                            'commanded_actions_1kf': commanded_actions_1kf}
        else:
            end_episode = False
            episode_data = {}

        return end_episode, episode_data

    def reset(self, seed=-1):
        """Reset the simulator. Optionally takes a seed to reset
        the simulator's random state."""
        if seed != -1:
            self.rng.seed(seed)

        # Note: Obstacle map must be reset independently of the fmm map.
        # Sampling start and goal may depend on the updated state of the
        # obstacle map. Updating the fmm map depends on the newly sampled goal.
        reset_start = True
        while reset_start:
            self._reset_obstacle_map(self.rng)  # Do nothing here

            self._reset_start_configuration(self.rng)  # Reset self.start_config
            # Reset self.goal_config. If there is no available goals, reset_start = True, then reset the start again.
            reset_start = self._reset_goal_configuration(self.rng)

            # Manually restart the start and goal (only work for single goal)
            # reset_start = self._reset_start_goal_manually(start_pos=[9, 10], goal_pos=[9, 20])#? IndexError: index 943 is out of bounds for axis 0 with size 521
            # reset_start = self._reset_start_goal_manually(start_pos=[8.65, 50.25], goal_pos=[8.60, 47.15])
        self._update_fmm_map()  # Compute fmm_angle and fmm_goal, wrap it into voxel func

        # Initiate and update a reachability map (reach_avoid or avoid)
        if self.params.cost == 'reachability':
            self._get_reachability_map()

        # Update objective functions, may include reachability cost
        self._update_obj_fn()

        self.vehicle_trajectory = Trajectory(dt=self.params.dt, n=1, k=0)
        self.obj_val = np.inf
        self.vehicle_data = {}

        ##

    ##


    def _reset_obstacle_map(self, rng):
        raise NotImplementedError

    def _update_fmm_map(self):
        raise NotImplementedError

    def _get_reachability_map(self):
        raise NotImplementedError

    def _reset_start_goal_manually(self, start_pos, goal_pos):

        p = self.params.reset_params.start_config

        start_112 = np.array([[start_pos]], dtype=np.float32)
        goal_112 = np.array([[goal_pos]], dtype=np.float32)

        heading_111 = np.zeros((1, 1, 1))
        speed_111 = np.zeros((1, 1, 1))
        ang_speed_111 = np.zeros((1, 1, 1))

        # Initialize the start configuration
        self.start_config = SystemConfig(dt=p.dt, n=1, k=1,
                                         position_nk2=start_112,
                                         heading_nk1=heading_111,
                                         speed_nk1=speed_111,
                                         angular_speed_nk1=ang_speed_111)

        # The system dynamics may need the current starting position for
        # coordinate transforms (i.e. realistic simulation)
        self.system_dynamics.reset_start_state(self.start_config)

        # Initialize the goal configuration
        self.goal_config = SystemConfig(dt=p.dt, n=1, k=1,
                                        position_nk2=goal_112)

        return False

    def _reset_start_configuration(self, rng):
        """
        Reset the starting configuration of the vehicle.
        """
        p = self.params.reset_params.start_config

        # Reset the position
        if p.position.reset_type == 'random':
            # Select a random position on the map that is at least obstacle margin
            # away from the nearest obstacle
            obs_margin = self.params.avoid_obstacle_objective.obstacle_margin1
            dist_to_obs = 0.
            while dist_to_obs <= obs_margin:  # Change here for adversarial data collection (closer start position to
                # the obstacles)
                start_112 = self.obstacle_map.sample_point_112(rng)
                dist_to_obs = tf.squeeze(self.obstacle_map.dist_to_nearest_obs(start_112))
        elif p.position.reset_type == 'custom':
            x, y = p.position.start_pos
            start_112 = np.array([[[x, y]]], dtype=np.float32)
            dist_to_obs = tf.squeeze(self.obstacle_map.dist_to_nearest_obs(start_112))
            assert (dist_to_obs.numpy() > 0.0)
        else:
            raise NotImplementedError('Unknown reset type for the vehicle starting position.')

        # Reset the heading
        if p.heading.reset_type == 'zero':
            heading_111 = np.zeros((1, 1, 1))
        elif p.heading.reset_type == 'random':
            heading_111 = rng.uniform(p.heading.bounds[0], p.heading.bounds[1], (1, 1, 1))
        elif p.position.reset_type == 'custom':
            theta = p.heading.start_heading
            heading_111 = np.array([[[theta]]], dtype=np.float32)
        else:
            raise NotImplementedError('Unknown reset type for the vehicle starting heading.')
            return true
        # Reset the speed
        if p.speed.reset_type == 'zero':
            speed_111 = np.zeros((1, 1, 1))
        elif p.speed.reset_type == 'random':
            speed_111 = rng.uniform(p.speed.bounds[0], p.speed.bounds[1], (1, 1, 1))
        elif p.speed.reset_type == 'custom':
            speed = p.speed.start_speed
            speed_111 = np.array([[[speed]]], dtype=np.float32)
        else:
            raise NotImplementedError('Unknown reset type for the vehicle starting speed.')
            return true

        # Reset the angular speed
        if p.ang_speed.reset_type == 'zero':
            ang_speed_111 = np.zeros((1, 1, 1))
        elif p.ang_speed.reset_type == 'random':
            ang_speed_111 = rng.uniform(p.ang_speed.bounds[0], p.ang_speed.bounds[1], (1, 1, 1))
        elif p.ang_speed.reset_type == 'gaussian':
            ang_speed_111 = rng.normal(p.ang_speed.gaussian_params[0],
                                       p.ang_speed.gaussian_params[1], (1, 1, 1))
        elif p.ang_speed.reset_type == 'custom':
            ang_speed = p.ang_speed.start_ang_speed
            ang_speed_111 = np.array([[[ang_speed]]], dtype=np.float32)
        else:
            raise NotImplementedError('Unknown reset type for the vehicle starting angular speed.')



        # Initialize the start configuration
        self.start_config = SystemConfig(dt=p.dt, n=1, k=1,
                                         position_nk2=start_112,
                                         heading_nk1=heading_111,
                                         speed_nk1=speed_111,
                                         angular_speed_nk1=ang_speed_111)

        # The system dynamics may need the current starting position for
        # coordinate transforms (i.e. realistic simulation)
        self.system_dynamics.reset_start_state(self.start_config)

    def _reset_goal_configuration(self, rng):
        p = self.params.reset_params.goal_config
        goal_norm = self.params.goal_dist_norm
        goal_radius = self.params.goal_cutoff_dist
        start_112 = self.start_config.position_nk2()
        obs_margin = self.params.avoid_obstacle_objective.obstacle_margin1

        # Reset the goal position
        if p.position.reset_type == 'random':
            # Select a random position on the map that is at least obstacle margin away from the nearest obstacle, and
            # not within the goal margin of the start position.
            dist_to_obs = 0.
            dist_to_goal = 0.
            while dist_to_obs <= obs_margin or dist_to_goal <= goal_radius:
                goal_112 = self.obstacle_map.sample_point_112(rng)
                dist_to_obs = tf.squeeze(self.obstacle_map.dist_to_nearest_obs(goal_112))
                dist_to_goal = np.linalg.norm((start_112 - goal_112)[0], ord=goal_norm)
        elif p.position.reset_type == 'custom':
            x, y = p.position.goal_pos
            goal_112 = np.array([[[x, y]]], dtype=np.float32)
            dist_to_obs = tf.squeeze(self.obstacle_map.dist_to_nearest_obs(goal_112))
            assert (dist_to_obs.numpy() > 0.0)
        elif p.position.reset_type == 'random_v1':
            assert self.obstacle_map.name == 'SBPDMap'
            # Select a random position on the map that is at least obs_margin away from the
            # nearest obstacle, and not within the goal margin of the start position.
            # Additionaly the goal position must satisfy:
            # fmm_dist(start, goal) - l2_dist(start, goal) > fmm_l2_gap (goal should not be
            # fmm_dist(start, goal) < max_dist (goal should not be too far away)

            # Construct an fmm map where the 0 level set is the start position
            start_fmm_map = self._init_fmm_map(goal_pos_n2=self.start_config.position_nk2()[:, 0])
            # enforce fmm_dist(start, goal) < max_fmm_dist
            free_xy = np.where(start_fmm_map.fmm_distance_map.voxel_function_mn <
                               p.position.max_fmm_dist)
            free_xy = np.array(free_xy).T
            free_xy = free_xy[:, ::-1]
            free_xy_pts_m2 = self.obstacle_map._map_to_point(free_xy)

            # enforce dist_to_nearest_obstacle > obs_margin
            dist_to_obs = tf.squeeze(self.obstacle_map.dist_to_nearest_obs(free_xy_pts_m2[:, None])).numpy()

            dist_to_obs_valid_mask = dist_to_obs > obs_margin

            # enforce dist_to_goal > goal_radius
            fmm_dist_to_goal = np.squeeze(
                start_fmm_map.fmm_distance_map.compute_voxel_function(free_xy_pts_m2[:, None]).numpy())
            fmm_dist_to_goal_valid_mask = fmm_dist_to_goal > goal_radius

            # enforce fmm_dist - l2_dist > fmm_l2_gap
            fmm_l2_gap = rng.uniform(0.0, p.position.max_dist_diff)
            l2_dist_to_goal = np.linalg.norm((start_112 - free_xy_pts_m2[:, None]), axis=2)[:, 0]
            fmm_dist_to_goal = np.squeeze(
                start_fmm_map.fmm_distance_map.compute_voxel_function(free_xy_pts_m2[:, None]).numpy())
            fmm_l2_gap_valid_mask = fmm_dist_to_goal - l2_dist_to_goal > fmm_l2_gap

            valid_mask = np.logical_and.reduce((dist_to_obs_valid_mask,
                                                fmm_dist_to_goal_valid_mask,
                                                fmm_l2_gap_valid_mask))
            free_xy = free_xy[valid_mask]
            if len(free_xy) == 0:
                # there are no goal points within the max_fmm_dist of start
                # return True so the start is reset
                return True

            goal_112 = self.obstacle_map.sample_point_112(rng, free_xy_map_m2=free_xy)
        else:
            raise NotImplementedError('Unknown reset type for the vehicle goal position.')

        # Initialize the goal configuration
        self.goal_config = SystemConfig(dt=p.dt, n=1, k=1,
                                        position_nk2=goal_112)
        return False

    def _compute_objective_value(self, vehicle_trajectory):
        p = self.params.objective_fn_params
        if p.obj_type == 'valid_mean':
            vehicle_trajectory.update_valid_mask_nk()
        else:
            assert (p.obj_type in ['valid_mean', 'mean'])
        obj_val = tf.squeeze(self.obj_fn.evaluate_function(vehicle_trajectory))
        return obj_val

    def _update_obj_fn(self):
        """
        Update the objective function to use a new
        obstacle_map and fmm map
        """
        for objective in self.obj_fn.objectives:
            if self.params.cost == 'heuristics':
                if isinstance(objective, ObstacleAvoidance):
                    objective.obstacle_map = self.obstacle_map
                elif isinstance(objective, GoalDistance):
                    objective.fmm_map = self.fmm_map
                elif isinstance(objective, AngleDistance):
                    objective.fmm_map = self.fmm_map
            elif self.params.cost == 'reachability':
                if isinstance(objective, ReachAvoid4d):
                    objective.reachability_map = self.reachability_map
                elif isinstance(objective, Avoid4d):
                    objective.reachability_map = self.reachability_map
                elif isinstance(objective, GoalDistance):
                    objective.fmm_map = self.fmm_map
            else:
                assert (False)

    def _init_obstacle_map(self, obstacle_params=None):
        """ Initializes a new obstacle map."""
        raise NotImplementedError

    def _init_system_dynamics(self):
        """
        If there is a control pipeline (i.e. model based method)
        return its system_dynamics. Else create a new system_dynamics
        instance.
        """
        try:
            return self.planner.control_pipeline.system_dynamics
        except AttributeError:
            p = self.params.planner_params.control_pipeline_params.system_dynamics_params
            return p.system(dt=p.dt, params=p)

    def _init_obj_fn(self):
        """
        Initialize the objective function.
        Use fmm_map = None as this is undefined
        until a goal configuration is specified.
        """
        p = self.params

        obj_fn = ObjectiveFunction(p.objective_fn_params)
        if self.params.cost == 'heuristics':
            if not p.avoid_obstacle_objective.empty():
                obj_fn.add_objective(ObstacleAvoidance(
                    params=p.avoid_obstacle_objective,
                    obstacle_map=self.obstacle_map))
            if not p.goal_angle_objective.empty():
                obj_fn.add_objective(AngleDistance(
                    params=p.goal_angle_objective,
                    fmm_map=None))
            if not p.goal_distance_objective.empty():
                obj_fn.add_objective(GoalDistance(
                    params=p.goal_distance_objective,
                    fmm_map=None))
        elif self.params.cost == 'reachability':
            if not p.avoid_4d_objective.empty():
                obj_fn.add_objective(Avoid4d(
                    params=p.avoid_4d_objective,
                    reachability_map=None))
            if not p.goal_distance_objective.empty():
                obj_fn.add_objective(GoalDistance(
                    params=p.goal_distance_objective,
                    fmm_map=None))
            if not p.reach_avoid_4d_objective.empty():
                obj_fn.add_objective(ReachAvoid4d(
                    params=p.reach_avoid_4d_objective,
                    reachability_map=None))
        return obj_fn

    def _init_fmm_map(self, goal_pos_n2=None):
        p = self.params
        self.obstacle_occupancy_grid = self.obstacle_map.create_occupancy_grid_for_map()

        if goal_pos_n2 is None:
            goal_pos_n2 = self.goal_config.position_nk2()[0]
        # Create fmm_obstacle, fmm_angle and goal map
        return FmmMap.create_fmm_map_based_on_goal_position(
            goal_positions_n2=goal_pos_n2,
            map_size_2=np.array(p.obstacle_map_params.map_size_2),
            dx=p.obstacle_map_params.dx,
            map_origin_2=p.obstacle_map_params.map_origin_2,
            mask_grid_mn=self.obstacle_occupancy_grid)

    def _init_reachability_map(self, goal_pos_n2=None):
        # Different from fmm map, reachability map's dimensions are (x, y), not (y, x)
        p = self.params

        # Get goal map. (goal -1, else 1)
        if goal_pos_n2 is None:
            goal_pos_n2 = self.goal_config.position_nk2()[0]
        goal_array_transpose = ReachabilityMap.get_goal_array_transpose(goal_positions_n2=goal_pos_n2,
                                                                        map_size_2d=np.array(p.obstacle_map_params.map_size_2),
                                                                        map_origin_2d=p.obstacle_map_params.map_origin_2,
                                                                        dx=p.obstacle_map_params.dx)
        goal_array_2d = goal_array_transpose.transpose()

        # Get obstacle map (obstacle -1, else 1)
        if hasattr(self, 'obstacle_occupancy_grid'):
            obstacle_array_transpose = copy.copy(self.obstacle_occupancy_grid)
            obstacle_array_transpose[obstacle_array_transpose == 1] = -1
            obstacle_array_transpose[obstacle_array_transpose == 0] = 1
        else:
            raise Exception("No occupancy grids!")
        obstacle_array_2d = obstacle_array_transpose.transpose()

        # Get map_origin_2d and map_bdry_2d
        map_origin_2d = np.asarray(p.obstacle_map_params.map_origin_2)
        map_bdry_2d = self.obstacle_map.map_bounds[1]

        # Get the resolution
        dx = p.obstacle_map_params.dx

        # Get goal and start position
        goal_pos_2d = self.goal_config.position_nk2()[0, 0].numpy()
        start_pos_2d = self.start_config.position_nk2()[0, 0].numpy()

        return ReachabilityMap(goal_grid_2d=goal_array_2d,
                               obstacle_grid_2d=obstacle_array_2d,
                               map_origin_2d=map_origin_2d,
                               map_bdry_2d=map_bdry_2d,
                               dx=dx,
                               start_pos_2d=start_pos_2d,
                               goal_pos_2d=goal_pos_2d,
                               params=p.reachability_map_params)

    def _init_planner(self):
        p = self.params
        return p.planner_params.planner(simulator=self,
                                        params=p.planner_params)

    # Functions for computing relevant metrics
    # on robot trajectories
    def _dist_to_goal(self, trajectory):
        """Calculate the FMM distance between
        each state in trajectory and the goal."""
        for objective in self.obj_fn.objectives:
            if isinstance(objective, GoalDistance):
                dist_to_goal_nk = objective.compute_dist_to_goal_nk(trajectory)
        return dist_to_goal_nk

    def _calculate_min_obs_distances(self, vehicle_trajectory):
        """Returns an array of dimension 1k where each element is the distance to the closest
        obstacle at each time step."""
        pos_1k2 = vehicle_trajectory.position_nk2()
        obstacle_dists_1k = self.obstacle_map.dist_to_nearest_obs(pos_1k2)
        return obstacle_dists_1k

    def _calculate_trajectory_collisions(self, vehicle_trajectory):
        """Returns an array of dimension 1k where each element is a 1 if the robot collided with an
        obstacle at that time step or 0 otherwise. """
        pos_1k2 = vehicle_trajectory.position_nk2()
        obstacle_dists_1k = self.obstacle_map.dist_to_nearest_obs(pos_1k2)
        return tf.cast(obstacle_dists_1k < 0.0, tf.float32)

    def get_metrics(self):
        """After the episode is over, call the get_metrics function to get metrics
        per episode.  Returns a structure, lists of which are passed to accumulate
        metrics static function to generate summary statistics."""
        dists_1k = self._dist_to_goal(self.vehicle_trajectory)
        init_dist = dists_1k[0, 0].numpy()
        final_dist = dists_1k[0, -1].numpy()
        collisions_mu = np.mean(self._calculate_trajectory_collisions(self.vehicle_trajectory))
        min_obs_distances = self._calculate_min_obs_distances(self.vehicle_trajectory)
        return np.array([self.obj_val,
                         init_dist,
                         final_dist,
                         self.vehicle_trajectory.k,
                         collisions_mu,
                         np.min(min_obs_distances),
                         self.episode_type])

    @staticmethod
    def collect_metrics(ms, termination_reasons=['Timeout', 'Collision', 'Success']):
        ms = np.array(ms)
        if len(ms) == 0:
            return None, None
        obj_vals, init_dists, final_dists, episode_length, collisions, min_obs_distances, episode_types = ms.T
        keys = ['Objective Value', 'Initial Distance', 'Final Distance',
                'Episode Length', 'Collisions_Mu', 'Min Obstacle Distance']
        vals = [obj_vals, init_dists, final_dists,
                episode_length, collisions, min_obs_distances]

        # mean, 25 percentile, median, 75 percentile
        fns = [np.mean, lambda x: np.percentile(x, q=25), lambda x:
        np.percentile(x, q=50), lambda x: np.percentile(x, q=75)]
        fn_names = ['mu', '25', '50', '75']
        out_vals, out_keys = [], []
        for k, v in zip(keys, vals):
            for fn, name in zip(fns, fn_names):
                _ = fn(v)
                out_keys.append('{:s}_{:s}'.format(k, name))
                out_vals.append(_)

        # Log the number of episodes
        num_episodes = len(episode_types)
        out_keys.append('Number Episodes')
        out_vals.append(num_episodes)

        # Log Percet Collision, Timeout, Success, Etc.
        for i, reason in enumerate(termination_reasons):
            out_keys.append('Percent {:s}'.format(reason))
            out_vals.append(1. * np.sum(episode_types == i) / num_episodes)

            # Log the Mean Episode Length for Each Episode Type
            episode_idxs = np.where(episode_types == i)[0]
            episode_length_for_this_episode_type = episode_length[episode_idxs]
            if len(episode_length_for_this_episode_type) > 0:
                mean_episode_length_for_this_episode_type = np.mean(episode_length_for_this_episode_type)
                out_keys.append('Mean Episode Length for {:s} Episodes'.format(reason))
                out_vals.append(mean_episode_length_for_this_episode_type)

        return out_keys, out_vals

    def start_recording_video(self, video_number):
        """ By default the simulator does not support video capture."""
        return None

    def stop_recording_video(self, video_number, video_filename):
        """ By default the simulator does not support video capture."""
        return None

    def render(self, axs, freq=4, render_velocities=False, prepend_title=''):
        if type(axs) is list or type(axs) is np.ndarray:
            self._render_trajectory(axs[0], freq)

            if render_velocities:
                self._render_velocities(axs[1], axs[2])
            [ax.set_title('{:s}{:s}'.format(prepend_title, ax.get_title())) for ax in axs]
        else:
            self._render_trajectory(axs, freq)
            axs.set_title('{:s}{:s}'.format(prepend_title, axs.get_title()))

    def _render_obstacle_map(self, ax):
        raise NotImplementedError

    def _render_trajectory(self, ax, freq=4):
        p = self.params

        self._render_obstacle_map(ax)

        if 'waypoint_config' in self.vehicle_data.keys():
            self.vehicle_trajectory.render([ax], freq=freq, plot_quiver=False)
            self._render_waypoints(ax)
        else:
            self.vehicle_trajectory.render([ax], freq=freq, plot_quiver=True)

        boundary_params = {'norm': p.goal_dist_norm, 'cutoff':
            p.goal_cutoff_dist, 'color': 'g'}
        self.start_config.render(ax, batch_idx=0, marker='o', color='blue')
        self.goal_config.render_with_boundary(ax, batch_idx=0, marker='*', color='black',
                                              boundary_params=boundary_params)

        goal = self.goal_config.position_nk2()[0, 0]
        start = self.start_config.position_nk2()[0, 0]
        text_color = p.episode_termination_colors[self.episode_type]
        ax.set_title('Start: [{:.2f}, {:.2f}] '.format(*start) +
                     'Goal: [{:.2f}, {:.2f}]'.format(*goal), color=text_color)

        final_pos = self.vehicle_trajectory.position_nk2()[0, -1]
        ax.set_xlabel('Cost: {cost:.3f} '.format(cost=self.obj_val) +
                      'End: [{:.2f}, {:.2f}]'.format(*final_pos), color=text_color)

    def _render_waypoints(self, ax):
        # Plot the system configuration and corresponding
        # waypoint produced in the same color
        system_configs = self.vehicle_data['system_config']
        waypt_configs = self.vehicle_data['waypoint_config']
        cmap = matplotlib.cm.get_cmap(self.params.waypt_cmap)
        for i, (system_config, waypt_config) in enumerate(zip(system_configs, waypt_configs)):
            color = cmap(i / system_configs.n)
            system_config.render(ax, batch_idx=0, plot_quiver=True,
                                 marker='o', color=color)

            # Render the waypoint's number at each
            # waypoint's location
            pos_2 = waypt_config.position_nk2()[0, 0].numpy()
            ax.text(pos_2[0], pos_2[1], str(i), color=color)

    def _render_velocities(self, ax0, ax1):
        speed_k = self.vehicle_trajectory.speed_nk1()[0, :, 0].numpy()
        angular_speed_k = self.vehicle_trajectory.angular_speed_nk1()[0, :, 0].numpy()

        time = np.r_[:self.vehicle_trajectory.k] * self.vehicle_trajectory.dt

        if self.system_dynamics.simulation_params.simulation_mode == 'realistic':
            ax0.plot(time, speed_k, 'r--', label='Applied')
            ax0.plot(time, self.commanded_actions_1kf[0, :, 0], 'b--', label='Commanded')
            ax0.set_title('Linear Velocity')
            ax0.legend()

            ax1.plot(time, angular_speed_k, 'r--', label='Applied')
            ax1.plot(time, self.commanded_actions_1kf[0, :, 1], 'b--', label='Commanded')
            ax1.set_title('Angular Velocity')
            ax1.legend()
        else:

            ax0.plot(time, speed_k, 'r--')
            ax0.set_title('Linear Velocity')

            ax1.plot(time, angular_speed_k, 'r--')
            ax1.set_title('Angular Velocity')
