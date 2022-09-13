import tensorflow as tf
# from matplotlib.pyplot import hold

from utils import utils

tf.enable_eager_execution(**utils.tf_session_config())

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



##
from systems.dubins_car import DubinsCar

# from Simulator import reset
# from Simulator import _iterate
# from trajectory import SystemConfig
##

from trajectory.trajectory import SystemConfig
import numpy as np
import argparse
import importlib
import os
import time


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
import math
import pickle




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
                                      a_bounds=[-0.4, 0.4]
                                      )
    p.system_dynamics_params.simulation_params = DotMap(simulation_mode='ideal',
                                                        noise_params=DotMap(is_noisy=False,
                                                                            noise_type='uniform',
                                                                            noise_lb=[-0.02, -0.02, 0.],
                                                                            noise_ub=[0.02, 0.02, 0.],
                                                                            noise_mean=[0., 0., 0.],
                                                                            noise_std=[0.02, 0.02, 0.]))
    return p


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

    commanded_actions_nkf = []  # random actions can be added

    dV = []
    goal_states = []


# class SupervisedQ3_waypoint(object):
class Simulator(SimulatorHelper):

# class Simulator(SimulatorHelper):

    def __init__(self, params):

        self.params = params.simulator.parse_params(params)
        self.rng = np.random.RandomState(params.seed)  # Sample some random states, used for initalizing map
        self.obstacle_map = self._init_obstacle_map(self.rng)
        self.obj_fn = self._init_obj_fn()
        self.planner = self._init_planner()
        self.system_dynamics = self._init_system_dynamics()
        self.Q=[]
        self.labels=[]
        self.label0=[]
        self.episode_counter = 0




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


    # while not end_episode:
    # def run(self):
    def simulate(self):


        vehicle_trajectory = self.vehicle_trajectory
        vehicle_data = self.planner.empty_data_dict()
        end_episode = False


        x = np.linspace(0, self.obstacle_map.map_bounds[1][0], self.obstacle_occupancy_grid.shape[1])

        y = np.linspace(0, self.obstacle_map.map_bounds[1][1], self.obstacle_occupancy_grid.shape[0])

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
##
        # if isinstance(self.p.data_creation.data_dir, list):
        #     assert len(self.p.data_creation.data_dir) == 1
        #     self.p.data_creation.data_dir = self.p.data_creation.data_dir[0]
        #
        # # Create the data directory if required
        # if not os.path.exists(self.p.data_creation.data_dir):
        #     os.makedirs(self.p.data_creation.data_dir)
        #
        # # Save a copy of the parameter file in the data_directory
        # utils.log_dict_as_json(self.p, os.path.join(self.p.data_creation.data_dir, 'params.json'))
        #
        # # Initialize the simulator
        # simulator = self.p.simulator_params.simulator(
        #     self.p.simulator_params)  # Get obstacle map and free_xy_map. (x,y)

        start_time = time.time()
        # simulator.reset()

        # Generate the data
        counter = 1
        num_points = 0

        # Reset the data dictionary
        # data = self.reset_data_dictionary(self.p)


        num_episode = 10

        count = 0

        # for episode in range(num_episode):

##my simulate code


        config = self.start_config


# with tf.device('/cpu:0'):

        dataForAnImage=[]
        #num = 10000
        num_samples = 1


        actions_waypoints = [[0.1, 0.05, .8],     [0.1, -0.05, -.8],
                             [0.15, -0.1, -0.7],  [0.15, 0.1, 0.7],
                             [0.2, 0.1, 0.55],    [0.2, -0.1, -0.55],
                             [0.25, -0.05, -0.3], [0.25, 0.05, 0.3],
                             [0.35, -0.1, -0.6],  [0.35, 0.1, 0.6],
                             [0.35, -0.05, -0.2], [0.35, 0.05, 0.2],
                             [0.4, 0.1, 0.7],     [0.4, -0.1, -.7],
                             [0.45, -0.1, -0.2],  [0.45, 0.1, 0.2],
                             [.5, 0, 0.1],        [.5, 0, -0.1],
                             [.55, 0.1, 0.05],    [.55, -0.1, -0.05]]

        f=6


        # actions_waypoints=[[ f * i for i in inner ] for inner in actions_waypoints]

        # actions_waypoints =[[10, 5, .8]]


        # speedf = np.float16(np.linspace(0, 0.6, num=3))


        v0 = config.speed_nk1()[0][0][0].numpy()
        start_state = [0, 0, 0, v0]
        x0 = start_state
                # for v0 in speed:
        #
        # start_state = [0, 0, 0, v0]
        # x0 = start_state
        # if v0 >= 0.1 and v0 <= 0.5:
        #     speedf = [v0 - .1, v0, v0 + 0.1]
        # elif v0 < 0.10:
        #     speedf = [0, v0, v0 + 0.1]
        # elif v0 > 0.5:
        #     speedf = [v0 - .1, v0, 0.6]
        #
        #

        start_pose = []
        waypointAction = []
        image = []
        global_pts = []
        local_pts = []
        local_pts_camera=[]
        num_images=range (4)
        num_sample_generated=0
        self.labels=[]


        for xf in actions_waypoints:

            # we may discover that partway through a trajectory that the trajectory takes us out of bounds;
            # therefore, we need to use a boolean var to track whether or not we will continue processing the
            # current waypoint
            processWaypoint = True

            Transformation = [[np.cos(config.heading_nk1()[0][0][0]),
                               -np.sin(config.heading_nk1()[0][0][0]),
                               config.position_nk2()[0][0][0]],
                              [np.sin(config.heading_nk1()[0][0][0]),
                               np.cos(config.heading_nk1()[0][0][0]),
                               config.position_nk2()[0][0][1]],
                              [0, 0, 1]]
            target_state = np.array(Transformation).dot([xf[0], xf[1], 1])
            # phi1 = phi % 2 * math.pi
            # goal_heading_nk1=phi1-math.pi
            # # if goal_heading_nk1 < np.float32(-np.pi):
            # #     goal_heading_nk1=goal_heading_nk1 + 2*np.pi
            # # if goal_heading_nk1 > np.float32(np.pi):
            # #     goal_heading_nk1 = goal_heading_nk1 - 2 * np.pi
            if 0 < target_state[0] < self.obstacle_map.map_bounds[1][0] and 0 < target_state[1] < self.obstacle_map.map_bounds[1][1]:

                count += 1
                target_state[2]= xf[2] + config.heading_nk1()[0][0][0]
                target_state[2] = np.arctan2(np.sin(target_state[2]), np.cos(target_state[2]))

                # speedf = np.float16(np.random.uniform(low=0, high=0.6, size=(3,)))
                # speedf = np.random.uniform(low=0, high=0.6, size=(3,))
                speedf = [v0]
                # print('speedf is' ,v0)

                for vf in speedf:

                    vf1=[vf]
                    goal_state = [y for x in [target_state, vf1] for y in x]

                    # t = np.arange(0, 10, f*)
                    # p.dt=0.05
                    t = np.arange(0, 1*f, 0.05 )

                    traj_const = fs.point_to_point(vehicle_flat, t, x0, u0, goal_state, uf,basis=fs.PolyFamily(8))# constraints=constraints,

                    # ,cost=cost_fcn)
                    # Create the trajectory

                    x, u = traj_const.eval(t)

                    max_val_w = 1.1
                    max_val_a = 0.4
                    max_val_v = 0.6
                    if abs(u[0]).max()<=max_val_w and abs(u[1]).max()<=max_val_a and abs(x[3]).max()<=max_val_v and x[3].min()>=0 :

                        print('it can reach to', goal_state)

                        # local_point=x
                        global_traj = x

                        ttc = []
                        V = []

                        # local_pts_camera=[]
                        n=1

                        for k in range(len(t)): #timestep over traj

                            global_point=x[:, k]
                            V.append(my_interpolating_functionV(global_point))
                            # ttc.append(my_interpolating_function(global_point))

                        #print("final state in world: ", global_point)
                        # print("final state in world: ", local_point[:][])
                        # self.TTCmin = min(ttc)
                        # print("min of TTC is: ", self.TTCmin)
                        self.discount = 0.90
                        self.gamma = 1.0
                        # self.Q0=self.gamma * (  #
                        #         dt + self.discount * (1 - pow(self.discount, self.TTCmin + 1)) / (
                        #         1 - self.discount))
                        # print("Q of this action-waypoint is: ", self.Q0)
                        self.V = min(V)

                        self.label0 = np.sign(self.V)
                        self.labels.append(self.label0)

                        num_sample_generated +=1

                        print("label of this action-waypoint is: ", self.label0)


                        # self.Q.append(self.Q0)


                        # print("num samples collected: ", count)
                        # print(" ")
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
                        start=[ 0, (crop_size[0]-1)/2]
                        # local_point_camera=np.array(local_point[0][0])// dx_m+start[0], np.array(local_point[0][1])/ dx_m+start[1]
                        # local_pts_camera.append(local_point_camera)

                        start_pose.append(np.concatenate((start_speed_nk1.numpy(), start_heading_nk1.numpy())))
                        waypointAction.append(goal_state)
                        image.append(rgb_image_1mk3)

                    # end of if
                    else:
                        out_x3 = np.clip(x[3], a_min=0, a_max=max_val_v)
                        x[3] = out_x3
                        out_u0 = np.clip(u[0], a_min = -max_val_w, a_max = max_val_w)
                        out_u1 = np.clip(u[1], a_min=np.maximum(-max_val_a*np.ones((len(x[3]),1)),-1 * x[3] / dt), a_max=np.minimum(max_val_a*np.ones((len(x[3]),1)),(max_val_v-x[3])/dt))
                        out_u0[0] = 0
                        u[0]=out_u0
                        u[1]=out_u1[1,:]

                        # p = create_params()

                        p = DotMap()
                        p.seed = 1
                        p.n = 1
                        p.dt = .05
                        p.dk = int(f/p.dt)
                        p.quad_coeffs = [1.0, 1.0, 1.0, 1e-10, 1e-10]
                        p.linear_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

                        p.system_dynamics_params = DotMap(system=DubinsV4,
                                                          dt=.05,
                                                          v_bounds=[0.0, .6],
                                                          w_bounds=[-1.1, 1.1],
                                                          a_bounds=[-0.4, 0.4]
                                                          )
                        p.system_dynamics_params.simulation_params = DotMap(simulation_mode='ideal',
                                                                            noise_params=DotMap(is_noisy=False,
                                                                                                noise_type='uniform',
                                                                                                noise_lb=[-0.02, -0.02,
                                                                                                          0.],
                                                                                                noise_ub=[0.02, 0.02,
                                                                                                          0.],
                                                                                                noise_mean=[0., 0., 0.],
                                                                                                noise_std=[0.02, 0.02,
                                                                                                           0.]))

                        n, k = p.n, p.dk

                        dubins = p.system_dynamics_params.system(p.dt, params=p.system_dynamics_params)

                        start_time_sim = time.time()
                        # start_pos_n13 = tf.constant(np.array([[[0.0, 0.0, 0.0, v0]]], dtype=np.float32)) ### change this to global frame: config.position_and_heading_nk3(),config.speed_nk1()
                        start_pos_n13 = np.concatenate([config.position_and_heading_nk3().numpy(), config.speed_nk1().numpy()], axis=2)
                        acceleration = (np.ones((n, k ), dtype=np.float32) * u[1]).reshape(1,k,1)
                        angular_speed_nk1 = np.float32(u[0][None, :, None])
                        u_nk2 = tf.constant(np.concatenate([angular_speed_nk1, acceleration], axis=2))
                        u_nk2=np.float32(u_nk2)

                        #
                        traj_ref = dubins.simulate_T(start_pos_n13, u_nk2, k)


                        # traj_ref .eval(session=tf.compat.v1.Session())
                        end_time_sim = time.time()
                        time_sime=end_time_sim - start_time_sim

                        # start_pos_nk2 = tf.concat([start_posx_nk1, start_posy_nk1], axis=2)
                        start_heading_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * config.heading_nk1()[0][0][0]
                        # # Initial SystemConfig is [0, 0, 0, v0, 0]
                        start_speed_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * config.speed_nk1()

                        V = []
                        # aborted = False
                        for j in range(1,k):

                            start_time = time.time()
                            # ###

                            global_point = traj_ref[j]
                            global_point = global_point.numpy()
                            print("iteration " + str(j) + " of " + str(k))
                            if global_point[0][0][0] > self.obstacle_map.map_bounds[1][0] or global_point[0][0][0] < 0 or  0 > global_point[0][0][1]  or global_point[0][0][1] > self.obstacle_map.map_bounds[1][1]:
                                processWaypoint = False
                                print("-- breaking")
                                break

                            # if not aborted:
                            global_point[0][0][2] = np.arctan2(np.sin(global_point[0][0][2]), np.cos(global_point[0][0][2]))

                            # ttc.append(my_interpolating_function(global_point))
                            # V.append(my_interpolating_functionV(global_point))
                            V.append(my_interpolating_functionV(global_point))

                        # did we not find an out-of-bounds trajectory?
                        if processWaypoint:

                            self.V = min(V)
                            self.label0 = np.sign(self.V)

                            num_sample_generated += 1

                            self.labels.append(self.label0)

                            count += 1
                            # print("num samples collected: ", count)

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

                            start_pose.append(np.concatenate((start_speed_nk1.numpy(), start_heading_nk1.numpy())))
                            waypointAction.append(np.array(goal_state))
                            image.append(rgb_image_1mk3)

                        #endif

                    # dataForAnImage={'start_pose':np.array(start_pose)*(np.array(waypointAction).shape[0]),
                    #     'image': np.array(image)*(np.array(waypointAction).shape[0]),'waypointAction':np.array(waypointAction), 'labels': np.transpose(np.array(self.labels))}
            dataForAnImage={'start_pose':np.array(start_pose , dtype=object),
                    'image': np.array(image  , dtype=object).squeeze(),'waypointAction':np.array(waypointAction  , dtype=object), 'labels': np.array(self.labels  , dtype=object)}

                    # dataForAnImage_TF=tf.data.Dataset.from_tensor_slices((np.array(start_pose),np.array(image).squeeze(), np.array(waypointAction), np.array(self.labels)))

                # episode_counter=self.episode_counter


            return dataForAnImage

    # plt.scatter(start[0], start[1], marker='*', color='green',s=200, label='start')
    # plt.scatter(local_pts_camera[0],local_pts_camera[1],marker='+',color='red',s=200, label='waypoints')
    #     # plt.scatter(local_pts_camera[0], local_point_camera[1], marker='+', color='red', s=200,
    #     #             label='trajectory')
    # plt.arrow(start[0], start[1], start_speed_nk1*np.cos(start_heading_nk1), start_speed_nk1*np.sin(start_heading_nk1), width = 0.05)
    # plt.arrow(local_pts_camera[0],local_pts_camera[1], start_speed_nk1*np.cos(start_heading_nk1), start_speed_nk1*np.sin(start_heading_nk1), width = 0.15)
    # plt.legend(loc='lower left')
    # plt.show()


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
            # print('start is', self.start_config)
            # Reset self.goal_config. If there is no available goals, reset_start = True, then reset the start again.
            reset_start = self._reset_goal_configuration(self.rng)
            # print('goal is' , self.goal_config)
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
            obs_margin1 = self.params.avoid_obstacle_objective.obstacle_margin1
            obs_margin0 = self.params.avoid_obstacle_objective.obstacle_margin0
            dist_to_obs = 0.
            while dist_to_obs >= obs_margin1 or obs_margin0 >= dist_to_obs:  # Change here for adversarial data collection (closer start position to
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
            # speed_111 = np.zeros((1, 1, 1))
            speed_111 = rng.uniform(p.speed.bounds[0], p.speed.bounds[1], (1, 1, 1))
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
            # ang_speed_111 = np.zeros((1, 1, 1))
            # ang_speed_111 = np.ones((1, 1, 1)) * (1e-10)
            ang_speed_111 = rng.uniform(p.ang_speed.bounds[0], p.ang_speed.bounds[1], (1, 1, 1))
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
        print("start_112 is", start_112)
        print("heading_111 is", heading_111)
        print('speed_111 is', speed_111)
        print('ang_speed_111 is' , ang_speed_111)

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
            print('fmm_l2_gap is' , fmm_l2_gap)
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
            print ('goal_112 is' , goal_112)
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

# if __name__ == '__main__':
#     os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#     SupervisedQ3_waypoint().run()

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
    u[0] = 1 / (1 + (zflag[0][1] / zflag[0][1]) ** 2 ) * (
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
    # plt.show()


vehicle_flat = fs.FlatSystem(forward=vehicle_flat_forward, reverse=vehicle_flat_reverse, inputs=2,
                             states=4)
# x0 = [0., 0,0, 0.25]
# x0 = [0., 0, 0, 0.35]
u0 = [0, 0.]
uf = [0, 0]
Tf = 1
dt = 0.05
t = np.linspace(0, Tf, 20)
scores = []
# Define a set of basis functions to use for the trajectories
poly = fs.PolyFamily(8)
fig = plt.figure()

# cost_fcn = opt.state_poly_constraint(vehicle_flat, np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]), [0,0,0,0.7])

lb, ub = [-1.1, -.4], [1.1, 0.4]
constraints = [opt.input_range_constraint(vehicle_flat, lb, ub)]
out_u0 = []
out_u1 = []
out_x3 = []

count = 0