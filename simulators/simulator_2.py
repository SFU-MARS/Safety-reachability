import tensorflow as tf
import numpy as np
from objectives.objective_function import ObjectiveFunction
from objectives.angle_distance import AngleDistance
from objectives.goal_distance import GoalDistance
from objectives.obstacle_avoidance import ObstacleAvoidance
from objectives.reach_avoid_4d import ReachAvoid4d
from objectives.reach_avoid_3d import ReachAvoid3d
from objectives.avoid_4d import Avoid4d
from params import simulator
from trajectory.trajectory import SystemConfig, Trajectory
from simulators.simulator_helper import SimulatorHelper
from utils.fmm_map import FmmMap
#from reachability.reachability_map import ReachabilityMap
from reachability.reachability_map1 import ReachabilityMap
import matplotlib
import copy
from objectives.objective_function import Objective
from sbpd.sbpd_renderer import SBPDRenderer
from obstacles.obstacle_map import ObstacleMap


class Simulator_2(SimulatorHelper):

    def __init__(self, params):
        self.params = params.simulator.parse_params(params)
        self.rng = np.random.RandomState(params.seed)  # Sample some random states, used for initalizing map
        self.obstacle_map = self._init_obstacle_map(self.rng)
        self.obj_fn = self._init_obj_fn()
        self.planner = self._init_planner()
        self.system_dynamics = self._init_system_dynamics()

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
        commanded_actions_nkf = []
        while not end_episode:
            trajectory_segment, next_config, data, commanded_actions_1kf = self._iterate(
                config)  # while not get to the goal, or not collide? Keep iterating
            print(next_config)
            print(data)
            print(commanded_actions_1kf)

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

        planner_data = self.planner.optimize(config)  # Given config (start configuration)
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

    def reset(self,seed=-1):
        """Reset the simulator. Optionally takes a seed to reset
        the simulator's random state."""

        if seed != -1:

            self.rng.seed(seed)

        # Note: Obstacle map must be reset independently of the fmm map.
        # Sampling start and goal may depend on the updated state of the
        # obstacle map. Updating the fmm map depends on the newly sampled goal.
        #reset_start = True
        #while reset_start:

        self._reset_obstacle_map(self.rng)  # Do nothing here
        self.fmm_map=self.obstacle_map.fmm_map
        #self.obstacle_map = self._init_obstacle_map(self.rng)

        #p = self.params
        self.obstacle_occupancy_grid = self.obstacle_map.occupancy_grid_map

        # Initiate and update a reachability map (reach_avoid or avoid)

        # if self.params.cost == 'reachability':
        #     self._get_reachability_map()

            # return self.reachability_map.avoid_4d_map.compute_voxel_function(trajectory.position_nk2(),
            #                                                                  trajectory.heading_nk1(),
            #                                                                  trajectory.speed_nk1())  # !
        self._reset_start_configuration(self.rng)  # Reset self.start_config

        #self.reachability_map = reachability_map
        #self._position_nk2, self._speed_nk1,
        #self._acceleration_nk1, self._heading_nk1,
            # Manually restart the start and goal (only work for single goal)
            # reset_start = self._reset_start_goal_manually(start_pos=[8.65, 50.25], goal_pos=[8.60, 47.15])

        # self._update_fmm_map()  # Compute fmm_angle and fmm_goal, wrap it into voxel func


        if self.params.cost == 'reachability':
            self._get_reachability_map()

        # Update objective functions, may include reachability cost
        self._update_obj_fn()

        self.vehicle_trajectory = Trajectory(dt=self.params.dt, n=1, k=0)
        self.obj_val = np.inf
        self.vehicle_data = {}


        # Initiate and update a reachability map (reach_avoid or avoid)


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
        self.TTCs=100
        # self.acceleration_nk1 = [0, 0.2, -0.4]
        # self.angular_speed_nk1 = [0, 5.5, - 1.1]
        # self.actions = np.array((self.acceleration_nk1, self.angular_speed_nk1)).T
        # for i in self.actions:

        # Reset the position
        # i=[0,0]
        dt=0.05 #0.05
        self.Q = []

        sample=[]
        while True:
            if p.position.reset_type == 'random':
            # Select a random position on the map that is at least obstacle margin
            # away from the nearest obstacle
            # obs_margin = self.params.avoid_obstacle_objective.obstacle_margin1
            # dist_to_obs = 0.
            #while dist_to_obs <= obs_margin:  # Change here for adversarial data collection (closer start position to
                # the obstacles)
                start_112 = self.obstacle_map.sample_point_112(rng)

                print(start_112)

                #dist_to_obs = tf.squeeze(self.obstacle_map.dist_to_nearest_obs(start_112))
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
            self._get_reachability_map()

            # from tempfile import TemporaryFile
            # outfile = TemporaryFile()
            # np.save(outfile,self.reachability_map.obstacle_grid_2d)
            # _ = outfile.seek(0)
            # np.load(outfile)
            # with open('test.npy', 'wb') as f:
            #     np.save(f, self.reachability_map.obstacle_grid_2d)
            # np.save('obstacle_grid_2d.npy', self.reachability_map.obstacle_grid_2d)
            # obstacle_2d = np.load("/local-scratch/tara/project/WayPtNav-reachability/obstacle_grid_2d.npy")
            # obstacle_2dT = obstacle_2d.T
            # obstacle_4dT = np.empty((9, 31) + obstacle_2dT.shape, dtype=obstacle_2d.dtype)
            # obstacle_4dT[:] = obstacle_2dT
            # obstacle_4d = obstacle_4dT.T
            # np.save('obstacle_grid_4d_ver2.npy', obstacle_4d)
            #elf.speed_nk1 = [0, 0.6]
            # self.acceleration_nk1 = np.zeros((1, 1, 1))

            ####
            # self.acceleration_nk1 = [-0.4]

            #self.angular_speed_nk1 = [-0.5, 0.5]
            # self.angular_speed_nk1 = np.zeros((1, 1, 1))
            # self.angular_speed_nk1 = [1.1]
            # self.TTCs = []
            self.TTCs0 = self.reachability_map.avoid_4d_map.compute_voxel_function(start_112,
                                                                                  heading_111,
                                                                                  speed_111)
            # print(self.TTCs0)
            # self.acceleration_nk1 = [-0.4, 0, 0.4]
            # self.angular_speed_nk1 = [-1.1, 0, 1.1]
            num_actions=5
            self.acceleration_nk1 = np.arange(-1.4, 1.4, 2.8 / num_actions)  # num_actions-1?
            self.angular_speed_nk1 = np.arange(-1.1, 1.1, 2.2 / num_actions)
            self.actions = np.array((self.acceleration_nk1, self.angular_speed_nk1)).T
            # self.start_config.trainable_variables=self.start_config.trainable_variables
            # self.start_config.trainable_variables=np.array(self.start_config.trainable_variables[0].read_value())
            for i in self.actions:
                position_nk1_next0, position_nk1_next1, self.speed_nk1_next, self.heading_nk1_next = self.carmodel(
                    self.start_config.trainable_variables, i, dt)

                # position_nk1_next = np.array(
                #     [np.array(position_nk1_next1)[:, :, :, :, 0], np.array(position_nk1_next0)[:, :, :, :, 0]])
                position_nk1_next=np.concatenate((position_nk1_next0, position_nk1_next1),axis=0)
                # print(position_nk1_next.shape)

                position_nk1_next_reshape = np.reshape(position_nk1_next, (1, 1, 2))  # y,x
                # position_nk1_next=[]
                # position_nk1_next[:,:,0],position_nk1_next[:,:,1] =position_nk1_next1, position_nk1_next0

                self.TTCs = self.reachability_map.avoid_4d_map.compute_voxel_function(position_nk1_next_reshape,
                                                                                           self.heading_nk1_next,
                                                                                           self.speed_nk1_next)
                # print(self.TTCs)

                self.discount = 0.90
                self.Q0 = dt + (1 - pow(self.discount, self.TTCs0 + 1)) / (1 - self.discount)
                self.Q= dt + (1 - pow(self.discount, self.TTCs + 1)) / (1 - self.discount)  # dt from TRAJECTORY (action)
                print(self.Q0)
                print(self.Q)
                count = +1

                self._r = SBPDRenderer.get_renderer(self.params.obstacle_map_params.renderer_params)

                # self.top=self._r._get_topview(simulator.start_config.trainable_variables[0].numpy().transpose(),simulator.start_config.trainable_variables[3].numpy())

                self.dx=0.05
                self.map_origin_2=[0,0]
                # starts_n2=self._point_to_map (start_112).transpose()
                starts_n2=self._point_to_map(np.reshape(start_112, (1, 2)))
                # starts_n2 = self._point_to_map(np.reshape(self.start_config.trainable_variables[0],(1,2)))# .transpose()
                # starts_n2 = self._point_to_map(np.array(self.start_config.trainable_variables[0]))

                next_n2 = self._point_to_map(np.reshape(position_nk1_next, (1,2)))
                # next_n2 = self._point_to_map(position_nk1_next)

                thetas_n1 = np.reshape(self.start_config.trainable_variables[3], (1, 1))
                # thetas_n1 = self.start_config.trainable_variables[3]
                # next_n1 = np.reshape(self.heading_nk1_next,(1,1))
                next_heading_n1 = self.heading_nk1_next

                self.image = self._r._get_rgb_image(starts_n2, thetas_n1)
                self.image2 = self._r._get_rgb_image(next_n2, next_heading_n1)

                # self.image = self._r.render_images(starts_n2, thetas_n1)
                sample=[self.image,i,self.Q]
                self.image=np.reshape(self.image,(224,224,3))
                import matplotlib.pyplot as plt
                plt.imshow(np.uint8(self.image), interpolation='none')
                plt.show()

                self.top = self._r._get_topview(starts_n2, thetas_n1)
                self.top = np.reshape(self.top, (64,64) )
                plt.imshow(np.uint8(self.top), interpolation='none')
                plt.show()

                ####

            # The system dynamics may need the current starting position for
            # coordinate transforms (i.e. realistic simulation)
            self.system_dynamics.reset_start_state(self.start_config)

    def _point_to_map(self, pos_2, cast_to_int=False):
        """
        Convert pos_2 in real world coordinates
        to a point on the map.
        """
        map_pos_2 = pos_2/self.dx -self.map_origin_2
        if cast_to_int:
            map_pos_2 = map_pos_2.astype(np.int32)
        return map_pos_2

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
            #if not p.goal_distance_objective.empty():
            #    obj_fn.add_objective(GoalDistance(
            #        params=p.goal_distance_objective,
            #        fmm_map=None))
            #if not p.reach_avoid_4d_objective.empty():
            #    obj_fn.add_objective(ReachAvoid4d(
            #        params=p.reach_avoid_4d_objective,
            #        reachability_map=None))
        return obj_fn

    def _init_fmm_map(self, goal_pos_n2=None):
        p = self.params
        self.obstacle_occupancy_grid = self.obstacle_map.create_occupancy_grid_for_map()

        #if goal_pos_n2 is None:
            #goal_pos_n2 = self.goal_config.position_nk2()[0]
        # Create fmm_obstacle, fmm_angle and goal map
        return FmmMap.create_fmm_map_based_on_goal_position(
            goal_positions_n2=goal_pos_n2,
            map_size_2=np.array(p.obstacle_map_params.map_size_2),
            dx=p.obstacle_map_params.dx,
            map_origin_2=p.obstacle_map_params.map_origin_2,
            mask_grid_mn=self.obstacle_occupancy_grid)

    def _init_reachability_map(self, goal_pos_n2=[0, 0]):
        # Different from fmm map, reachability map's dimensions are (x, y), not (y, x)
        p = self.params
        #self.obstacle_occupancy_grid = self.obstacle

        # Get goal map. (goal -1, else 1)
        #if goal_pos_n2 is None:
        #    goal_pos_n2 = self.goal_config.position_nk2()[0]
        # goal_array_transpose = ReachabilityMap.get_goal_array_transpose(goal_positions_n2=goal_pos_n2,
        #                                                                 map_size_2d=np.array(p.obstacle_map_params.map_size_2),
        #                                                                 map_origin_2d=p.obstacle_map_params.map_origin_2,
        #                                                                 dx=p.obstacle_map_params.dx)
        #goal_array_2d = goal_array_transpose.transpose()

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
        #goal_pos_2d = self.goal_config.position_nk2()[0, 0].numpy()
        start_pos_2d = self.start_config.position_nk2()[0, 0].numpy()

        return ReachabilityMap(#goal_grid_2d=goal_array_2d,
                               obstacle_grid_2d=obstacle_array_2d,
                               map_origin_2d=map_origin_2d,
                               map_bdry_2d=map_bdry_2d,
                               dx=dx,
                               start_pos_2d=start_pos_2d,
                               #goal_pos_2d=goal_pos_2d,
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
    def carmodel(self, start_config, i,dt):
        #self._position_nk2, self._speed_nk1,
        #self._acceleration_nk1, self._heading_nk1,
        #self._angular_speed_nk1, self._angular_acceleration_nk1
        """ A discrete time dubins car with state
        [x, y, theta] and actions [v, w]. The dynamics are:

        x(t+1) = x(t) + saturate_linear_velocity(v(t)) cos(theta_t)*delta_t
        y(t+1) = y(t) + saturate_linear_velocity(v(t)) sin(theta_t)*delta_t
        theta(t+1) = theta_t + saturate_angular_velocity(w(t))*delta_t
        """
        #
        # Dubins4D
        # x, y, v, theta and actions [a, w]
        # x(t+1) = x(t) + v(t)) cos(theta_t)*delta_t
        # y(t+1) = y(t) + (v(t)) sin(theta_t)*delta_t
        # v(t + 1) = v_t + a(t))*delta_t
        # theta(t+1) = theta_t + w(t))*delta_t

        self.acceleration_nk1=i[0]
        self.angular_speed_nk1 = i[1]


        delta_x_nk3 = [(start_config[2]) * tf.cos(start_config[3]),
                                (start_config[2]) * tf.sin(start_config[3]),
                                (self.acceleration_nk1).reshape(1,1,1), (self.angular_speed_nk1).reshape(1,1,1)]

        # x_dot = tf. Variable(0, "x_dot")
        # y_dot = tf. Variable(0, "y_dot")
        # v_dot = tf. Variable(0, "v_dot")
        # theta_dot = tf. Variable(0, "theta_dot")
        # x_dot[0] = start_config[2] * tf.cos(start_config[3])
        # y_dot[0] = start_config[2] * tf.sin(start_config[3])
        # v_dot[0] = self.acceleration_nk1
        # theta_dot[0] = self.angular_speed_nk1
        position_nk1_next0=start_config[0][:,:,0]+ dt *delta_x_nk3[0]
        position_nk1_next1=start_config[0][:,:,1]+ dt *delta_x_nk3[1]
        self.speed_nk1_next=start_config[1]+ dt *delta_x_nk3[2]
        self.heading_nk1_next=start_config[2]+ dt *delta_x_nk3[3]
        # next_config=start_config[:3] + dt * delta_x_nk3
        return position_nk1_next0,position_nk1_next1, self.speed_nk1_next,self.heading_nk1_next
        # return next_config

        # return (x_dot[0], y_dot[0], v_dot[0], theta_dot[0])
        # self.speed_nk1_next = start_config[1] + self.acceleration_nk1 * dt
        # self.heading_nk1_next=start_config[3]+self.angular_speed_nk1*dt
        #
        # x0=start_config[0][0][0][0]
        # # print(x0.size)
        # # x0 = np.array(start_config[0, 0][0])
        # # x0=start_config[:,:,:1]
        # # x0=start_config[0][0][0]
        #
        # # y0 = np.array(start_config[0].read_value())[0, 0][1]
        # y0 = start_config[0][0][0][1]
        # position_nk1_next0=[]
        # position_nk1_next1 = []
        # position_nk1_next0 = start_config[1].numpy()*np.cos(start_config[3].numpy())*dt+x0
        # #position_nk1_next0 = self.speed_nk1_next.numpy() * np.cos(self.heading_nk1_next) * dt + x0,
        # # position_nk1_next0=list(position_nk1_next0)
        # position_nk1_next1 = start_config[1].numpy()*np.sin(start_config[3].numpy())*dt+y0
        # #position_nk1_next1 = self.speed_nk1_next.numpy() * np.sin(self.heading_nk1_next) * dt + y0,
        # # position_nk1_next1 = list(position_nk1_next1)
        #
        # x0 = position_nk1_next0
        # y0 = position_nk1_next1
        # position_nk1_next0 = []
        # position_nk1_next1 = []
        # position_nk1_next0 = start_config[1].numpy() * np.cos(start_config[3].numpy()) * dt + x0,
        # # position_nk1_next0 = self.speed_nk1_next.numpy() * np.cos(self.heading_nk1_next) * dt + x0,
        # # position_nk1_next0 = list(position_nk1_next0)
        # position_nk1_next1 = start_config[1].numpy() * np.sin(start_config[3].numpy()) * dt + y0,
        # # position_nk1_next1 = self.speed_nk1_next.numpy() * np.sin(self.heading_nk1_next) * dt + y0,
        # # position_nk1_next1 = list(position_nk1_next1)
        # start_config[0], start_config[1], start_config[2], start_config[3], start_config[4] = [position_nk1_next0,
        #                                                                                        position_nk1_next1], self.speed_nk1_next, self.acceleration_nk1, self.heading_nk1_next, self.angular_speed_nk1
        #
        # return position_nk1_next0,position_nk1_next1, self.speed_nk1_next,self.heading_nk1_next

