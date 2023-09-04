import os
import pickle
import numpy as np
import time
import tensorflow as tf

from data_sources.image_data_source import ImageDataSource
from systems.dubins_car import DubinsCar
from utils import utils

import random


class VisualNavigationDataSource(ImageDataSource):

    def _get_image_dir_name(self):
        """
        Return the name of a unique directory
        where image data can be saved.
        """
        camera_params = self.p.simulator_params.obstacle_map_params.renderer_params.camera_params
        robot_params = self.p.simulator_params.obstacle_map_params.renderer_params.robot_params
        model_params = self.p.model

        dir_name = 'img_data_{:s}'.format(camera_params.modalities[0])
        dir_name += '_{:d}_{:d}_{:d}'.format(camera_params.width, camera_params.height,
                                             camera_params.img_channels)

        if camera_params.modalities[0] == 'occupancy_grid':
            dir_name += '_{:.3f}_{:.3f}'.format(*model_params.occupancy_grid_dx)

        dir_name += '_{:.2f}_{:.2f}'.format(camera_params.fov_horizontal,
                                            camera_params.fov_vertical)
        dir_name += '_{:.2f}_{:.2f}'.format(camera_params.z_near,
                                            camera_params.z_far)
        dir_name += '_{:.2f}'.format(camera_params.im_resize)

        dir_name += '_{:d}_{:d}_{:d}_{:d}_{:d}_{:.3f}'.format(robot_params.radius,
                                                              robot_params.base,
                                                              robot_params.height,
                                                              robot_params.sensor_height,
                                                              robot_params.camera_elevation_degree,
                                                              robot_params.delta_theta)
        return dir_name

    def _get_n(self, data):
        """
        Returns n, the batch size of the data inside
        this data dictionary.
        """
        return data['vehicle_state_nk3'].shape[0]

    # TODO: Varun- look into efficiency at some point to see if data collection can be sped up
    def generate_data(self):

        # Note (Somil): Since we moved from a string to a list convention for data directories, we are adding
        # additional code here to make sure it is backwards compatible. Moreover, only a single data creation directory
        # can be provided to create the data at the moment.
        if isinstance(self.p.data_creation.data_dir, list):
            assert len(self.p.data_creation.data_dir) == 1
            self.p.data_creation.data_dir = self.p.data_creation.data_dir[0]

        # Create the data directory if required
        if not os.path.exists(self.p.data_creation.data_dir):
            os.makedirs(self.p.data_creation.data_dir)

        # Save a copy of the parameter file in the data_directory
        utils.log_dict_as_json(self.p, os.path.join(self.p.data_creation.data_dir, 'params.json'))

        # Initialize the simulator
        simulator = self.p.simulator_params.simulator(
            self.p.simulator_params)  # Get obstacle map and free_xy_map. (x,y)

        # Generate the data
        counter = 1
        num_points = 0
        self.episode_counter = 0
        while num_points < self.p.data_creation.data_points:  # if num_points less than total set up
            # Reset the data dictionary
            data = self.reset_data_dictionary(self.p)

            while self._num_data_points(data) < self.p.data_creation.data_points_per_file:
                start = time.time()
                # Reset the simulator
                # For a simulator, compute goal_distance and angle_distance, and initiate trajectory data
                simulator.reset()
                # Run the planner for one step
                # Sample a bunch of waypoints, evaluate the cost along the trajectory, and return optimal waypoints and
                # its corresponding image
                simulator.simulate()

                # Ensure that the episode simulated is valid
                if simulator.valid_episode:
                    # Append the data to the current data dictionary
                    self.append_data_to_dictionary(data, simulator)
                    self.episode_counter += 1

                end = time.time()
                elapsed = end - start
                print("The episode", self.episode_counter, "takes time", elapsed)

            # Prepare the dictionary for saving purposes
            self.prepare_and_save_the_data_dictionary(data, counter)

            # Increase the counter
            counter += 1
            num_points += self._num_data_points(data)
            print(num_points)

    def _create_image_dataset(self):
        """
        Create the image dataset by calling the super
        function. Also if needed created a subfolder
        with data from successful navigational goals
        only.
        """
        new_data_dirs = super(VisualNavigationDataSource, self)._create_image_dataset()
        self.p.data_creation.data_dir = new_data_dirs
        return new_data_dirs

    @staticmethod
    def reset_data_dictionary(params):
        """
        Create a dictionary to store the data.
        """
        # Data dictionary to store the data
        data = {}

        # Start configuration information
        data['vehicle_state_nk3'] = []
        data['vehicle_controls_nk2'] = []

        # Goal configuration information
        data['goal_position_n2'] = []
        data['goal_position_ego_n2'] = []

        # Optimal waypoint configuration information
        data['optimal_waypoint_n3'] = []
        data['optimal_waypoint_ego_n3'] = []

        # The horizon of waypoint
        data['waypoint_horizon_n1'] = []

        # Optimal control information
        data['optimal_control_nk2'] = []

        # Episode type information
        data['episode_type_string_n1'] = []
        data['episode_number_n1'] = []

        # Last step information
        # Saved separately from other episode information
        # So that we can decide whether to train on this or not
        data['last_step_vehicle_state_nk3'] = []
        data['last_step_vehicle_controls_nk2'] = []
        data['last_step_goal_position_n2'] = []
        data['last_step_goal_position_ego_n2'] = []
        data['last_step_optimal_waypoint_n3'] = []
        data['last_step_optimal_waypoint_ego_n3'] = []
        data['last_step_optimal_control_nk2'] = []
        data['last_step_data_valid_n'] = []

        data['start_state'] = []
        data['all_waypoint_ego'] = []
        data['all_waypoint'] = []
        data['labels'] = []

        return data

    def _num_data_points(self, data):
        """
        Returns the number of data points inside
        data.
        """
        if type(data['start_state']) is list:
            if len(data['start_state']) == 0:
                return 0
            ns = [x.shape[0] for x in data['start_state']]
            return np.sum(ns)
        elif type(data['start_state']) is np.ndarray:
            return data['start_state'].shape[0]
        else:
            raise NotImplementedError

    # TODO Varun T.: Clean up this code so the structure isnt repeating
    # the function below
    def _append_last_step_info_to_dictionary(self, data, simulator):
        """
        Append data from the last trajectory segment
        to the data dictionary.
        """
        data_last_step = simulator.vehicle_data_last_step
        n = data_last_step['system_config'].n

        data['last_step_vehicle_state_nk3'].append(
            simulator.vehicle_data_last_step['trajectory'].position_and_heading_nk3().numpy())
        data['last_step_vehicle_controls_nk2'].append(
            simulator.vehicle_data_last_step['trajectory'].speed_and_angular_speed_nk2().numpy())

        last_step_goal_n13 = np.broadcast_to(simulator.goal_config.position_and_heading_nk3().numpy(), (n, 1, 3))
        last_step_waypoint_n13 = data_last_step['waypoint_config'].position_and_heading_nk3().numpy()

        # Convert to egocentric coordinates
        start_nk3 = data_last_step['system_config'].position_and_heading_nk3().numpy()
        goal_ego_n13 = DubinsCar.convert_position_and_heading_to_ego_coordinates(start_nk3,
                                                                                 last_step_goal_n13)
        waypoint_ego_n13 = DubinsCar.convert_position_and_heading_to_ego_coordinates(start_nk3,
                                                                                     last_step_waypoint_n13)

        data['last_step_goal_position_n2'].append(last_step_goal_n13[:, 0, :2])

        data['last_step_goal_position_ego_n2'].append(goal_ego_n13[:, 0, :2])

        data['last_step_optimal_waypoint_n3'].append(last_step_waypoint_n13[:, 0, :])
        data['last_step_optimal_waypoint_ego_n3'].append(waypoint_ego_n13[:, 0, :])

        data['last_step_optimal_control_nk2'].append(
            simulator.vehicle_data_last_step['trajectory'].speed_and_angular_speed_nk2().numpy())
        data['last_step_data_valid_n'].append([simulator.last_step_data_valid])
        return data

    def append_data_to_dictionary(self, data, simulator):
        """
        Append the appropriate data from the simulator to the existing data dictionary.
        """
        # Batch Dimension
        n = simulator.vehicle_data['system_config'].n

        # Vehicle data
        # data['vehicle_state_nk3'].append(simulator.vehicle_data['trajectory'].position_and_heading_nk3().numpy())
        data['vehicle_state_nk3'].append(simulator.vehicle_data['spline_trajectory'].position_heading_speed_nk4().numpy())

        # data['vehicle_controls_nk2'].append(simulator.vehicle_data['trajectory'].speed_and_angular_speed_nk2().numpy())
        data['vehicle_controls_nk2'].append(simulator.vehicle_data['spline_trajectory'].acceleration_and_angular_speed_nk2().numpy())

        # Convert to egocentric coordinates
        start_nk3 = simulator.vehicle_data['system_config'].position_and_heading_nk3().numpy()
        data['start_state'].append(start_nk3)
        goal_n13 = np.broadcast_to(simulator.goal_config.position_and_heading_nk3().numpy(), (n, 1, 3))
        waypoint_n13 = simulator.vehicle_data['waypoint_config'].position_and_heading_nk3().numpy()
        # waypoint_n13 = simulator.vehicle_data['waypoint_config'].position_heading_speed_nk4.numpy()
        speed_n11= simulator.vehicle_data['waypoint_config']._speed_nk1.numpy()
        goal_ego_n13 = DubinsCar.convert_position_and_heading_to_ego_coordinates(start_nk3,
                                                                                 goal_n13)
        waypoint_ego_n13 = DubinsCar.convert_position_and_heading_to_ego_coordinates(start_nk3,
                                                                                     waypoint_n13)
        i = 0
        for waypoints in simulator.vehicle_data['all_waypoint']:

            waypoints_n13 = waypoints.position_and_heading_nk3().numpy()
            # print(waypoints_n13.shape)
            # print(waypoints._speed_nk1.numpy().shape)
            # data['all_waypoint'].append(np.concatenate((np.squeeze(waypoints_n13),waypoints._speed_nk1.numpy().reshape(-1,1)),axis=1))
            data['all_waypoint'].append(
                np.concatenate((waypoints_n13, waypoints._speed_nk1.numpy()), axis=2).reshape(-1, 4))
            print (data['all_waypoint'][-1].shape)
            start = np.tile(start_nk3[i], (np.shape(waypoints_n13)[0], 1, 1))
            waypoints_ego = DubinsCar.convert_position_and_heading_to_ego_coordinates(start, waypoints_n13)
            waypoints_ego =np.concatenate((waypoints_ego[:, 0],waypoints._speed_nk1.numpy().reshape(-1,1)),axis=1)


            i += 1
            data['all_waypoint_ego'].append(waypoints_ego)  # 10521,3
        # data['all_waypoint'].append(simulator.vehicle_data['all_waypoint'])
        data['labels'].append(simulator.vehicle_data['labels'])
        # Goal Data
        data['goal_position_n2'].append(goal_n13[:, 0, :2])
        data['goal_position_ego_n2'].append(goal_ego_n13[:, 0, :2])

        # Waypoint data
        data['optimal_waypoint_n3'].append(waypoint_n13[:, 0])
        data['optimal_waypoint_ego_n3'].append(waypoint_ego_n13[:, 0])

        # Waypoint horizon
        data['waypoint_horizon_n1'].append(simulator.vehicle_data['planning_horizon_n1'])

        # Optimal control data
        data['optimal_control_nk2'].append(simulator.vehicle_data['trajectory'].speed_and_angular_speed_nk2().numpy())

        # Episode Type Information
        data['episode_type_string_n1'].append(
            [simulator.params.episode_termination_reasons[simulator.episode_type]] * n)
        # print('typeeeeee')
        # print(data['episode_type_string_n1'])
        data['episode_number_n1'].append([self.episode_counter] * n)

        data = self._append_last_step_info_to_dictionary(data, simulator)
        return data

    def prepare_and_save_the_data_dictionary(self, data, counter):
        """
        Stack the lists in the dictionary to make an array, and then save the dictionary.
        """
        N = 100 # MIN # OF WPS 4000

        # N = 200
        # randomRow = np.random.randint(3, size=N)
        # arr[np.random.randint(arr.numpy().shape[0], size=N), :]
        # random.sample(arr, N)
        # idx = np.random.randint(arr.numpy().shape[0], size=4000)
        # Stack the lists
        # counter= 0
        data_tags = data.keys()
        import matplotlib.pyplot as plt
        fig = plt.figure()
        # ax = fig.add_subplot(111)
        from mpl_toolkits import mplot3d
        ax = plt.axes(projection="3d")

        X = data['all_waypoint_ego'][0]
        y = np.squeeze(data['labels'][0][0])
        # ax.scatter(X[:, 0], X[:, 1], marker='o', c=y)
        # ax.scatter3D(X[:, 0], X[:, 1],X[:, 2], marker='x', c=y)
        # plt.show()

        idxes =[]
        for arr in data['all_waypoint_ego']:
            idx = np.random.randint(arr.shape[0], size=N)
            idxes.append(idx)

        for tag in data_tags:
            # data[tag] = np.concatenate(data[tag], axis=0)
            if tag not in ['labels', 'all_waypoint_ego', 'all_waypoint', 'vehicle_state_nk3', 'vehicle_controls_nk2']:
                data[tag] = np.concatenate(data[tag], axis=0)
            else:
                # tag == 'all_waypoint_ego':
                arr2 = []
                for arr, idx in zip(data[tag], idxes):
                    if tag == 'labels':
                        assert(len(arr) == 1)
                        arr = arr[0]
                    arr2.append(np.expand_dims(arr[idx, :], axis=0))
                data[tag] = np.concatenate(arr2, axis=0)
            # elif tag == 'labels':
            #     i = 0
            #     arr2 = []
            #     for arr3 in data['labels']:
            #         for arr in arr3:
            #             arr2.append(np.expand_dims(arr[idxes[i], :], axis=0))
            #             i += 1
            #     data['labels'] = np.concatenate(arr2, axis=0)
            # elif tag == 'all_waypoint':
            #     for arr3 , idx in zip(data['all_waypoint'],idxes) :
            #         arr1.append(np.expand_dims(arr3[idx, :], axis=0))
            #     data['all_waypoint'] = np.concatenate(arr1, axis=0)
            #     print(data['all_waypoint'] .shape)


                # arr1.append(np.expand_dims(op , axis=0))
            # else:
            #     pass

        # Save the data
        filename = os.path.join(self.p.data_creation.data_dir, 'file%i.pkl' % counter)
        if counter == 6:
            pass

        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)



