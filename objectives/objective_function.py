import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from time import time

# Within the goal_mask distance to the goal, the robot is assumed to reach the goal
goal_mask = 0.3

class Objective(object):
    def evaluate_objective(self, trajectory):
        raise NotImplementedError


from objectives.reach_avoid_4d import ReachAvoid4d
from objectives.reach_avoid_3d import ReachAvoid3d
from objectives.avoid_4d import Avoid4d


from objectives.goal_distance import GoalDistance


class ObjectiveFunction(object):
    """
    Define an objective function.
    """

    def __init__(self, params):
        self.params = params
        self.objectives = []


    def vz_values(self, trajectory):

        y = np.linspace(0, 26.05, 521)
        x = np.linspace(0, 30, 600)
        xx, yy = np.meshgrid(x, y)
        xy = np.stack((xx, yy), axis=-1).reshape((-1, 2))

        theta = np.ones_like(xy[:, 0]) * np.deg2rad(45)
        v = np.ones_like(xy[:, 0]) * 0.25

        xytv = np.zeros((xy.shape[0], 4))
        xytv[:, :2] = xy
        xytv[:, 2] = theta
        xytv[:, 3] = v

        # xyt = np.zeros((xy.shape[0], 3))
        # xyt[:, :2] = xy
        # start = np.tile(start_nk3, (np.shape(xyt)[0], 1))
        # with tf.device('cpu'):
        #     start = tf.convert_to_tensor(np.expand_dims(start, 1), dtype=tf.float32)
        #     xyt = tf.convert_to_tensor(np.expand_dims(xyt, 1), dtype=tf.float32)
        #     xyt_world = DubinsCar.convert_position_and_heading_to_ego_coordinates(start, xyt)

        xytv = np.expand_dims(xytv, 0).astype(np.float32)
        values = self.objectives[0].reachability_map.avoid_4d_map.compute_voxel_function(xytv[:, :, :2],
                                                                                         xytv[:, :, 2:3],
                                                                                         xytv[:, :, 3:4])
        img = values.numpy().reshape((521, 600)).T
        img[(img > 1.2) | (img < -1.1)] = -1

        # plt.matshow(img)
        # plt.colorbar()
        # plt.show()
        #
        # plt.matshow(self.objectives[0].reachability_map.obstacle_grid_2d)
        # plt.colorbar()
        # plt.show()

        objective_values_by_tag = self.evaluate_function_by_objective(trajectory)
        obj = objective_values_by_tag[0][1].numpy()
        unsafe =0
        stamp = time() / 1e5
        pp = PdfPages(f"traj_labels_{stamp:.6f}.pdf")

        for k in range(trajectory.position_nk2().shape[0]):

            xy = trajectory.position_nk2()[k:k+1]
            t = trajectory.heading_nk1()[k:k+1]
            v = trajectory.speed_nk1()[k:k+1]

            voxel_space_position_nk1_x, voxel_space_position_nk1_y, voxel_space_heading_nk1, voxel_space_speed_nk1 \
                = self.objectives[0].reachability_map.avoid_4d_map.grid_world_to_voxel_world(xy, t, v)

            voxel_space_position_nk1_x = tf.cast(tf.round(voxel_space_position_nk1_x), dtype=tf.int32)
            voxel_space_position_nk1_y = tf.cast(tf.round(voxel_space_position_nk1_y), dtype=tf.int32)
            voxel_space_heading_nk1 = tf.cast(tf.round(voxel_space_heading_nk1), dtype=tf.int32)
            voxel_space_speed_nk1 = tf.cast(tf.round(voxel_space_speed_nk1), dtype=tf.int32)

            # if k < 109:
            #     continue

            img_ = np.copy(img)
            img_[:] = -1

            img_[voxel_space_position_nk1_x[0], voxel_space_position_nk1_y[0]] = np.expand_dims(obj[k], -1)
            plt.matshow(img_)
            plt.colorbar(location='bottom',shrink=0.7)
            plt.title(str(k) + 'start @ '+ 'heading: ' + str(np.rad2deg(t.numpy()[0][0]))+'speed: '+str(v.numpy()[0][0])+'\n'+
                      'end @ '+ 'heading: ' + str(np.rad2deg(t.numpy()[0][-1]))+'speed: '+str(v.numpy()[0][-1]), fontdict={'fontsize': 8} )

            # plt.title(str(k))
            fig = plt.gcf()
            pp.savefig(fig)
            # plt.show()

            img_ = np.copy(img)
            img_[voxel_space_position_nk1_x[0], voxel_space_position_nk1_y[0]] = np.expand_dims(obj[k], -1) + 15
            plt.matshow(img_)
            plt.colorbar(location='bottom',shrink=0.7)
            if np.any(obj[k] < 0):
                unsafe+=1
            plt.title(str(k) + ' unsafe:'+ str(np.any(obj[k] < 0)))
            fig1 = plt.gcf()
            pp.savefig(fig1)
            plt.close('all')
            # plt.show()

        pp.close()

        print ("unsafe: ", unsafe)            # print(k, 'unsafe:', np.any(obj[k] < 0))


        return values

    def add_objective(self, objective):
        """
        Add an objective to the objective function.
        """
        self.objectives.append(objective)

    def evaluate_function_by_objective(self, trajectory):
        """
        Evaluate each objective corresponding to a system trajectory.
        """
        avoid_values_by_tag=[]

        for objective in self.objectives:
            if isinstance(objective, Avoid4d):
                avoid_values_by_tag.append([objective.tag, objective.evaluate_avoid(trajectory)])

        return avoid_values_by_tag

    def evaluate_function(self, trajectory):
        """
        Evaluate the entire objective function corresponding to a system trajectory.
        """
        vz = self.vz_values(trajectory)
        objective_values_by_tag = self.evaluate_function_by_objective(trajectory)
        objective_function_values = 0.
        labels=[]

        reachability_cost = False

        # Judge if we are using reachability cost
        for tag, objective_values in objective_values_by_tag:
            if tag == 'avoid_4d':
            # if tag == 'avoid_4d':
                reachability_cost = True

        # No freezing cost!
        if reachability_cost:
            for tag, objective_values in objective_values_by_tag:
                if tag == 'avoid_4d':
                # if tag == 'avoid_4d':
                    objective_function_values += self._reduce_objective_values(trajectory, objective_values)
                    #
                    # labels = self._reduce_objective_values(trajectory, objective_values)
        else:
            for tag, objective_values in objective_values_by_tag:
                objective_function_values += self._reduce_objective_values(trajectory, objective_values)

        if reachability_cost:
            for tag, objective_values in objective_values_by_tag:
                if tag == 'avoid_4d':
                    obj = objective_values
                    # label_11 = (np.min(np.sign(np.array(obj)), axis=1))
                    label_11 = (np.min(np.sign(np.array(obj)), axis=1))
                    label_11[np.where(label_11 <= 0)[0]] = -1  # -1 and 1
                    # label_01 = (np.min((obj.numpy()), axis=1))== 100
                    # label_11 = 2 * label_01 - 1
                    # labels = np.reshape(np.array(label_11), (1, -1))
                    # labels = np.expand_dims(np.reshape(np.array(label_11), (-1, 1)), axis=0)
                    labels = np.reshape(np.array(label_11), (-1, 1))

        return objective_function_values,objective_values,  labels

    def _reduce_objective_values(self, trajectory, objective_values):
        """Reduce objective_values according to
        self.params.obj_type."""
        if self.params.obj_type == 'mean':
            res = tf.reduce_mean(objective_values, axis=1)
        elif self.params.obj_type == 'valid_mean':
            valid_mask_nk = trajectory.valid_mask_nk
            obj_sum = tf.reduce_sum(objective_values * valid_mask_nk, axis=1)
            res = obj_sum / trajectory.valid_horizons_n1[:, 0]
            # valid_mask_nk = trajectory.valid_mask_nk
        elif self.params.obj_type == 'itself':
            res = objective_values
        else:
            assert (False)
        return res



    def _freeze_cost_v1(self, objective_values, objective_distance_to_goal):

        obj_val_np = objective_values.numpy()
        goal_dist_np = objective_distance_to_goal.numpy()

        size_val = obj_val_np.shape

        if np.min(goal_dist_np) <= goal_mask:
            for i in range(size_val[0]):
                if np.min(goal_dist_np[i, :]) < goal_mask:
                    goal_enter_index = np.min(np.where(goal_dist_np[i, :] < goal_mask))
                    if goal_enter_index < size_val[0]:
                        min_cost = np.min(obj_val_np[i, goal_enter_index:size_val[0]])
                        min_cost_index = np.argmin(obj_val_np[i, goal_enter_index:size_val[0]])
                        obj_val_np[i, (goal_enter_index+min_cost_index):] = min_cost

        return tf.constant(obj_val_np, dtype=tf.float32)

    def _freeze_cost_v2(self, objective_values_by_tag, objective_values, objective_distance_to_goal):

        obj_val_np = objective_values.numpy()
        goal_dist_np = objective_distance_to_goal.numpy()

        size_val = obj_val_np.shape

        # First, find the index to start freeze from only reach_avoid_cost
        for tag, objective_values in objective_values_by_tag:
            if tag == 'reach_avoid_4d':
                reach_avoid_4d_values = objective_values

        # # Freeze the cost when it collides with the obstacles
        # for i in range(size_val[0]):
        #     if np.max(obj_val_np[i, :]) >= 300:
        #         obstacle_enter_index = np.min(np.where(obj_val_np[i, :] >= 300))
        #         if obstacle_enter_index < size_val[0]:
        #             # Freeze the cost when hitting obstacles
        #             max_cost = np.max(obj_val_np[i, obstacle_enter_index:size_val[0]])
        #             obj_val_np[i, obstacle_enter_index:] = max_cost

        # Freeze the cost to previous cost when it reach the goal
        if np.min(goal_dist_np) <= goal_mask:
            for i in range(size_val[0]):
                if np.min(goal_dist_np[i, :]) < goal_mask:
                    goal_enter_index = np.min(np.where(goal_dist_np[i, :] < goal_mask))
                    if goal_enter_index < size_val[0] and np.max(reach_avoid_4d_values[i, :goal_enter_index + 1]) < 100:
                        # set all cost to be reach aovid cost
                        # min_cost = np.min(reach_avoid_4d_values[i, goal_enter_index:size_val[0]])
                        # Set all cost to be the previous cost
                        min_cost = np.min(obj_val_np[i, goal_enter_index:size_val[0]])
                        min_cost_index = np.argmin(reach_avoid_4d_values[i, goal_enter_index:size_val[0]])
                        obj_val_np[i, (goal_enter_index+min_cost_index):] = min_cost

        # # Freeze the cost to only reach_avoid cost when it reach the goal
        # if np.min(goal_dist_np) <= goal_mask:
        #     for i in range(size_val[0]):
        #         if np.min(goal_dist_np[i, :]) < goal_mask:
        #             goal_enter_index = np.min(np.where(goal_dist_np[i, :] < goal_mask))
        #             if goal_enter_index < size_val[0] and np.max(reach_avoid_4d_values[i, :goal_enter_index + 1]) < 100:
        #                 # set all cost to be reach aovid cost
        #                 # min_cost = np.min(reach_avoid_4d_values[i, goal_enter_index:size_val[0]])
        #                 # Set all cost to be the previous cost
        #                 min_cost = np.min(reach_avoid_4d_values[i, goal_enter_index:size_val[0]])
        #                 min_cost_index = np.argmin(reach_avoid_4d_values[i, goal_enter_index:size_val[0]])
        #                 obj_val_np[i, (goal_enter_index+min_cost_index):] = min_cost

        # if np.min(goal_dist_np) <= goal_mask:
        #     for i in range(size_val[0]):
        #         if np.min(goal_dist_np[i, :]) < goal_mask and np.max(reach_avoid_4d_values[i, :goal_enter_index + 1]) < 100:
        #             goal_enter_index = np.min(np.where(goal_dist_np[i, :] < goal_mask))
        #             if goal_enter_index < size_val[0]:
        #                 min_cost = np.min(obj_val_np[i, goal_enter_index:size_val[0]])
        #                 min_cost_index = np.argmin(obj_val_np[i, goal_enter_index:size_val[0]])
        #                 obj_val_np[i, (goal_enter_index+min_cost_index):] = min_cost

        return tf.constant(obj_val_np, dtype=tf.float32)

    def _freeze_cost(self, objective_values):

        obj_val_np = objective_values.numpy()

        size_val = obj_val_np.shape

        # Freeze the cost when get to the goal
        for i in range(size_val[0]):
            min_val = np.amin(obj_val_np[i, :])
            min_index = np.argmin(obj_val_np[i, :])

            if min_val <= 0 and min_index < size_val[1]:
                obj_val_np[i, min_index:] = 0

        return tf.constant(obj_val_np, dtype=tf.float32)

