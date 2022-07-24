import tensorflow as tf
from matplotlib.pyplot import hold

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

class SupervisedQ(object):
    # def __init__(self, params, dt, n, k, position_nk2=None, speed_nk1=None, acceleration_nk1=None, heading_nk1=None,
    # angular_speed_nk1=None, angular_acceleration_nk1=None,
    # dtype=tf.float32, variable=True, direct_init=False,
    # valid_horizons_n1=None,
    # track_trajectory_acceleration=True)
    # super().__init__(params=params)
    # self.p = params
    def __init__(self):
        self.Q= []
        self.x = []
        self.y = []
        self.v = []
        self.theta = []
        self.TTC = []

    def run(self):

        with tf.device('/cpu:0'):
            simulator,p = self.get_simulator()

            #num = 10000
            num_samples = 3
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
                self.acceleration_nk1 = [0,.2,0.3]
                # self.acceleration_nk1=np.linspace(0, 0.4, 5)
                self.angular_speed_nk1 = [0, 0,0]#,-np.pi/8, np.pi/8]
                # self.angular_speed_nk1 = np.linspace(0, 1.1, 12)
                self.TTC=[]
                #
                # self.actions = np.array((self.acceleration_nk1, self.angular_speed_nk1)).T
                # self.actions=[0,0]
                simulator.reset()
                count=0
                # dt = 0.05
                dt = 1
                # TTC0=0
                # TTC0=simulator.reachability_map.avoid_4d_map.compute_voxel_function(simulator.start_config.trainable_variables[0][0][0], simulator.start_config.trainable_variables[3][0], simulator.start_config.trainable_variables[1])
                # print(TTC0)
                start_state_list=[]
                start_state=np.zeros(4)
                start_state_list=[simulator.start_config.position_nk2()[0][0][0], simulator.start_config.position_nk2()[0][0][1],simulator.start_config.heading_nk1()[0][0][0], simulator.start_config.speed_nk1()[0][0][0]]
                start_state=np.array(start_state_list)
                # actions_waypoint=[[0.25,0,0,0.05],[0.3,0.1,0.3,-0.05],[0.227,0.2225,0,0.2],[0.113,0.15,0,0.05],[0.15,-0.1,-0.2,0.05]]
                # actions_waypoint = [[0.25, 0, 0], [0.3, 0.1, 0.3], [0.227, 0.2225, 0],
                #                     [0.113, 0.15, 0], [0.15, -0.1, -0.2]]

                actions_waypoint = [[.5, 0, 0], [0.25, -0.5, -1.1], [0.25, 0.5, 1.1], [0.3, 0.4, 0.9],
                                    [0.3, -0.4, -0.9],
                                    [0.25, 0.4, 1], [0.25, -0.4, -1], [0.4, 0.2, 0.3], [0.4, -0.2, -0.3],
                                    [0.1, 0.5, 0.5], [0.1, -0.5, -0.5],
                                    [0.4, 0.4, 0.8], [0.4, -0.4, -0.8], [0.45, 0.2, 0.9], [0.35, -0.3, -1],
                                    [0.35, 0.3, 1], [0.45, -0.2, -0.9], [0.45, -0.1, -0.2], [0.45, 0.1, 0.2],
                                    [0.1, 0.1, 0.4], [0.1, -0.1, -0.4], [0.15, -0.1, -0.6], [0.15, 0.1, 0.6],
                                    [0.15, 0.35, -0.3], [0.15, -0.35, -0.3], [0.25, -0.25, 0.1], [0.25, 0.25, 0.1]]
                # actions_waypoint =[0.227,0.2225,0,0.2]
                v0=[0.2,0.5]
                vf = [0.2, 0.5]
                import os
                import matplotlib.pyplot as plt
                import control as ct
                import control.flatsys as fs

                # Function to take states, inputs and return the flat flag
                def vehicle_flat_forward(x, u):
                    # Get the parameter values

                    # Create a list of arrays to store the flat output and its derivatives
                    zflag = [np.zeros(3), np.zeros(3)]

                    # Flat output is the x, y position of the rear wheels
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

                vehicle_flat = fs.FlatSystem(forward=vehicle_flat_forward, reverse=vehicle_flat_reverse, inputs=2,
                                             states=4)
                x=[]
                x1 = np.zeros((1,4,100))
                x1 = np.zeros(( 100, 4,len(actions_waypoint)))
                state_traj = []
                # x0 = [0., 0,0, 0.25]
                # x0 = [0., 0, 0, 0.35]
                u0 = [0, 0.]
                # xf = [0.2, 0.12, 1,0.25]
                # xf = [0.4, 0.2, 0.2, 0.4]
                x0 = start_state
                xf = start_state+actions_waypoint
                uf = [0, 0]
                Tf = 1
                t = np.linspace(0, Tf, 20) #dt=0.05

                # Define a set of basis functions to use for the trajectories
                poly = fs.PolyFamily(6)
                r = SBPDRenderer.get_renderer(p.simulator_params.obstacle_map_params.renderer_params)
                dx_cm, traversible = r.get_config()
                dx_m = dx_cm / 100.
                # print(type(simulator.start_config.trainable_variables[0]))
                # camera_pos_13 = self.heading_nk1_next[0]
                # camera_grid_world_pos_12 = position_nk1_next[0] / dx_m
                # rgb_image_1mk3 = r._get_rgb_image(camera_grid_world_pos_12, camera_pos_13)

                camera_pos_13 = simulator.start_config.heading_nk1()[0]
                camera_grid_world_pos_12 = simulator.start_config.position_nk2()[0] / dx_m
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

                # x1 = np.zeros(len(actions_waypoint),None, None)

                # Find a trajectory between the initial condition and the final condition
                for i in range (len(actions_waypoint)):
                    traj = fs.point_to_point(vehicle_flat, Tf, x0, u0, xf[i], uf,
                                         basis=poly)  # , constraints =[( -1.1, 1.1) ,(  -0.4, 0.4)])

                # Create the trajectory

                    point, u = traj.eval(t) # i
                    # x1[i-1,:,:]=x

                    ax2.plot((point[0]-np.ones(20)*point[0][0])/0.05+robot[0], (point[1]-np.ones(20)*point[1][0])/0.05+robot[1], ls='dotted', linewidth=2, color='red')




                    ttc=[]
                    while -1.4<=u[0]<=1.4 and -.4<=u[0]<=.4 and -.7<=point[3]<=.7:
                        for j in range (20): #trajectror

                            state_traj=[point[0][j],point[1][j],point[2][j],point[3][j]]# j
                            # ax.plot(start_2[0], start_2[1], 'bo', markersize=14)


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
                            from scipy.interpolate import RegularGridInterpolator
                            my_interpolating_function = RegularGridInterpolator((x, y, theta, v), data)

                            pts=state_traj
                            # camera_pos_13 = simulator.start_config.heading_nk1()[0]
                            # camera_grid_world_pos_12 = simulator.start_config.position_nk2()[0] / dx_m
                            # rgb_image_1mk3 = r._get_rgb_image(camera_grid_world_pos_12, camera_pos_13)
                            # pts = np.array([position_nk1_next0, position_nk1_next1, self.speed_nk1_next, self.heading_nk1_next])
                            ttc.append(my_interpolating_function(pts))
                            #
                            # self.TTC.append(simulator.reachability_map.avoid_4d_map.compute_voxel_function(position_nk1_next,self.heading_nk1_next,self.speed_nk1_next))#update with new file
                            # ttc.append(
                            #     simulator.reachability_map.avoid_4d_map.compute_voxel_function([point[0][j],point[1][j]],point[3][j],point[2][j]))
                            self.discount = 0.90
                            self.gamma = 1.0
                            # self.theta = 1e-10
                    self.TTC = min(ttc)
                    self.Q.append([self.gamma * ( #
                                    dt + self.discount * (1 - pow(self.discount, self.TTC + np.ones(len(self.TTC)))) / (
                                        1 - self.discount))])  # dt from TRAJECTORY (action) #fix this

                plt.show()

                d = {'image': rgb_image_1mk3, 'action': actions_waypoint, 'q values': self.Q}
                                    # save to file
                import pickle
                with open('file.pkl', 'wb') as f:
                    pickle.dump(d, f)




###
#                 for i in self.actions:
#                     #position_nk1_next1: List[Any]
#                     position_nk1_next0, position_nk1_next1, self.speed_nk1_next, self.heading_nk1_next = self.carmodel(simulator, i, dt)
#                     self.x.append(position_nk1_next0)
#                     self.y.append(position_nk1_next1)
#                     self.v.append(self.speed_nk1_next)
#                     self.theta.append(self.heading_nk1_next)
#
#                     # position_nk1_next=np.array([np.array(position_nk1_next1)[:, :, :, :, 0], np.array(position_nk1_next0)[:, :, :, :, 0]])
#                     position_nk1_next=np.asarray([[[position_nk1_next0[0][0][0],position_nk1_next1[0][0][0]]]])
#                     # position_nk1_next_reshape = np.reshape(position_nk1_next,(1,1,2)) #y,x
# #                    self.states.append(position_nk1_next_reshape)
#                     # self.states.append(self.State.nxtPosition(action))
#                     # position_nk1_next=[]
#                     # position_nk1_next[:,:,0],position_nk1_next[:,:,1] =position_nk1_next1, position_nk1_next0
#                     pts=np.array([position_nk1_next0,position_nk1_next1, self.speed_nk1_next, self.heading_nk1_next])
#                     self.TTC.append(my_interpolating_function(pts))
#                     # self.TTC.append(simulator.reachability_map.avoid_4d_map.compute_voxel_function(position_nk1_next,self.heading_nk1_next,self.speed_nk1_next))#update with new file
#
#                     self.discount = 0.90
#                     self.gamma = 1.0
#                     # self.theta = 1e-10
#
#                     self.Q. append([self.gamma*(dt +self.discount  *(1 - pow(self.discount, self.TTC + np.ones(len(self.TTC)))) / (1 - self.discount))])  # dt from TRAJECTORY (action) #fix this
#                     r = SBPDRenderer.get_renderer(p.simulator_params.obstacle_map_params.renderer_params)
#                     dx_cm, traversible = r.get_config()
#                     dx_m = dx_cm / 100.
#                     # print(type(simulator.start_config.trainable_variables[0]))
#                     camera_pos_13 = self.heading_nk1_next[0]
#                     camera_grid_world_pos_12 = position_nk1_next[0] / dx_m
#                     rgb_image_1mk3 = r._get_rgb_image(camera_grid_world_pos_12, camera_pos_13)
#                     import matplotlib.pyplot as plt
#                     fig = plt.figure(figsize=(30, 10))
#                     ax = fig.add_subplot(1, 3, 2)
#                     ax.imshow(rgb_image_1mk3[0].astype(np.uint8))
#                     ax.set_xticks([])
#                     ax.set_yticks([])
#                     ax.set_title('next')
#                     plt.show()
#                     d = {'image': rgb_image_1mk3, 'action': actions_waypoint, 'q values': self.Q}
#                     # save to file
#                     import pickle
#                     with open('file.pkl', 'wb') as f:
#                         pickle.dump(d, f)

                    # count=+1
                    # How much our value function changed (across any states)
                #     delta = max(delta, np.abs(v - V[s]))
                #     V[s] = v
                #     # Stop evaluating once our value function change is below a threshold
                # if delta < theta:
                #     break
                # return np.array(V)

#                    self._r = SBPDRenderer.get_renderer(p.simulator_params.obstacle_map_params.renderer_params)
#                 r = SBPDRenderer.get_renderer(p.simulator_params.obstacle_map_params.renderer_params)
#                 dx_cm , traversible = r.get_config()
#                 dx_m = dx_cm / 100.
#                 # print(type(simulator.start_config.trainable_variables[0]))
#                 camera_pos_13=simulator.start_config.trainable_variables[3][0]
#                 camera_grid_world_pos_12 = simulator.start_config.trainable_variables[0][0]/dx_m
#                 rgb_image_1mk3 = r._get_rgb_image(camera_grid_world_pos_12, camera_pos_13)
#
#                 import matplotlib.pyplot as plt
#                 fig = plt.figure(figsize=(30, 10))
#                 ax = fig.add_subplot(1, 3, 2)
#                 ax.imshow(rgb_image_1mk3[0].astype(np.uint8))
#                 ax.set_xticks([])
#                 ax.set_yticks([])
#                 ax.set_title('initial')
#                 plt.show()
#
#                 d={'image':rgb_image_1mk3, 'state':simulator.start_config, 'q values': self.Q}
#                 #save to file
#                 import pickle
#                 with open('file.pkl', 'wb') as f:
#                     pickle.dump(d,f)
###
                # # Plot the 5x5 meter occupancy grid centered around the camera
                # ax = fig.add_subplot(1, 3, 1)
                # ax.imshow(traversible, extent=extent, cmap='gray',
                #           vmin=-.5, vmax=1.5, origin='lower')
                #
                # # Plot the camera
                # ax.plot(camera_pos_13[0, 0], camera_pos_13[0, 1], 'bo', markersize=10, label='Camera')
                # ax.quiver(camera_pos_13[0, 0], camera_pos_13[0, 1], np.cos(camera_pos_13[0, 2]),
                #           np.sin(camera_pos_13[0, 2]))
                #
                #
                # ax.legend()
                # ax.set_xlim([camera_pos_13[0, 0] - 5., camera_pos_13[0, 0] + 5.])
                # ax.set_ylim([camera_pos_13[0, 1] - 5., camera_pos_13[0, 1] + 5.])
                # ax.set_xticks([])
                # ax.set_yticks([])
                # ax.set_title('Topview')
                #self.top=self._r._get_topview(simulator.start_config.trainable_variables[0].numpy().transpose(),simulator.start_config.trainable_variables[3].numpy())
                    # starts_n2=np.reshape(np.array(simulator.start_config.trainable_variables[0]), (1,2))#.transpose()
# ###
#                     starts_n2=np.array(simulator.start_config.trainable_variables[0])
#                     thetas_n1 = np.array(simulator.start_config.trainable_variables[3])
#                     #thetas_n1=np.reshape(simulator.start_config.trainable_variables[3],(1,1))
#                     self.image = self._get_rgb_image(starts_n2,thetas_n1)
#                     # self.image = self._r.render_images(starts_n2,thetas_n1)
#                     import matplotlib.pyplot as plt
#                     img = np.reshape(self.image, (224, 224, 3))
#                     imgplot = plt.imshow(img)
#                     plt.show()
# ###
                    # img=np.reshape(self.image, (224, 224, 3))
                    # lum_img = img[:, :, 0]
                    # plt.imshow(lum_img)
                    # imgplot = plt.imshow(img)

                # self.best_index = np.argmax(self.Q)
                # self.best_action = self.actions[self.best_index]
                #
                # ##
                # # self.Dict = {(self._image:  "self._Q"}
                # # self.render1 = SBPDRenderer.get_renderer(p.renderer_params)
                #
                #
                #
                #
                # self.input = self.image
                # # self.input = np.concatenate(self._image.shape[::-1], actions)
                # self.output = self.Q
                #
                # # train
                #
                # self.arch, self.is_batchnorm_training = self._make_architecture()  # for me it is not cnn
                # # test
                #
                # self._preds = self._predict_nn_output(self, data, is_training=None)
                # self._error = self._Q - self._preds
                #

                #mymethod()
                end_time = time.time()
                print("episode", action, "takes", end_time-start_time)
    """
    Initialize a map for Stanford Building Parser Dataset (SBPD)
    """
    #def __init__(self, params):

        #self.params = params.simulator.parse_params(params)
        #self.rng = np.random.RandomState(params.seed)  # Sample some random states, used for initalizing map
        #self.obstacle_map = self._init_obstacle_map(self.rng)
        #self.obj_fn = self._init_obj_fn()
        #self.planner = self._init_planner()
        #self.system_dynamics = self._init_system_dynamics()


    #def mymethod(self):

        #self.params = params.simulator.parse_params(params)
        #self.rng = np.random.RandomState(params.seed)  # Sample some random states, used for initalizing map
        #self.obstacle_map = self._init_obstacle_map(self.rng)
        #self.obj_fn = self._init_obj_fn()
        #self.planner = self._init_planner()
        #self.system_dynamics = self._init_system_dynamics()

        # self._r = SBPDRenderer.get_renderer(self.p.renderer_params)  # Intialize map here
    def reset1(self, seed=-1):
        """Reset the simulator. Optionally takes a seed to reset
        the simulator's random state."""
        if seed != -1:
        #if seed == -1:
            self.rng.seed(seed)

        # Note: Obstacle map must be reset independently of the fmm map.
        # Sampling start and goal may depend on the updated state of the
        # obstacle map. Updating the fmm map depends on the newly sampled goal.

        #self._reset_obstacle_map(self.rng)  # Do nothing here

        self._reset_start_configuration(self.rng)  # Reset self.start_config

        #self.start_config=self._reset_start_configuration(self.rng)  # Reset self.start_config

        #self._image = self._get_observation(self, config=self.start_config, pos_n3=None, **kwargs)
        #self.p = params
        self.render1=SBPDRenderer.get_renderer(self.p.renderer_params)
        self.image=self.render1.SBPDRenderer.render_images(self.start_config.position_nk2, self.start_config.heading_nk1, crop_size=None)
        #self.image=self.render1.SBPDRenderer.render_images(starts_n2, thetas_n1, crop_size=None))


        #self._r = SBPDRenderer.get_renderer(self.p.renderer_params) # Intialize map here
        #self.acceleration_nk1 = np.random.uniform(-0.4, 0.4, 1)
        #self.angular_speed_nk1= np.random.uniform(-1.1, 1.1, 1)
        #a=0.5
        #b=0.5
        #loc=0.5
        #scale_a=0.5/0.4
        #scale_w=0.5/1.1
        #y = (x - loc) / scale ,  beta.pdf(x, a, b, loc, scale)

        #self.acceleration_nk1= beta.rvs(x, a, b, loc, scale_a)
        #self.angular_speed_nk1= beta.rvs(x, a, b, loc, scale_w)

        self.acceleration_nk1=[-0.4, 0, 0.4]
        self.angular_speed_nk1=[-1.1, 0, 1.1]

        self.actions = np.array((self.acceleration_nk1, self.angular_speed_nk1)).T
        print(self.actions)
        # self._ttr_avoid_4d = self._compute_avoid_4d_map_LFsweep() # can I have it as a  output?

        for i in self.actions:

            self.next_config=self.carmodel(self.start_config,i,dt)

            self.TTCs = self.ttr_avoid_4d(self.next_config)


            self.discount = 0.99

            self.dt=0.05


            self._Q= self.dt + (1 - pow(self.discount, self.TTCs + 1)) / (1 - self.discount)  # dt from TRAJECTORY (i)
            scipy.interpolate.LinearNDInterpolator(self.next_config, self._Q, fill_value=np.nan, rescale=False)

        self.best_index=np.argmax(self._Q)
        self.best_action=self.actions(self.best_index)

        ##
        #self.Dict = {(self._image:  "self._Q"}
        self.input=self.image(start_config)
        #self.input = np.concatenate(self._image.shape[::-1], actions)
        self.output = self._Q

        # train

        self.arch, self.is_batchnorm_training = self._make_architecture()  # for me it is not cnn
        # test

        self._preds = self._predict_nn_output(self, data, is_training=None)
        self._error = self._Q - self._preds

    def get_simulator(self):

        parser = argparse.ArgumentParser(description='Process the command line inputs')
        parser.add_argument("-p", "--params", required=True, help='the path to the parameter file')
        parser.add_argument("-d", "--device", type=int, default=1, help='the device to run the training/test on')
        args = parser.parse_args()

        p = self.create_params(args.params)

        p.simulator_params = p.data_creation.simulator_params # ! change param to use simulator_2 and different branch in github
        p.simulator_params.simulator.parse_params(p.simulator_params)

        simulator = p.simulator_params.simulator(p.simulator_params)

        return simulator , p

    def create_params(self, param_file):
        """
        Create the parameters given the path of the parameter file.
        """
        # Execute this if python > 3.4
        try:
            spec = importlib.util.spec_from_file_location('parameter_loader', param_file)
            foo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(foo)
        except AttributeError:
            # Execute this if python = 2.7 (i.e. when running on real robot with ROS)
            module_name = param_file.replace('/', '.').replace('.py', '')
            foo = importlib.import_module(module_name)
        return foo.create_params()




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

        # The system dynamics may need the current starting position for
        # coordinate transforms (i.e. realistic simulation)
        self.system_dynamics.reset_start_state(self.start_config)

    def carmodel(self, simulator, i,dt):
        #self._position_nk2, self._speed_nk1,
        #self._acceleration_nk1, self._heading_nk1,
        #self._angular_speed_nk1, self._angular_acceleration_nk1

        # x(t + 1) = x(t) + v(t) * cos(theta_t) * delta_t
        # y(t + 1) = y(t) + v(t) * sin(theta_t) * delta_t
        # theta(t + 1) = theta(t) + w(t) * delta_t
        # v(t + 1) = saturate_linear_velocity(a(t) * dt + v(t))

        self.acceleration_nk1=i[0]
        self.angular_speed_nk1 = i[1]


        self.speed_nk1_next = simulator.start_config.trainable_variables[1] + self.acceleration_nk1 * dt
        self.heading_nk1_next=simulator.start_config.trainable_variables[3]+self.angular_speed_nk1*dt

        # x0=np.array(start_config[0].read_value())[0,0][0]
        x0=simulator.start_config.trainable_variables[0][0][0][0]
        # y0 = np.array(start_config[0].read_value())[0, 0][1]
        y0=simulator.start_config.trainable_variables[0][0][0][1]
        position_nk1_next0=[]
        position_nk1_next1 = []
        # position_nk1_next0 = start_config[1].numpy()*np.cos(start_config[3].numpy())*dt+x0,
        position_nk1_next0=simulator.start_config.trainable_variables[1]*np.cos(simulator.start_config.trainable_variables[3][0][0])*dt+x0
        #position_nk1_next0 = self.speed_nk1_next.numpy() * np.cos(self.heading_nk1_next) * dt + x0,
        # position_nk1_next0=list(position_nk1_next0)
        position_nk1_next1 = simulator.start_config.trainable_variables[1] * np.sin(simulator.start_config.trainable_variables[3][0][0]) * dt + y0
        # position_nk1_next1 = start_config[1].numpy()*np.sin(start_config[3].numpy())*dt+y0,
        #position_nk1_next1 = self.speed_nk1_next.numpy() * np.sin(self.heading_nk1_next) * dt + y0,
        # position_nk1_next1 = list(position_nk1_next1)

        x0 = position_nk1_next0
        y0 = position_nk1_next1
        # position_nk1_next0 = []
        # position_nk1_next1 = []
        # position_nk1_next0 = start_config[1].numpy() * np.cos(start_config[3].numpy()) * dt + x0,
        # # position_nk1_next0 = self.speed_nk1_next.numpy() * np.cos(self.heading_nk1_next) * dt + x0,
        # position_nk1_next0 = list(position_nk1_next0)
        # position_nk1_next1 = start_config[1].numpy() * np.sin(start_config[3].numpy()) * dt + y0,
        # # position_nk1_next1 = self.speed_nk1_next.numpy() * np.sin(self.heading_nk1_next) * dt + y0,
        # position_nk1_next1 = list(position_nk1_next1)
        # start_config[0], start_config[1], start_config[2], start_config[3], start_config[4] = [position_nk1_next0,
        #                                                                                        position_nk1_next1], self.speed_nk1_next, self.acceleration_nk1, self.heading_nk1_next, self.angular_speed_nk1

        return position_nk1_next0,position_nk1_next1, self.speed_nk1_next,self.heading_nk1_next


        def dist_to_nearest_obs(self, pos_nk2):
            with tf.name_scope('dist_to_obs'):
                distance_nk = self.fmm_map.fmm_distance_map.compute_voxel_function(pos_nk2)
                return distance_nk

        def sample_point_112(self, rng, free_xy_map_m2=None):
            """
            Samples a real world x, y point in free space on the map.
            Optionally the user can pass in free_xy_m2 a list of m (x, y)
            points from which to sample.
            """
            if free_xy_map_m2 is None:
                free_xy_map_m2 = self.free_xy_map_m2

            idx = rng.choice(len(free_xy_map_m2))
            pos_112 = free_xy_map_m2[idx][None, None]
            return self._map_to_point(pos_112)

        def create_occupancy_grid_for_map(self, xs_nn=None, ys_nn=None):
            """
            Return the occupancy grid for the SBPD map.
            """
            return self.occupancy_grid_map

        def _get_observation(self, config=None, pos_n3=None, **kwargs):
            """
            Render the robot's observation from system configuration config
            or pos_nk3.
            """
            # One of config and pos_nk3 must be not None
            assert ((config is None) != (pos_n3 is None))

            if config is not None:
                pos_n3 = config.position_and_heading_nk3()[:, 0].numpy()

            if 'occupancy_grid' in self.p.renderer_params.camera_params.modalities:
                occupancy_grid_world_1mk12 = kwargs['occupancy_grid_positions_ego_1mk12']
                _, m, k, _, _ = [x.value for x in occupancy_grid_world_1mk12.shape]
                occupancy_grid_nk2 = tf.reshape(occupancy_grid_world_1mk12, (1, -1, 2))

                # Broadcast the occupancy grid to batch size n if needed
                n = pos_n3.shape[0]
                if n != 1:
                    occupancy_grid_nk2 = tf.broadcast_to(occupancy_grid_nk2, (n,
                                                                              occupancy_grid_nk2.shape[1].value,
                                                                              2))
                occupancy_grid_world_nk2 = DubinsCar.convert_position_and_heading_to_world_coordinates(pos_n3[:, None, :],
                                                                                                       occupancy_grid_nk2.numpy())
                dist_to_nearest_obs_nk2 = self.dist_to_nearest_obs(occupancy_grid_world_nk2)
                dist_to_nearest_obs_nmk1 = tf.reshape(dist_to_nearest_obs_nk2, (n, m, k, 1))
                imgs = 0.5 * (1. - tf.sign(dist_to_nearest_obs_nmk1)).numpy()
            else:
                starts_n2 = self._point_to_map(pos_n3[:, :2])
                thetas_n1 = pos_n3[:, 2:3]
                imgs = self._r.render_images(starts_n2, thetas_n1, **kwargs)
            return imgs

        def render(self, ax, start_config=None):
            p = self.p
            ax.imshow(self.occupancy_grid_map, cmap='gray_r',
                      extent=np.array(self.map_bounds).flatten(order='F'),
                      vmax=1.5, vmin=-.5, origin='lower')

            if start_config is not None:
                start_2 = start_config.position_nk2()[0, 0].numpy()
                delta = p.plotting_grid_steps * p.dx
                ax.set_xlim(start_2[0] - delta, start_2[0] + delta)
                ax.set_ylim(start_2[1] - delta, start_2[1] + delta)

        def render_with_obstacle_margins(self, ax, start_config=None, margin0=.3, margin1=.5):
            p = self.p
            occupancy_grid_masked = np.ma.masked_where(self.occupancy_grid_map == 0,
                                                       self.occupancy_grid_map)
            ax.imshow(occupancy_grid_masked, cmap='Blues_r',
                      extent=np.array(self.map_bounds).flatten(order='F'),
                      origin='lower', vmax=2.0)

            self._render_margin(ax, margin=margin0, alpha=.5)
            self._render_margin(ax, margin=margin1, alpha=.35)

            if start_config is not None:
                start_2 = start_config.position_nk2()[0, 0].numpy()
                delta = p.plotting_grid_steps * p.dx
                ax.set_xlim(start_2[0] - delta, start_2[0] + delta)
                ax.set_ylim(start_2[1] - delta, start_2[1] + delta)

        def _render_margin(self, ax, margin, alpha):
            """
            Render a margin around the occupied space indicating the intensity
            of the obstacle avoidance cost function.
            """
            y_dim, x_dim = self.occupancy_grid_map.shape
            xs = np.linspace(self.map_bounds[0][0], self.map_bounds[1][0], x_dim)
            ys = np.linspace(self.map_bounds[0][1], self.map_bounds[1][1], y_dim)
            xs, ys = np.meshgrid(xs, ys)
            xs = xs.ravel()
            ys = ys.ravel()
            pos_n12 = np.stack([xs, ys], axis=1)[:, None]
            dists_nk = self.dist_to_nearest_obs(pos_n12).numpy()

            margin_mask_n = (dists_nk < margin)[:, 0]
            margin_mask_mn = margin_mask_n.reshape(self.occupancy_grid_map.shape)
            mask = np.logical_and(self.occupancy_grid_map, margin_mask_mn == 0)

            margin_img = np.ma.masked_where(mask, margin_mask_mn)
            ax.imshow(margin_img, cmap='Blues',
                      extent=np.array(self.map_bounds).flatten(order='F'),
                      origin='lower', alpha=alpha, vmax=2.0)


        def __init__(self, dt, n, k, position_nk2=None, speed_nk1=None, acceleration_nk1=None, heading_nk1=None,
                     angular_speed_nk1=None, angular_acceleration_nk1=None,
                     dtype=tf.float32, variable=True, direct_init=False,
                     valid_horizons_n1=None,
                     track_trajectory_acceleration=True):
            assert (k == 1)
            # Don't pass on valid_horizons_n1 as a SystemConfig has no horizon
            super(SystemConfig, self).__init__(dt, n, k, position_nk2, speed_nk1, acceleration_nk1,
                                               heading_nk1, angular_speed_nk1,
                                               angular_acceleration_nk1, dtype=tf.float32,
                                               variable=variable, direct_init=direct_init,
                                               track_trajectory_acceleration=track_trajectory_acceleration)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    SupervisedQ().run()
