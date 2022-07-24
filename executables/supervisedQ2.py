import tensorflow as tf

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

class SupervisedQ2(object):
    # def __init__(self, params, dt, n, k, position_nk2=None, speed_nk1=None, acceleration_nk1=None, heading_nk1=None,
    # angular_speed_nk1=None, angular_acceleration_nk1=None,
    # dtype=tf.float32, variable=True, direct_init=False,
    # valid_horizons_n1=None,
    # track_trajectory_acceleration=True)
    # super().__init__(params=params)
    # self.p = params
    def __init__(self):
        self.Q=[]
        self.x = []
        self.y = []
        self.v = []
        self.theta = []

    def run(self):

        with tf.device('/cpu:0'):
            simulator,p = self.get_simulator()

            #num = 10000
            num = 100
            for action in range(num):
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
                simulator.reset()
                self.acceleration_nk1 = [0, 0.2,-0.4]

                self.angular_speed_nk1 = [0, 5.5,- 1.1]
                self.TTCs=[]

                self.actions = np.array((self.acceleration_nk1, self.angular_speed_nk1)).T

                count=0
                dt = 1#0.05
                for i in self.actions:
                    #position_nk1_next1: List[Any]
                    position_nk1_next0, position_nk1_next1, self.speed_nk1_next, self.heading_nk1_next = self.carmodel(simulator.start_config.trainable_variables, i, dt)
                    self.x.append(position_nk1_next0)
                    self.y.append(position_nk1_next1)
                    self.v.append(self.speed_nk1_next)
                    self.theta.append(self.heading_nk1_next)

                    position_nk1_next=np.array([np.array(position_nk1_next1)[:, :, :, :, 0], np.array(position_nk1_next0)[:, :, :, :, 0]])

                    position_nk1_next_reshape = np.reshape(position_nk1_next,(1,1,2)) #y,x
                    # position_nk1_next=[]
                    # position_nk1_next[:,:,0],position_nk1_next[:,:,1] =position_nk1_next1, position_nk1_next0

                    self.TTCs[count]=simulator.reachability_map.avoid_4d_map.compute_voxel_function(position_nk1_next_reshape,self.heading_nk1_next, self.speed_nk1_next,invalid_value=100)

                    self.discount = 0.90

                    self.Q. append(dt + (1 - pow(self.discount, self.TTCs + 1)) / (1 - self.discount))  # dt from TRAJECTORY (action)

                    count=+1

    def carmodel(self, start_config, i,dt):
        #self._position_nk2, self._speed_nk1,
        #self._acceleration_nk1, self._heading_nk1,
        #self._angular_speed_nk1, self._angular_acceleration_nk1

        self.acceleration_nk1=i[0]
        self.angular_speed_nk1 = i[1]


        self.speed_nk1_next = start_config[1] + self.acceleration_nk1 * dt
        self.heading_nk1_next=start_config[3]+self.angular_speed_nk1*dt

        #x0 = start_config[0][0]
        #y0 = start_config[0][1]

        x0=np.array(start_config[0].read_value())[0,0][0]
        y0 = np.array(start_config[0].read_value())[0, 0][1]
        position_nk1_next0=[]
        position_nk1_next1 = []
        position_nk1_next0 = start_config[1].numpy()*np.cos(start_config[3].numpy())*dt+x0,
        #position_nk1_next0 = self.speed_nk1_next.numpy() * np.cos(self.heading_nk1_next) * dt + x0,
        position_nk1_next0=list(position_nk1_next0)
        position_nk1_next1 = start_config[1].numpy()*np.sin(start_config[3].numpy())*dt+y0,
        #position_nk1_next1 = self.speed_nk1_next.numpy() * np.sin(self.heading_nk1_next) * dt + y0,
        position_nk1_next1 = list(position_nk1_next1)

        #start_config[0],start_config[1],start_config[2],start_config[3], start_config[4]=[position_nk1_next0,position_nk1_next1],self.speed_nk1_next,self.acceleration_nk1, self.heading_nk1_next, self.angular_speed_nk1

        # x0 = position_nk1_next0
        # y0 = position_nk1_next1
        # position_nk1_next0 = []
        # position_nk1_next1 = []
        # position_nk1_next0 = self.speed_nk1_next.numpy() * np.cos(self.heading_nk1_next.numpy()) * dt + x0,
        # # position_nk1_next0 = self.speed_nk1_next.numpy() * np.cos(self.heading_nk1_next) * dt + x0,
        # position_nk1_next0 = list(position_nk1_next0)
        # position_nk1_next1 = self.speed_nk1_next.numpy() * np.sin(self.heading_nk1_next.numpy()) * dt + y0,
        # # position_nk1_next1 = self.speed_nk1_next.numpy() * np.sin(self.heading_nk1_next) * dt + y0,
        # position_nk1_next1 = list(position_nk1_next1)
        #
        # return position_nk1_next0,position_nk1_next1, self.speed_nk1_next,self.heading_nk1_next
        #start_config[0], start_config[1], start_config[2], start_config[3], start_config[4] = [position_nk1_next0,
                                                                                               #position_nk1_next1], self.speed_nk1_next, self.acceleration_nk1, self.heading_nk1_next, self.angular_speed_nk1

        x0 = position_nk1_next0
        y0 = position_nk1_next1
        position_nk1_next0 = []
        position_nk1_next1 = []
        position_nk1_next0 = start_config[1].numpy() * np.cos(start_config[3].numpy()) * dt + x0,
        # position_nk1_next0 = self.speed_nk1_next.numpy() * np.cos(self.heading_nk1_next) * dt + x0,
        position_nk1_next0 = list(position_nk1_next0)
        position_nk1_next1 = start_config[1].numpy() * np.sin(start_config[3].numpy()) * dt + y0,
        # position_nk1_next1 = self.speed_nk1_next.numpy() * np.sin(self.heading_nk1_next) * dt + y0,
        position_nk1_next1 = list(position_nk1_next1)
        start_config[0], start_config[1], start_config[2], start_config[3], start_config[4] = [position_nk1_next0,
                                                                                               position_nk1_next1], self.speed_nk1_next, self.acceleration_nk1, self.heading_nk1_next, self.angular_speed_nk1

        return position_nk1_next0,position_nk1_next1, self.speed_nk1_next,self.heading_nk1_next

    if __name__ == '__main__':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        SupervisedQ2().run()
