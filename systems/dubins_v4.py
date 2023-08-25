from systems.dubins_4d import Dubins4D
import tensorflow as tf


class DubinsV4(Dubins4D):
    """ A discrete time 4 dimensional dubins car with linear clipping saturation
    functions on linear or angular velocity.
    """
    name = 'dubins_v4'

    def __init__(self, dt, params):
        super().__init__(dt)
        self.v_bounds = params.v_bounds
        self.w_bounds = params.w_bounds
        self.a_bounds = params.a_bounds


    def _saturate_linear_velocity(self, vtilde_nk):
        """ Linear clipping saturation function for linear velocity"""
        v_nk = tf.clip_by_value(vtilde_nk, self.v_bounds[0], self.v_bounds[1])
        return v_nk

    def _saturate_angular_velocity(self, wtilde_nk):
        """ Linear clipping saturation function for angular velocity"""
        w_nk = tf.clip_by_value(wtilde_nk, self.w_bounds[0], self.w_bounds[1])
        return w_nk

    def _saturate_acceleration(self, atilde_nk):
        """ Linear clipping saturation function for angular velocity"""
        a_nk = tf.clip_by_value(atilde_nk, self.a_bounds[0], self.a_bounds[1])
        # a_nk = tf.clip_by_value(atilde_nk, max(self.a_bounds[0],-1*tf.squeeze(vtilde_nk)/dt), min(self.a_bounds[1],tf.squeeze((self.v_bounds[1]-vtilde_nk)/dt)))
        return a_nk
    
    def _saturate_linear_velocity_prime(self, vtilde_nk):
        """ Time derivative of linear clipping saturation function"""
        less_than_idx = (vtilde_nk < self.v_bounds[0])
        greater_than_idx = (vtilde_nk > self.v_bounds[1])
        zero_idxs = tf.logical_or(less_than_idx, greater_than_idx)
        res = tf.cast(tf.logical_not(zero_idxs), vtilde_nk.dtype)
        return res

    def _saturate_angular_velocity_prime(self, wtilde_nk):
        """ Time derivative of linear clipping saturation function"""
        less_than_idx = (wtilde_nk < self.w_bounds[0])
        greater_than_idx = (wtilde_nk > self.w_bounds[1])
        zero_idxs = tf.logical_or(less_than_idx, greater_than_idx)
        res = tf.cast(tf.logical_not(zero_idxs), wtilde_nk.dtype)
        return res

    def _saturate_acceleration_prime(self, atilde_nk):
        """ Time derivative of linear clipping saturation function"""
        less_than_idx = (atilde_nk < self.a_bounds[0])
        greater_than_idx = (atilde_nk > self.a_bounds[1])
        zero_idxs = tf.logical_or(less_than_idx, greater_than_idx)
        res = tf.cast(tf.logical_not(zero_idxs), atilde_nk.dtype)
        return res

