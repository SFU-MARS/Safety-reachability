import heterocl as hcl
import numpy as np
import time

# import computeGraphs
""" 4D DUBINS CAR DYNAMICS WIH DISTURBANCE IMPLEMENTATION 
 x_dot = v * cos(theta) + d_x
 y_dot = v * sin(theta) + d_y
 theta_dot = w + d_w
 v_dot = a
 """

class DubinsCar4D_new2:
    def __init__(self, x=[0,0,0,0], uMin = [-0.4, -1.1], uMax = [0.4, 1.1], dTheta_Min = -0.0, \
                 dTheta_Max=0.0, d_xy = 0.0 , uMode="max", dMode="min"):
        self.x = x
        self.uMax = uMax
        self.uMin = uMin
        self.dTheta_Max = dTheta_Max
        self.dTheta_Min = dTheta_Min
        self.d_xy = d_xy
        self.uMode = uMode
        self.dMode = dMode

    def opt_ctrl(self, t, state, spat_deriv):
        """
        :param t: time t
        :param state: tuple of coordinates
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return:
        """
        # System dynamics
        # x_dot     = v * cos(theta) + d_x
        # y_dot     = v * sin(theta) + d_y
        # theta_dot = w + d_w
        # v_dot     = a

        # Graph takes in 4 possible inputs, by default, for now
        opt_a = hcl.scalar(self.uMax[0], "opt_a")
        opt_w = hcl.scalar(self.uMax[1], "opt_w")
        # Just create and pass back, even though they're not used
        in3   = hcl.scalar(0, "in3")
        in4   = hcl.scalar(0, "in4")

        # if self.uMode == "min":
        #     with hcl.if_(spat_deriv[3] > 0):
        #         opt_a[0] = self.uMin[0]
        #     with hcl.if_(spat_deriv[2] > 0):
        #         opt_w[0] = self.uMin[1]
        if self.uMode == "max":
            with hcl.if_(spat_deriv[3] < 0):
                opt_a[0] = self.uMin[0]
            with hcl.if_(spat_deriv[2] < 0):
                opt_w[0] = self.uMin[1]
        # return 3, 4 even if you don't use them
        return (opt_a[0] ,opt_w[0], in3[0], in4[0])

    def opt_dstb(self, t, state, spat_deriv):
        """
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return: a tuple of optimal disturbances
        """

        # System dynamics
        # x_dot     = v * cos(theta) + d_x
        # y_dot     = v * sin(theta) + d_y
        # theta_dot = w + d_w
        # v_dot     = a

        # Graph takes in 4 possible inputs, by default, for now
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        # Just create and pass back, even though they're not used
        d3 = hcl.scalar(self.dTheta_Max, "d3")
        d4 = hcl.scalar(0, "d4")

        spat1_sq = hcl.scalar(0, "spat1_sq")
        spat2_sq = hcl.scalar(0,"spat2_sq")
        sum_v = hcl.scalar(0,"sum_v")
        norm = hcl.scalar(0,"norm")
        spat1_sq[0] = spat_deriv[0] * spat_deriv[0]
        spat2_sq[0] = spat_deriv[1] * spat_deriv[1]
        sum_v[0]    = spat1_sq[0] + spat2_sq[0]
        norm[0]	    = hcl.sqrt(sum_v[0])

        with hcl.if_(self.dMode == "min"):
            d1[0] = -self.d_xy * spat_deriv[0] / (norm[0] + 0.00001)
            d2[0] = -self.d_xy * spat_deriv[1] / (norm[0] + 0.00001)
            with hcl.if_(spat_deriv[2] > 0):
                d3[0] = self.dTheta_Min

        return (d1[0], d2[0], d3[0], d4[0])

    def dynamics(self, t, state, uOpt, dOpt):
        x_dot = hcl.scalar(0, "x_dot")
        y_dot = hcl.scalar(0, "y_dot")
        theta_dot = hcl.scalar(0, "theta_dot")
        v_dot = hcl.scalar(0, "v_dot")

        x_dot[0] = state[3] * hcl.cos(state[2]) + dOpt[0]
        y_dot[0] = state[3] * hcl.sin(state[2]) + dOpt[1]
        theta_dot[0] = uOpt[1] + dOpt[2]
        v_dot[0] = uOpt[0]

        return (x_dot[0], y_dot[0], theta_dot[0], v_dot[0])
