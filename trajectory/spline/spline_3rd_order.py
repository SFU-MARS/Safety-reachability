from trajectory.spline.spline import Spline
import tensorflow as tf
import numpy as np
import control.flatsys as fs
import control.optimal as opt
from scipy.optimize import LinearConstraint


# Function to take states, inputs and return the flat flag
def vehicle_flat_forward(x, u, params={}):
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
    thdot = u[1]
    vdot = u[0]
    # Second derivatives of the flat output (setting vdot = 0)
    zflag[0][2] = (- vel * thdot * np.sin(theta)) + (vdot * np.cos(theta))
    zflag[1][2] = (vel * thdot * np.cos(theta)) + (vdot * np.sin(theta))
    return zflag


# Function to take the flat flag and return states, inputs
def vehicle_flat_reverse(zflag, params={}):
    # Get the parameter values
    # Create a vector to store the state and inputs
    X = np.zeros(4)
    u = np.zeros(2)

    x = zflag[0][0]
    y = zflag[1][0]

    xd = zflag[0][1]
    yd = zflag[1][1]

    xdd = zflag[0][2]
    ydd = zflag[1][2]

    theta = np.arctan2(yd, xd+1e-5)
    v = np.linalg.norm([yd, xd])

    # Given the flat variables, solve for the state
    X[0] = x  # x position
    X[1] = y # y position
    X[2] = theta # tan(theta) = ydot/xdot
    X[3] = v
    # And next solve for the inputs
    u[1] = 1 / (1 + (yd / (xd + 1e-5)) ** 2) * (
            (ydd * xd) - (xdd * yd) / (xd ** 2 + 1e-5)
    )
    u[0] = 1.0 / (2*(v + 1e-5)) * ((2 * ydd * yd) + (2 * xdd * xd))
    return X, u


class Spline3rdOrder(Spline):
    def __init__(self, dt, n, k, params):
        super(Spline3rdOrder, self).__init__(dt=dt, n=n, k=k)
        self.params = params

    """ A class representing a 3rd order spline for a mobile ground robot
    (in a 2d cartesian plane). The 3rd order spline allows for constraints
    on the start config, [x0, y0, theta0, v0], and goal config,
    [xg, yg, thetag,vg]. Angular speeds w0 and wg are not constrainable.
    """


    # def fit1(self, start_config, goal_config, final_times_n1=None, factors=None):
    #
    #     u0 = np.asarray([0, 0])
    #     uf = np.asarray([0, 0])
    #     dt = 0.05
    #     t = np.arange(0, 6, dt)
    #     Av = np.zeros((3, 6))
    #     Av[0, 3] = 1
    #     Av[1, 4] = 1
    #     Av[2, 5] = 1
    #     lbv = [0, -0.4, -1.1]
    #     ubv = [0.7, 0.4, 1.1]
    #
    #     trajectory_constraints = [
    #         (LinearConstraint, Av, lbv, ubv)
    #     ]
    #     vehicle_flat = fs.FlatSystem(forward=vehicle_flat_forward, reverse=vehicle_flat_reverse, inputs=2, states=4)
    #     # x0=start_config
    #     x0 = start_config.position_heading_speed_nk4()[0].numpy().reshape((4,))
    #     # xf=goal_config
    #
    #     xfs = goal_config.position_heading_speed_nk4().numpy().reshape((-1, 4))
    #     xs, ys, headings, speeds, accs, omegas = [], [], [], [], [], []
    #     valid_idxs = []
    #
    #     import tqdm
    #     for idx, xf in tqdm.tqdm(enumerate(xfs), total=xfs.shape[0]):
    #         try:
    #             traj_const = fs.point_to_point(
    #                 vehicle_flat, t, x0, u0, xf, uf, basis=None,  # fs.PolyFamily(8),
    #                 # constraints=trajectory_constraints
    #             )
    #         except RuntimeError:
    #             continue
    #         # ,cost=cost_fcn)
    #         # Create the trajectory
    #         # x, u = traj_const.eval(t)
    #         # traj_const = fs.point_to_point(vehicle_flat, t, x0, u0, xf, uf,
    #         #                                basis=fs.PolyFamily(8))  # constraints=constraints,
    #         # ,cost=cost_fcn)
    #         # Create the trajectory
    #
    #         valid_idxs.append(idx)
    #         x, u = traj_const.eval(t)
    #         xs.append(x[0, :])
    #         ys.append(x[1, :])
    #         headings.append(x[2, :])
    #         speeds.append(x[3, :])
    #         accs.append(u[0, :])
    #         omegas.append(u[1, :])
    #
    #     xs, ys, headings, speeds, accs, omegas \
    #         = np.stack(xs), np.stack(ys), np.stack(headings), np.stack(speeds), np.stack(accs), np.stack(omegas)
    #
    #     xs = xs[:, :, np.newaxis].astype(np.float32)
    #     ys = ys[:, :, np.newaxis].astype(np.float32)
    #     headings = headings[:, :, np.newaxis].astype(np.float32)
    #     omegas = omegas[:, :, np.newaxis].astype(np.float32)
    #     accs = accs[:, :, np.newaxis].astype(np.float32)
    #     # accs = np.stack([accs[0] for _ in range(21483)])
    #     speeds = speeds[:, :, np.newaxis].astype(np.float32)
    #
    #     # self.spline_trajectory._acceleration_nk1.assign(tf.convert_to_tensor(accs))
    #     self.spline_trajectory._position_nk2 = tf.convert_to_tensor(np.concatenate((xs, ys), axis=2))
    #     self.spline_trajectory._heading_nk1 = tf.convert_to_tensor(headings)
    #     self.spline_trajectory._acceleration_nk1 = tf.convert_to_tensor(accs)
    #     self.spline_trajectory._speed_nk1 = tf.convert_to_tensor(speeds)
    #     self.spline_trajectory._angular_speed_nk1 = tf.convert_to_tensor(omegas)
    #
    #     self.spline_trajectory.vars = [self.spline_trajectory._position_nk2, self.spline_trajectory._speed_nk1,
    #                                    self.spline_trajectory._acceleration_nk1, self.spline_trajectory._heading_nk1,
    #                                    self.spline_trajectory._angular_speed_nk1,
    #                                    self.spline_trajectory._angular_acceleration_nk1]
    #
    #     p = self.params
    #     times_nk = tf.tile(tf.linspace(0., p.planning_horizon_s, p.planning_horizon)[None],
    #                        [self.n, 1])  # number of waypoints * number of planning horizon. maximum time = 6
    #     final_times_n1 = tf.ones((self.n, 1), dtype=tf.float32) * p.planning_horizon_s
    #     self.final_times_n1 = final_times_n1
    #
    #     # Update the batch size as the same spline object
    #     # can be used with multiple start/ goal configurations
    #     self.n = start_config.n
    #     self.valid_horizons_n1 = tf.ceil(self.final_times_n1 / self.dt)
    #
    #     return valid_idxs


    def fit(self, start_config, goal_config, final_times_n1=None, factors=None):  # Get the coefficients of Spline based on start and goal
        """Fit a 3rd order spline between start config and goal config.
        Factors_n2 represent 2 degrees of freedom in fitting the spline.
        If factors_n2=None it is set heuristically below.
        If final_time_n1=None, a final time of 1 is used.
        The spline is of the form:
            p(t) = a3(t/tf)^3+b3(t/tf)^2+c3(t/tf)+d3
            x(p) = a1p^3+b1p^2+c1p+d1
            y(p) = a2p^2+b2p^2+c2p+d2
        """

        # Compute the factors
        if factors is None:  # Compute them heuristically
            factor1_n1 = start_config.speed_nk1()[:, :, 0] + \
                         tf.norm(goal_config.position_nk2()-start_config.position_nk2(), axis=2)
            factor2_n1 = factor1_n1
            factors_n2 = tf.concat([factor1_n1, factor2_n1], axis=1)
        else:
            factors_n2 = factors

        # Compute the final times
        if final_times_n1 is None:
            final_times_n1 = tf.ones((self.n, 1))

        # Fit spline
        with tf.name_scope('fit_spline'):
            f1_n1, f2_n1 = factors_n2[:, 0:1], factors_n2[:, 1:]

            start_pos_n12 = start_config.position_nk2()
            goal_pos_n12 = goal_config.position_nk2()

            # Multiple solutions if start and goal are the same x,y coordinates
            assert(tf.reduce_all(tf.norm(goal_pos_n12-start_pos_n12, axis=2) >=
                                 self.params.epsilon))

            x0_n1, y0_n1 = start_pos_n12[:, :, 0], start_pos_n12[:, :, 1]
            t0_n1 = start_config.heading_nk1()[:, :, 0]
            v0_n1 = start_config.speed_nk1()[:, :, 0]

            xg_n1, yg_n1 = goal_pos_n12[:, :, 0], goal_pos_n12[:, :, 1]
            tg_n1 = goal_config.heading_nk1()[:, :, 0]
            vg_n1 = goal_config.speed_nk1()[:, :, 0]

            d1_n1 = x0_n1
            c1_n1 = f1_n1*tf.cos(t0_n1)
            a1_n1 = f2_n1*tf.cos(tg_n1)-2*xg_n1+c1_n1+2*d1_n1
            b1_n1 = 3*xg_n1-f2_n1*tf.cos(tg_n1)-2*c1_n1-3*d1_n1

            d2_n1 = y0_n1
            c2_n1 = f1_n1*tf.sin(t0_n1)
            a2_n1 = f2_n1*tf.sin(tg_n1)-2*yg_n1+c2_n1+2*d2_n1
            b2_n1 = 3*yg_n1-f2_n1*tf.sin(tg_n1)-2*c2_n1-3*d2_n1

            c3_n1 = (final_times_n1 * v0_n1) / f1_n1
            a3_n1 = (final_times_n1*vg_n1/f2_n1) + c3_n1 - 2.
            b3_n1 = 1. - c3_n1 - a3_n1

            self.x_coeffs_n14 = tf.stack([a1_n1, b1_n1, c1_n1, d1_n1], axis=2)
            self.y_coeffs_n14 = tf.stack([a2_n1, b2_n1, c2_n1, d2_n1], axis=2)
            self.p_coeffs_n14 = tf.stack([a3_n1, b3_n1, c3_n1, 0.0*c3_n1],
                                         axis=2)
            self.final_times_n1 = final_times_n1

            # Update the batch size as the same spline object
            # can be used with multiple start/ goal configurations
            self.n = start_config.n

    def _eval_spline(self, ts_nk, calculate_speeds=True):  # Get the spline trajectory values (x, y, theta, v, omega)
        """ Evaluates the spline on points in ts_nk
        Assumes ts is normalized to be in [0, 1.]
        """
        x_coeffs_n14 = self.x_coeffs_n14
        y_coeffs_n14 = self.y_coeffs_n14
        p_coeffs_n14 = self.p_coeffs_n14

        with tf.name_scope('eval_spline'):
            ts_n4k = tf.stack([tf.pow(ts_nk, 3), tf.pow(ts_nk, 2),
                               ts_nk, tf.ones_like(ts_nk)], axis=1)
            ps_nk = tf.squeeze(tf.matmul(p_coeffs_n14, ts_n4k), axis=1)

            ps_n4k = tf.stack([tf.pow(ps_nk, 3), tf.pow(ps_nk, 2),
                               ps_nk, tf.ones_like(ps_nk)], axis=1)
            ps_dot_n4k = tf.stack([3.0*tf.pow(ps_nk, 2), 2.0*ps_nk,
                                   tf.ones_like(ps_nk), tf.zeros_like(ps_nk)],
                                  axis=1)

            xs_nk = tf.squeeze(tf.matmul(x_coeffs_n14, ps_n4k), axis=1)
            ys_nk = tf.squeeze(tf.matmul(y_coeffs_n14, ps_n4k), axis=1)

            xs_dot_nk = tf.squeeze(tf.matmul(x_coeffs_n14, ps_dot_n4k), axis=1)
            ys_dot_nk = tf.squeeze(tf.matmul(y_coeffs_n14, ps_dot_n4k), axis=1)

            self._position_nk2 = tf.stack([xs_nk, ys_nk], axis=2)
            self._heading_nk1 = tf.atan2(ys_dot_nk, xs_dot_nk)[:, :, None]

            if calculate_speeds:
                ts_dot_n4k = tf.stack([3.0*tf.pow(ts_nk, 2), 2.0*ts_nk,
                                       tf.ones_like(ts_nk), tf.zeros_like(ts_nk)],
                                      axis=1)
                ps_ddot_n4k = tf.stack([6.0*ps_nk, 2.0*tf.ones_like(ps_nk),
                                        tf.zeros_like(ps_nk),
                                        tf.zeros_like(ps_nk)], axis=1)

                ps_dot_nk = tf.squeeze(tf.matmul(p_coeffs_n14, ts_dot_n4k), axis=1)

                xs_ddot_nk = tf.squeeze(tf.matmul(x_coeffs_n14, ps_ddot_n4k), axis=1)
                ys_ddot_nk = tf.squeeze(tf.matmul(y_coeffs_n14, ps_ddot_n4k), axis=1)

                speed_ps_nk = tf.sqrt(xs_dot_nk**2 + ys_dot_nk**2) # Using differential flatness, v = sqrt(x^2 + y^2)
                speed_nk = (speed_ps_nk*ps_dot_nk)

                numerator_nk = xs_dot_nk*ys_ddot_nk-ys_dot_nk*xs_ddot_nk
                angular_speed_nk = numerator_nk/(speed_ps_nk**2) * ps_dot_nk

                self._speed_nk1 = speed_nk[:, :, None]
                self._angular_speed_nk1 = angular_speed_nk[:, :, None]
                # numerator_nk_a = xs_dot_nk * xs_ddot_nk + ys_dot_nk * ys_ddot_nk
                # acceleration_nk = numerator_nk_a / (speed_ps_nk) * ps_dot_nk
                # acceleration_nk1=acceleration_nk[:, :, None]
                # print(acceleration_nk1)
                self._acceleration_nk1 = tf.zeros_like(self._speed_nk1)
                self._angular_acceleration_nk1 = tf.zeros_like(self._speed_nk1)

    def check_dynamic_feasibility(self, speed_max_system, angular_speed_max_system, horizon_s):
        """Checks whether the current computed spline can be executed in time <= horizon_s (specified in seconds)
        while respecting max speed and angular speed constraints. Returns the batch indices of all valid splines."""
        
        # Compute the minimum horizon required to execute the spline while ensuring dynamic feasibility
        required_horizon_n1 = self.compute_dynamically_feasible_horizon(speed_max_system, angular_speed_max_system)
        
        # Compute the valid splines
        valid_idxs_n = tf.where(required_horizon_n1 <= horizon_s)[:, 0]
        return tf.cast(valid_idxs_n, tf.int32)
    
    def compute_dynamically_feasible_horizon(self, speed_max_system, angular_speed_max_system):
        """
        Compute the horizon (in seconds) such that the computed spline respect the speed and angular
        speed at all times.
        Speed assumed to be in [0, speed_max_system]
        Angular speed assumed to be in [-angular_speed_max_system, angular_speed_max_system]
        """
        # Compute the horizon required to make sure that we satisfy the speed constraints at all times
        max_speed_n1 = tf.reduce_max(self.speed_nk1(), axis=1)
        required_horizon_speed_n1 = self.final_times_n1 * max_speed_n1/speed_max_system

        # Compute the horizon required to make sure that we satisfy the angular speed constraints at all times
        max_angular_speed_n1 = tf.reduce_max(tf.abs(self.angular_speed_nk1()), axis=1)
        required_horizon_angular_speed_n1 = self.final_times_n1 * max_angular_speed_n1 / angular_speed_max_system
        
        # Compute the horizon required to make sure that we satisfy all control constraints at all times
        return tf.maximum(required_horizon_speed_n1, required_horizon_angular_speed_n1)
    
    def rescale_spline_horizon_to_dynamically_feasible_horizon(self, speed_max_system,
                                                               angular_speed_max_system,
                                                               minimum_horizon=0.0):
        """
        Rescale the spline horizon to a new horizon without recomputing the spline coefficients.
        """
        # Compute the minimum horizon required to execute the spline while ensuring dynamic feasibility
        
        required_horizon_n1 = self.compute_dynamically_feasible_horizon(speed_max_system, angular_speed_max_system)
       
        # Enforce a minimum horizon
        required_horizon_n1 = tf.maximum(required_horizon_n1, minimum_horizon)
        
        # Reset the final times
        self.final_times_n1 = required_horizon_n1
        
        # Valid horizon for each trajectory in the batch
        # in discrete time steps
        self.valid_horizons_n1 = tf.ceil(self.final_times_n1/self.dt)

        # Reevaluate the spline to be consistent with the new horizon
        self.eval_spline(self.ts_nk)
        
    def find_trajectories_within_a_horizon(self, horizon_s):
        """
        Find the indices of splines whose final time is within the horizon [0, horizon_s].
        """
        valid_idxs_n = tf.where(self.final_times_n1 <= horizon_s)[:, 0]
        return tf.cast(valid_idxs_n, tf.int32)

    @staticmethod
    def ensure_goals_valid(start_x, start_y, goal_x_nk1, goal_y_nk1, goal_theta_nk1, epsilon):
        """ Perturbs goal_x and goal_y by epsilon if needed ensuring that a unique spline exists.
        Assumes that all goal angles are within [-pi/2., pi/2]."""
        # assert((goal_theta_nk1 >= -np.pi/2.).all() and (goal_theta_nk1 <= np.pi/2.).all())
        norms = np.linalg.norm(np.concatenate([goal_x_nk1-start_x, goal_y_nk1-start_y], axis=2), axis=2)
        invalid_idxs = (norms == 0.0)
        goal_x_nk1[invalid_idxs] += epsilon
        goal_y_nk1[invalid_idxs] += np.sign(np.sin(goal_theta_nk1[invalid_idxs]))*epsilon
        return goal_x_nk1, goal_y_nk1, goal_theta_nk1

    def render(self, axs, batch_idx=0, freq=4, plot_heading=False,
               plot_velocity=False, label_start_and_end=True):
        super().render(axs, batch_idx, freq, plot_heading=plot_heading,
                       plot_velocity=plot_velocity,
                       label_start_and_end=label_start_and_end, name='Spline')
