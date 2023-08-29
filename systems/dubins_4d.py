import tensorflow as tf
from trajectory.trajectory import Trajectory
from systems.dubins_car import DubinsCar


class Dubins4D(DubinsCar):
    """ A discrete time dubins car with state
    [x, y, theta,v] and actions [w, a]. The dynamics are:

    x(t+1) = x(t) + v(t)*cos(theta_t)*delta_t
    y(t+1) = y(t) + v(t)*sin(theta_t)*delta_t
    theta(t+1) = theta_t + saturate_angular_velocity(w(t))*delta_t
    v(t+1) = saturate_linear_velocity(v(t) + a(t)*delta_t)
    """

    def __init__(self, dt, simulation_params=None):
        super(Dubins4D, self).__init__(dt=0.05, x_dim=4, u_dim=2)
        self._angle_dims = 2
        self.simulation_params = simulation_params

        # if self.simulation_params.noise_params.is_noisy:
        #     print('This Dubins car model has some noise. Please turn off the noise if this was not intended.')

    def _simulate_ideal(self, x_nk4, u_nk2, t=None):
        with tf.name_scope('simulate'):
            theta_nk1 = x_nk4[:, :, 2:3]
            v_nk1 = x_nk4[:, :, 3]
            delta_x_nk4 = tf.stack([self._saturate_linear_velocity(x_nk4[:, :, 3]) * tf.cos(x_nk4[:, :, 2]),
                                    self._saturate_linear_velocity(x_nk4[:, :, 3]) * tf.sin(x_nk4[:, :, 2]),
                                    self._saturate_angular_velocity(u_nk2[:, :, 1]),
                                    self._saturate_linear_velocity(
                                        v_nk1 + self._dt * self._saturate_linear_acceleration(u_nk2[:, :, 0]))],
                                   axis=2)

            # theta_nk1 = x_nkd[:, :, 2:3]
            # v_nk1 = x_nkd[:, :, 3:4]
            # x_new_nkd = tf.concat([x_nkd[:, :, :3],
            #                        self._saturate_linear_velocity(v_nk1 + self._dt*u_nkf[:, :, 0:1]),
            #                        self._saturate_angular_velocity(w_nk1 + self._dt*u_nkf[:, :, 1:2])],
            #                       axis=2)
            # delta_x_nkd = tf.concat([v_nk1*tf.cos(theta_nk1),
            #                          v_nk1*tf.sin(theta_nk1),
            #                          w_nk1,
            #                          tf.zeros_like(u_nkf)], axis=2)
            # return x_new_nkd + self._dt*delta_x_nk4
            # Add noise (or disturbance) if required
            # if self.simulation_params.noise_params.is_noisy:
            #     noise_component = self.compute_noise_component(required_shape=tf.shape(x_nk3), data_type=x_nk3.dtype)
            #     return x_nk3 + self._dt * delta_x_nk3 + noise_component
            # else:
            return x_nk4 + self._dt * delta_x_nk4

    def jac_x(self, trajectory):
        # Computes the A matrix in x_{t+1} = Ax_{t} + Bu_{t}
        x_nk4, u_nk2 = self.parse_trajectory(trajectory)
        with tf.name_scope('jac_x'):
            v_nk1 = x_nk4[:, :, 3:4]
            theta_nk1 = x_nk4[:, :, 2:3]

            diag_nk4 = tf.concat([tf.ones_like(x_nk4[:, :, :3]),
                                  self._saturate_linear_velocity_prime(u_nk2[:, :, 0:1])*self._dt + v_nk1], axis=2)
            # theta column
            column2_nk4 = tf.concat([-v_nk1*tf.sin(theta_nk1),
                                     v_nk1*tf.cos(theta_nk1),
                                     tf.zeros_like(x_nk4[:, :, :2])], axis=2)
            # v column
            column3_nk4 = tf.concat([tf.cos(theta_nk1),
                                     tf.sin(theta_nk1),
                                     tf.zeros_like(x_nk4[:, :, :2])], axis=2)
            update_nk44 = tf.stack([tf.zeros_like(x_nk4),
                                    tf.zeros_like(x_nk4),
                                    column2_nk4,
                                    column3_nk4], axis=3)
            return tf.linalg.diag(diag_nk4) + self._dt*update_nk44

    def jac_u(self, trajectory):
        # This function computes the B matrix in x_{t+1} = Ax_{t} + Bu_{t}
        x_nk4, u_nk2 = self.parse_trajectory(trajectory)
        with tf.name_scope('jac_u'):
            # TODO: check if the index 0 corresponds to acceleration and index 1 corresponds to angle speed
            wtilde_prime_nk = self._saturate_angular_velocity_prime(u_nk2[:, :, 1:2])
            #vtilde_prime_nk = self._saturate_linear_velocity_prime(u_nk2[:, :, 1])
            v_nk1 = x_nk4[:, :, 3:4]
            # w column
            b1_nk3 = tf.concat([tf.zeros_like(x_nk4[:, :, :2]),
                                wtilde_prime_nk,
                                tf.zeros_like(x_nk4[:, :, 0:1])], axis=2)

            # v column
            b2_nk3 = tf.concat([tf.zeros_like(x_nk4[:, :, :3]),
                                self._saturate_linear_velocity_prime(u_nk2[:, :, 1:2] * self._dt + v_nk1)],
                               axis=2)
            B_nk32 = tf.stack([b1_nk3, b2_nk3], axis=3)
        return B_nk32 * self._dt

    def parse_trajectory(self, trajectory):
        """ A utility function for parsing a trajectory object.
        Returns x_nkd, u_nkf which are states and actions for the
        system """
        return tf.concat((trajectory.position_and_heading_nk3(), trajectory.speed_nk1()), axis=2), tf.concat((trajectory.acceleration_nk1(),trajectory.angular_speed_nk1()), axis=2)

    def assemble_trajectory(self, x_nkd, u_nkf, pad_mode=None):
        """ A utility function for assembling a trajectory object
        from x_nkd, u_nkf, a list of states and actions for the system.
        Here d=3=state dimension and u=2=action dimension. """
        n = x_nkd.shape[0].value
        k = x_nkd.shape[1].value
        u_nkf = self._pad_control_vector(u_nkf, k, pad_mode=pad_mode)
        position_nk2, heading_nk1 = x_nkd[:, :, :2], x_nkd[:, :, 2:3]
        speed_nk1, angular_speed_nk1 = u_nkf[:, :, 0:1], u_nkf[:, :, 1:2]
        speed_nk1 = self._saturate_linear_velocity(speed_nk1)
        angular_speed_nk1 = self._saturate_angular_velocity(angular_speed_nk1)
        return Trajectory(dt=self._dt, n=n, k=k, position_nk2=position_nk2,
                          heading_nk1=heading_nk1, speed_nk1=speed_nk1,
                          angular_speed_nk1=angular_speed_nk1, variable=False)
    
    def compute_noise_component(self, required_shape, data_type):
        """
        Compute a noise component for the Dubins car.
        """
        if self.simulation_params.noise_params.noise_type == 'uniform':
            return tf.random_uniform(required_shape, self.simulation_params.noise_params.noise_lb,
                                     self.simulation_params.noise_params.noise_ub,
                                     dtype=data_type)
        elif self.simulation_params.noise_params.noise_type == 'gaussian':
            return tf.random_normal(required_shape, mean=self.simulation_params.noise_params.noise_mean,
                                    stddev=self.simulation_params.noise_params.noise_std, dtype=data_type)
        else:
            raise NotImplementedError('Unknown noise type.')
