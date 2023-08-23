import tensorflow as tf
from trajectory.trajectory import Trajectory
from systems.dubins_car import DubinsCar  # Make sure to import the appropriate DubinsCar class

class Dubins4D(DubinsCar):
    """ A discrete time dubins car with state
    [x, y, theta, v] and actions [w, a]
    (angular velocity and linear acceleration).
    The dynamics are:

    x(t+1) = x(t) + saturate_linear_velocity(v(t)) * cos(theta_t) * delta_t
    y(t+1) = y(t) + saturate_linear_velocity(v(t)) * sin(theta_t) * delta_t
    theta(t+1) = theta(t) + saturate_angular_velocity(w(t)) * delta_t
    v(t+1) = v(t) + saturate_acceleration(a(t)) * delta_t
    """

    def __init__(self, dt):
        super().__init__(dt, x_dim=4, u_dim=2)  # Adjust x_dim to 4
        self._angle_dims = 1

    def _simulate_ideal(self, x_nkd, u_nkf, t=None):
        with tf.name_scope('simulate'):
            theta_nk1 = x_nkd[:, :, 2:3]
            v_nk1 = x_nkd[:, :, 3:4]
            x_new_nkd = tf.concat([x_nkd[:, :, :2],
                                   theta_nk1 + self._dt * self._saturate_angular_velocity(u_nkf[:, :, 0:1]),
                                   v_nk1 + self._dt * self._saturate_acceleration(u_nkf[:, :, 1:2])],
                                  axis=2)
            delta_x_nkd = tf.concat([v_nk1 * tf.cos(theta_nk1),
                                     v_nk1 * tf.sin(theta_nk1),
                                     tf.zeros_like(u_nkf)], axis=2)
            return x_new_nkd + self._dt * delta_x_nkd

    # Modify the jac_x function according to the new dynamics
    def jac_x(self, trajectory):
        x_nk4, u_nk2 = self.parse_trajectory(trajectory)
        with tf.name_scope('jac_x'):
            theta_nk1 = x_nk4[:, :, 2:3]
            v_nk1 = x_nk4[:, :, 3:4]

            diag_nk4 = tf.concat([tf.ones_like(x_nk4[:, :, :2]),
                                  self._saturate_angular_velocity_prime(u_nk2[:, :, 0:1] * self._dt + theta_nk1),
                                  self._saturate_acceleration_prime(u_nk2[:, :, 1:2] * self._dt + v_nk1)],
                                 axis=2)

            column2_nk4 = tf.concat([-v_nk1 * tf.sin(theta_nk1),
                                     v_nk1 * tf.cos(theta_nk1)], axis=2)
            column3_nk4 = tf.concat([tf.cos(theta_nk1),
                                     tf.sin(theta_nk1)], axis=2)

            update_nk44 = tf.stack([tf.zeros_like(x_nk4),
                                    tf.zeros_like(x_nk4),
                                    column2_nk4,
                                    column3_nk4], axis=3)

            return tf.linalg.diag(diag_nk4) + self._dt * update_nk44

    # Modify the parse_trajectory function to match the new action dimensions
    def parse_trajectory(self, trajectory):
        u_nk2 = tf.concat([trajectory.angular_speed_nk1(),
                           trajectory.acceleration_nk1()], axis=2)
        return trajectory.position_heading_speed_nk4(), u_nk2

    # Modify the assemble_trajectory function to match the new state dimensions
    def assemble_trajectory(self, x_nkd, u_nkf, pad_mode=None):
        n = x_nkd.shape[0].value
        k = x_nkd.shape[1].value
        u_nkf = self._pad_control_vector(u_nkf, k, pad_mode=pad_mode)
        position_nk2, heading_nk1 = x_nkd[:, :, :2], x_nkd[:, :, 2:3]
        speed_nk1 = x_nkd[:, :, 3:4]
        angular_speed_nk1, acceleration_nk1 = u_nkf[:, :, 0:1], u_nkf[:, :, 1:2]
        return Trajectory(dt=self._dt, n=n, k=k, position_nk2=position_nk2,
                          heading_nk1=heading_nk1, speed_nk1=speed_nk1,
                          angular_speed_nk1=angular_speed_nk1,
                          acceleration_nk1=acceleration_nk1, variable=False)
