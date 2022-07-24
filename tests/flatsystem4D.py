


import os
import numpy as np
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
    vel= x[3]
    # zflag[3][0] = x[3]

    # First derivatives of the flat output
    zflag[0][1] = vel * np.cos(theta)  # dx/dt
    zflag[1][1] = vel * np.sin(theta)  # dy/dt
    # zflag[2][1] = u[0]
    # zflag[3][1] = u[1]
    # First derivative of the angle
    thdot =u[0]
    vdot= u[1]

    # Second derivatives of the flat output (setting vdot = 0)
    zflag[0][2] = - vel * thdot * np.sin(theta)+vdot * np.cos(theta)
    zflag[1][2] =   vel * thdot * np.cos(theta)+vdot * np.sin(theta)

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
    u[0] = 1/(1+(zflag[0][1]/zflag[0][1])**2) *((zflag[1][2]*zflag[0][1])-(zflag[0][2]*zflag[1][1]))/(zflag[0][1]**2)
    u[1] = 0.5*(1/x[3])*(2*zflag[1][2]*zflag[1][1]+2*zflag[0][2]*zflag[0][1])

    return x, u


def plot_results(t, x, u):

    plt.figure(figsize=[9, 4.5])
    # Plot the trajectory in xy coordinate
    plt.subplot(1, 4, 1)
    plt.plot(x[1], x[0])
    plt.xlabel('y [m]')
    plt.ylabel('x [m]')


    # Time traces of the state and input

    plt.subplot(2, 4, 2)
    plt.plot(t, x[0])
    plt.ylabel('x [m]')
    plt.tight_layout()

    plt.subplot(2, 4, 3)
    plt.plot(t, x[1])
    plt.ylabel('y [m]')
    plt.tight_layout()


    plt.subplot(2, 4, 4)
    plt.plot(t, x[2])
    plt.ylabel('theta [rad]')
    plt.tight_layout()

    plt.subplot(2, 4, 6)
    plt.plot(t, x[3])
    plt.ylabel('v [m/s]')
    plt.tight_layout()


    plt.subplot(2, 4, 7)
    plt.plot(t, u[0])
    plt.xlabel('Time t [sec]')
    plt.ylabel('w [rad/s]')
    plt.tight_layout()

    # plt.axis([0, t[-1], u0[0] - 1, uf[0] + 1])

    plt.subplot(2, 4, 8)
    plt.plot(t, u[1]);
    plt.xlabel('Time t [sec]')
    plt.ylabel('a [m2/s]')
    plt.tight_layout()
    plt.show()

vehicle_flat = fs.FlatSystem(forward=vehicle_flat_forward, reverse=vehicle_flat_reverse,  inputs=2,states=4)
# x0 = [0., 0,0, 0.25]
# x0 = [0., 0, 0, 0.35]
u0 = [0, 0.]
# xf = [0.2, 0.12, 1,0.25]
# xf = [0.4, 0.2, 0.2, 0.4]
x0 = [0., 0, 0, 0.25]
xf = [0.25, 0, 0, 0.35]
uf = [0, 0]
Tf = 1
t = np.linspace(0, Tf, 100)


# Define a set of basis functions to use for the trajectories
poly = fs.PolyFamily(6)

# Find a trajectory between the initial condition and the final condition
traj = fs.point_to_point(vehicle_flat,Tf, x0, u0, xf, uf, basis=poly)#, constraints =[( -1.1, 1.1) ,(  -0.4, 0.4)])

# Create the trajectory

x, u = traj.eval(t)

plot_results(t, x, u)
