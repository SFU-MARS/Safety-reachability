


import os
import numpy as np
import matplotlib.pyplot as plt
import control as ct
import control.flatsys as fs
import control.optimal as opt


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
v0 = [0.2, 0.5]
vf = [0.2, 0.5]
x0 = [0., 0, 0, v0[1]]
# actions_waypoint = [[0.25, 0, 0], [0.3, 0.1, 0.3], [0.227, 0.2225, 0],
#                     [0.113, 0.15, 0], [0.15, -0.1, -0.2]]

# actions_waypoint = [[0.25, 0, 0],[.5, 0, 0],[0.25, 0.4, 0.6],[0.25, -0.4, -0.6], [0.3, 0.3, 0.9],[0.3, -0.3, -0.9], [0.4, 0.2, 0.1],[0.4, 0.2, -0.1]
#                     ,[0.10, 0.15, 0.4], [0.10, -0.15, -0.4] ,[0.15, 0.1, 0.2],  [0.15, -0.1, -0.2], [0.2,0.4,1],[0.2,-0.4,-1] ]
actions_waypoint = [[.5, 0, 0], [0.2, -0.4, -1],[0.25, 0.4, 1],[0.3, 0.25, 0.7],[0.3, -0.25, -0.7],

                    [0.25,0.3,0.9],[0.25,-0.3,-.9],[0.4, 0.2, 0.4],[0.4, -0.2, -0.4], [0.1, 0.4, 1.1],[0.1, -0.4, -1.1],

                    [0.4, 0.1, 0.3], [0.4, -0.1, -0.3],  [0.45, 0.05, 0.1],[0.35, -0.2, -.5], [0.35, 0.2, .5], [0.45, -0.05, -0.1], [0.45, -0.1, -0.2],[0.45, 0.1, 0.2],
                    [0.1, 0.2, 1],[0.1, -0.2, -1],[0.15, -0.15, -0.8], [0.15, 0.15, 0.8],[0.2, 0.35, 1],[0.2, -0.35, -1], [0.35, -0.1, -0.1],[0.35, 0.1, 0.3] ]

xf = np.concatenate((actions_waypoint,vf[1]*np.ones((len(actions_waypoint),1))),axis = 1)
uf = [0, 0]
Tf = 1
dt=0.05
t = np.linspace(0, Tf, 20)


# Define a set of basis functions to use for the trajectories
poly = fs.PolyFamily(8)
fig=plt.figure()
count=0
# cost_fcn = opt.state_poly_constraint(vehicle_flat, np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]), [0,0,0,0.7])

lb, ub = [-1.1, -.4], [1.1, 0.4]
constraints = [opt.input_range_constraint(vehicle_flat, lb, ub)]
for i in range (len(actions_waypoint)):
# Find a trajectory between the initial condition and the final condition
#     traj = fs.point_to_point(vehicle_flat,Tf, x0, u0, xf[i], uf, basis=poly)#, constraints =[( -1.1, 1.1) ,(  -0.4, 0.4)])
    traj_const = fs.point_to_point(vehicle_flat, t, x0, u0, xf[i], uf,constraints=constraints, basis=fs.PolyFamily(8))
                                   # ,cost=cost_fcn)
    # Create the trajectory

    x, u = traj_const.eval(t)
    plot_results(t, x, u)
    # if abs(u[0].max())<=1.1 and abs(u[1].max())<=.4 and abs(x[3].max())<=0.7:
    #
    #     count+=1
    #     print(xf[i])
    # else:
    #     continue
    from mpl_toolkits import mplot3d

#
#     ax = fig.gca(projection='3d')
#     # plt.plot(xf[i][0],xf[i][1],xf[i][2])
#     ax.plot3D(x[0], x[1], x[2])
#     ax.scatter3D(xf[i][0],xf[i][1],xf[i][2])
#
#     ax.set_xlabel('X Label')
#     ax.set_ylabel('Y Label')
#     ax.set_zlabel('Theta Label')
#     elev=90
#     azim=-90
#     ax.view_init(elev, azim)
#     # ax.text2D("v0=0.2->vf=0.2")
#     #plt.hold(True)
#
#
# plt.show()
print(count)
