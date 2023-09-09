from utils import utils
from dotmap import DotMap
from systems.dubins_v4 import DubinsV4



def create_params():
    p = DotMap()
    p.system = DubinsV4
    p.dt = .05
    p.v_bounds = [0.3,0.6]
    p.w_bounds = [-1.1, 1.1]
    p.a_bounds = [-0.4, 0.4]



    # Set the acceleration bounds such that
    # by default they are never hit
    p.linear_acc_max = 10e7
    p.angular_acc_max = 10e7
    
    p.simulation_params = DotMap(simulation_mode='ideal',
                                 noise_params = DotMap(is_noisy=False,
                                                       noise_type='uniform',
                                                       noise_lb=[-0.02, -0.02, 0.],
                                                       noise_ub=[0.02, 0.02, 0.],
                                                       noise_mean=[0., 0., 0.],
                                                       noise_std=[0.02, 0.02, 0.]))
    return p
