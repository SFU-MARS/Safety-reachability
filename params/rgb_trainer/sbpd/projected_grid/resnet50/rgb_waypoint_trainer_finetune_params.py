
from dotmap import DotMap
from params.reachability_map.reachability_map_params import create_reachability_data_dir_params

from params.simulator.sbpd_simulator_params import create_params as create_simulator_params
from params.visual_navigation_trainer_params import create_params as create_trainer_params

from params.waypoint_grid.sbpd_image_space_grid import create_params as create_waypoint_params
from params.model.resnet50_arch_v1_params import create_params as create_model_params

def create_rgb_trainer_params():
    from params.reachability_map.reachability_map_params import create_params as create_reachability_map_params

    # Load the dependencies
    simulator_params = create_simulator_params()

    # Seed
    # simulator_params.seed = 10
    simulator_params.seed = 17 # 5
    # simulator_params.seed = 9 

    # Ensure the waypoint grid is projected SBPD Grid
    simulator_params.planner_params.control_pipeline_params.waypoint_params = create_waypoint_params()

    # Ensure the renderer modality is rgb
    simulator_params.obstacle_map_params.renderer_params.camera_params.modalities = ['rgb','disparity']
    simulator_params.obstacle_map_params.renderer_params.camera_params.img_channels = 3
    simulator_params.obstacle_map_params.renderer_params.camera_params.width = 1024
    simulator_params.obstacle_map_params.renderer_params.camera_params.height = 1024
    simulator_params.obstacle_map_params.renderer_params.camera_params.im_resize = 0.21875

    # Change episode horizon
    simulator_params.episode_horizon_s = 6 #0.5 : TTR calculation, 80: waypoints calculation (seconds)

    # Ensure the renderer is using area3
    # TODO: When generating our own data, choose area 3, area4, area5
    #  When testing, choosing area 1!
    simulator_params.obstacle_map_params.renderer_params.building_name = 'area5a' # area 3 for training
    # simulator_params.obstacle_map_params.renderer_params.building_name = 'area1'  # area 1 for testing
    # TODO: area3: thread='v1'; area4: thread='v2'; area5a: thread='v3'
    simulator_params.reachability_map_params.thread = 'v3'
    # specify reachability data dir name according to building_name and thread
    create_reachability_data_dir_params(simulator_params.reachability_map_params)

    # Save trajectory data
    simulator_params.save_trajectory_data = True

    # Choose cost function
    # TODO: in training, always use 'p.data_creation.data_dirheuristics'
    #simulator_params.cost = 'heuristics'  #for train and test
    simulator_params.cost = 'reachability' #for generation

    p = create_trainer_params(simulator_params=simulator_params)

    # Create the model params
    p.model = create_model_params()

    return p


def create_params():
    p = create_rgb_trainer_params()

    # Change the number of inputs to the model
    p.model.num_outputs = 35# + 4 + 4 #3*4 kernel 3
    # p.model.num_outputs = 3 # (x, y ,theta)


    # Image size to [224, 224, 3]
    p.model.num_inputs.image_size = [224, 224, 3]

    # Finetune the resnet weights
    p.model.arch.finetune_resnet_weights = True # True

    # Change the learning rate and num_samples
    # p.trainer.lr = 2e-1
    p.trainer.lr = 1e-5
    p.trainer.batch_size = 5 # 5#48 original, changed after error 36
    p.trainer.model_version = 'v2'
    # p.trainer.batch_size = 32#60
    #
    # Todo: num_samples are too large
    # p.trainer.num_samples = int(200) # original: 150e3
    # p.trainer.num_samples = int(45) #int(2400)48e4
    # p.trainer.num_samples = int(1 *4) #to have one train and val with 20 wp
    # p.trainer.num_samples = int(100e3) #to have one train and val with 20 wp
    p.trainer.num_samples = int(200_000) #int(900*5) #int(5000*5) #int(200_000) #int(5000)
    # p.trainer.num_samples = int(60 * 133)
    # p.trainer.num_samples = int(3780)
    # p.trainer.num_samples = int(1050)
    # p.trainer.num_samples = int(295)

    # Checkpoint settings
    p.trainer.ckpt_save_frequency = 1
    p.trainer.restore_from_ckpt = False
    # p.trainer.num_epochs = 5
    p.trainer.num_epochs = 300

    # Change the Data Processing parameters
    p.data_processing.input_processing_function = 'resnet50_keras_preprocessing_and_distortion'

    # Input processing parameters
    p.data_processing.input_processing_params = DotMap(
        p=0.1,  # Probability of distortion
        version='v1'  # For new FOV (Intel Realsense D435, use version='v3')
    )

    # Checkpoint directory (choose which checkpoint to test):
    # oldcost WayPtNav
    # p.trainer.ckpt_path = '/home/anjianl/Desktop/project/WayPtNav/data/pretrained_weights/WayPtNav/session_2019-01-27_23-32-01/checkpoints/ckpt-9'
    # reachability WayPtNav
    #p.trainer.ckpt_path = '/local-scratch/tara/project/WayPtNav-reachability/Database/LB_WayPtNav_Data/pretrained_weights/WayPtNav/session_2019-01-27_23-32-01/checkpoints/ckpt-9'
    # p.trainer.ckpt_path = '/local-scratch/tara/project/WayPtNav-reachability/log/train/session_2020-09-10_10-14-56/checkpoints/ckpt-5' #Test last neural network for example, check point is nn parameters
    # p.trainer.ckpt_path = '/local-scratch/tara/project/WayPtNav-reachability/log/train/session_2021-06-20_14-56-40/checkpoints/ckpt-5'
    # p.trainer.ckpt_path = '/local-scratch/tara/project/WayPtNav-reachability/log/train/session_2022-06-22_16-56-28/checkpoints/ckpt-5'
    ## p.trainer.ckpt_path = "/local-scratch/tara/project/WayPtNav-reachability/log/train/session_2023-05-08_15-36-57/checkpoints_g1.0000_c1//ckpt-30"
    # p.trainer.ckpt_path ="/local-scratch/tara/project/WayPtNav-reachability/log/train/session_2023-03-06_23-09-09/checkpoints/ckpt-4"
    p.trainer.ckpt_path = "/local-scratch/tara/project/WayPtNav-reachability-master-Anjian/log/generate/session_2023-02-27_15-44-14/checkpoints/ckpt-10"
    # p.trainer.ckpt_path ="/local-scratch/tara/project/WayPtNav-reachability/log/train/session_2023-09-27_13-41-12/checkpoints_g1.0000_c1/ckpt-1"
    # p.trainer.ckpt_path ="/local-scratch/tara/project/WayPtNav-reachability/log/train/session_2023-11-02_21-58-48/checkpoints_g1.0000_c1/ckpt-4"
    # Change the data_dir
    # TODO: data dir name is a hack. Allowable name is xxx/area3/xxxx. The second last name
    #  should be the building name
    # TODO: In training, we have to render images for each area individually. That is,
    #   we will have only one area in data_dir each time, run training script, until the metadata is generated. After
    #   finishing this, we uncomment all data directories of all areas and start training together
    # reachability data (with disturbances)
    # p.data_creation.data_dir = [
    #     '/home/anjianl/Desktop/project/WayPtNav/data/successful_data/v2_no_filter_obstacle/area3/success_v2_50k', #last number shows number of datapoints
    #     '/home/anjianl/Desktop/project/WayPtNav/data/successful_data/v2_no_filter_obstacle/area4/success_v2_13k',
    #     '/home/anjianl/Desktop/project/WayPtNav/data/successful_data/v2_no_filter_obstacle/area4/success_v2_23k',
    #     '/home/anjianl/Desktop/project/WayPtNav/data/successful_data/v2_no_filter_obstacle/area4/success_v2_56k',
    #     '/home/anjianl/Desktop/project/WayPtNav/data/successful_data/v2_no_filter_obstacle/area5a/success_v2_8k',
    #     '/home/anjianl/Desktop/project/WayPtNav/data/successful_data/v2_no_filter_obstacle/area5a/success_v2_44k',
    #     '/home/anjianl/Desktop/project/WayPtNav/data/successful_data/v2_no_filter_obstacle/area5a/success_v2_45']
    # tmp test
    #data generation directory
    # p.data_creation.data_dir = ['/local-scratch/tara/project/WayPtNav-reachability/Database/LB_WayPtNav_Data/Generated-Data/area3/0622']
    # p.data_creation.data_dir = ['/local-scratch/tara/project/WayPtNav-reachability/Database/LB_WayPtNav_Data/Generated-Data/area3/0622_1']#area3/run date,  for us it can be train and test on the same hare so far
    # p.data_creation.data_dir = ['/local-scratch/tara/project/WayPtNav-reachability/Database/LB_WayPtNav_Data/Generated-Data/area3/0927']
    #    '/local-scratch/tara/project/WayPtNav-reachability/Database/LB_WayPtNav_Data/sbpd_projected_grid_include_last_step_successful_goals_only/area3/full_episode_random_v1_100k'] #they have training data as well
    #    '/local-scratch/tara/project/WayPtNav-reachability/Database/LB_WayPtNav_Data/sbpd_projected_grid_include_last_step_successful_goals_only / area4 / full_episode_random_v1_100k',
    #    '/local-scratch/tara/project/WayPtNav-reachability/Database/LB_WayPtNav_Data/sbpd_projected_grid_include_last_step_successful_goals_only / area5a / full_episode_random_v1_100k']
    # p.data_creation.data_dir = ['/local-scratch/tara/project/WayPtNav-reachability/Database/LB_WayPtNav_Data/Generated-Data/area3/tmp4-seperate4']
    # p.data_creation.data_dir = [
    #     '/local-scratch/tara/project/WayPtNav-reachability/Database/LB_WayPtNav_Data/Generated-Data/area3/0729-stack3-last-10b']
    # p.data_creation.data_dir = ['/local-scratch/tara/proiniynject/WayPtNav-reachability/Database/LB_WayPtNav_Data/Generated-Data/area3/1003-after-shuffling2']
    # p.data_creation.data_dir = [
    #     '/local-scratch/tara/project/WayPtNav-reachability/Database/LB_WayPtNav_Data/Generated-Data/area3/1115-SVM4-all1'] #test 2 datapoint IN EACH FILE
    # p.data_creation.data_dir = [
    #     '/local-scratch/tara/project/WayPtNav-reachability/Database/LB_WayPtNav_Data/Generated-Data/area3/1115-SVM4-easy']
    # p.data_creation.data_dir = [
    #     '/local-scratch/tara/project/WayPtNav-reachability/Database/LB_WayPtNav_Data/Generated-Data/area3/0215-more-toy-circle-gauss0.0001-lesswp']#1117-600
    #
    # p.data_creation.data_dir = [
    #     '/local-scratch/tara/project/WayPtNav-reachability/Database/LB_WayPtNav_Data/Generated-Data/area3/0222-rbf-gridsearch']#1117-600
    #    # p.data_creation.data_dir = ['/local-scratch/tara/project/WayPtNav-reachability-master-Anjian/Database/LB_WayPtNav_Data/Generated-Data/area3/0217-30wp-theta0bin1']
    # p.data_creation.data_dir = [
    #     '/local-scratch/tara/project/WayPtNav-reachability-master-Anjian/Database/LB_WayPtNav_Data/Generated-Data/area3/0222-30wp-two groups']
    p.data_creation.data_dir = [
            # '/local-scratch/tara/project/WayPtNav-reachability/Database/LB_WayPtNav_Data/Generated-Data/area4/anjian0-newv6_FRS-M-5image-4d-sample-spline200-2-safe2-sure-dv-wodis-wslack-futher33-v1',
            '/local-scratch/tara/project/WayPtNav-reachability/Database/LB_WayPtNav_Data/Generated-Data/area3/anjian0-newv6_FRS-M-5image-4d-sample-spline200-2-safe2-sure-dv-wodis-wslack-futher30-v1-10samples-samev'
            # '/local-scratch/tara/project/WayPtNav-reachability/Database/LB_WayPtNav_Data/Generated-Data/area4/anjian0-newv6_FRS-M-5image-4d-sample-spline200-2-safe2-sure-dv-wodis-wslack-futher33-v2',
    ]    # / localscratch / ttoufigh/ anjian0-newv6_FRS-1/

    # p.data_creation.data_dir = ['/local-scratch/tara/project/WayPtNav-reachability-master-Anjian/Database/LB_WayPtNav_Data/Generated-Data/area3/anjian0-newv-nearest-1.5']

    p.data_creation.data_points = 75_000# 250000
    p.data_creation.data_points_per_file = int(5) # in each pickle file, so 1000/100=10 .pkl files, pickle holds coordinates
    # Seed for selecting the test scenarios and the number of such scenarios
    p.test.seed = 10
    p.test.number_tests = 200
    # p.test.number_tests = 5 #number of test samples for my first test

    # Test the network only on goals where the expert succeeded
    p.test.expert_success_goals = DotMap(use=False,
                                         dirname='/local-scratch/tara/project/WayPtNav-reachability/Database/LB_WayPtNav_Data/expert_success_goals/sbpd_projected_grid')


    # Let's not look at the expert
    p.test.simulate_expert = False

    # Parameters for the metric curves
    p.test.metric_curves = DotMap(start_ckpt=1,
                                  end_ckpt=10,
                                  start_seed=1,
                                  end_seed=10,
                                  plot_curves=True
                                  )

    return p
