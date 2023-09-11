import tensorflow as tf
import numpy as np
from copy import deepcopy

from models.base import BaseModel
from training_utils.architecture.simple_cnn import simple_cnn

class VisualNavigationModelBase(BaseModel):
    """
    A model used for navigation that receives, among other inputs,
    an image as its observation of the environment.
    """
    
    def make_architecture(self):
        """
        Create the CNN architecture for the model.
        """
        self.arch = simple_cnn(image_size=self.p.model.num_inputs.image_size,
                               num_inputs=self.p.model.num_inputs.num_state_features,
                               num_outputs=self.p.model.num_outputs,
                               params=self.p.model.arch)

    def _optimal_labels(self, raw_data):
        """
        Return the optimal label based on raw_data.
        """
        raise NotImplementedError

    def _goal_position(self, raw_data):
        """
        Return the goal position (x, y) in egocentric
        coordinates.
        """
        return raw_data['goal_position_ego_n2']
   
    def _vehicle_controls(self, raw_data):
        """
        Return the vehicle linear and angular speed.
        """
        return raw_data['vehicle_controls_nk2'][:, 0]

    def _wp_speed(self, raw_data):
        """
        Return the vehicle linear and angular speed.
        """
        return raw_data['vehicle_controls_nk2'][:, -1]

    def create_nn_inputs_and_outputs(self, raw_data, is_training=None):
        """
        Create the occupancy grid and other inputs for the neural network.
        """

        
        if self.p.data_processing.input_processing_function is not None:
            raw_data = self.preprocess_nn_input(raw_data, is_training)
        #their code
        # Get the input image (n, m, k, d)
        # batch size n x (m x k pixels) x d channels
    
        img_nmkd_d = raw_data['img_nmkd']
        img_nmkd = img_nmkd_d [:,:,:,:3]

        # Concatenate the goal position in an egocentric frame with vehicle's speed information
        # goal_position = self._goal_position(raw_data)
        vehicle_controls = self._vehicle_controls(raw_data)
        # state_features_n4 = tf.concat([goal_position, vehicle_controls], axis=1)

        # Optimal Supervision
        optimal_labels_n = self._optimal_labels(raw_data)

        # Prepare and return the data dictionary
        data = {}

        # state_featres_n2 = vehicle_controls[:,0]
        # state_featres_n2 = raw_data['vehicle_state_nk3'][:, 0, -1]
        state_featres_n2 = raw_data['vehicle_state_nk3'][:, 0, 0, -1]
        state_features_n21 = np.reshape(state_featres_n2, (-1, 1))
        # Action_waypoint = raw_data['all_waypoint_ego']
        data['inputs'] = [img_nmkd, state_features_n21]
        # data['inputs'] = [img_nmkd]
        # data['inputs'] = [img_nmkd, Action_waypoint]
        data['labels'] = raw_data['labels']
        # speeds = self._wp_speed(raw_data)[:,0:1]
        data['Action_waypoint'] = raw_data['all_waypoint_ego']
        # data['Action_waypoint_withv'] = np.concatenate(raw_data['all_waypoint_ego'], speeds, axis=1)
        return data

        #my code data generation
        #
        # img_nmkd = raw_data['image']
        # # img_nmkd = raw_data['img_nmkd']
        # # img_nmkd=(img_nmkd - img_nmkd.mean(axis=0)) / (img_nmkd.std(axis=0) + 1e-8)
        # img_nmkd1=np.expand_dims(img_nmkd, axis=0)
        #
        # waypointAction = raw_data['waypointAction']
        #
        # waypointAction1 = np.expand_dims(waypointAction, axis=0)
        # # x=[]
        # # frames = np.empty( [1, 50,224,224,3])
        # #
        # # for i in range(50):
        # #     x=waypointAction[i]
        # #     frames[:,i, :, :, :]= x
        #
        # vehicle_state = raw_data['start_pose']
        # # np.mean(self.training_info_dict['data']['start_pose'], axis=0) #
        # # mean_t= [[0.3496886,  0.06099968]]
        # # np.var(self.training_info_dict['data']['start_pose'], axis=0) #
        # # var_t= [[0.02119822, 3.3798633]]
        # # vehicle_state= (vehicle_state - mean_t)/var_t
        # # vehicle_state_max = np.reshape([0.6,3.14], (1,1,2))
        # # vehicle_state_min = np.reshape([0,-3.14] , (1,1,2))
        # # # vehicle_state = (vehicle_state - vehicle_state_min) / (vehicle_state_max - vehicle_state_min)
        # # vehicle_state = vehicle_state.astype(dtype=np.float32)
        #
        # # state_features_n4 = tf.concat([goal_position, vehicle_controls], axis=1)
        # # state_features_n4 = np.array(tf.concat([vehicle_state, vehicle_controls], axis=1))
        # # state_featuures_n4 = np.array(vehicle_state)
        # state_featres_n2 = np.squeeze(vehicle_state)
        # state_features_n21 = np.reshape(state_featres_n2,(-1,2))
        # # state_features_n2=tf.convert_to_tensor(state_features_n2)
        # # state_features_n21=np.expand_dims(state_features_n2, axis=0)
        # # Optimal Supervision
        # # optimal_labels_n = self._optimal_labels(raw_data)
        # optimal_labels_n = raw_data['labels']
        # optimal_labels_n1=np.expand_dims(optimal_labels_n, axis=0)
        #
        # # Prepare and return the data dictionary
        # data = {}
        #
        # # normalized_img = img_nmkd /255
        #
        # data['inputs'] = [img_nmkd, state_features_n21]
        #
        # data['labels'] = optimal_labels_n
        # data['Action_waypoint'] = waypointAction
        # # data['Action_waypoint'] = np.expand_dims((np.squeeze(waypointAction)), axis=0)
        # return data

    def make_processing_functions(self):
        """
        Initialize the processing functions if required.
        """
        
        # Initialize the distortion function
        if self.p.data_processing.input_processing_function in ['distort_images', 'normalize_distort_images',
                                                                'resnet50_keras_preprocessing_and_distortion']:
            from training_utils.data_processing.distort_images import basic_image_distortor
            self.image_distortor = basic_image_distortor(self.p.data_processing.input_processing_params)
        else:
            # Add this assert here to make sure the input processing function isn't
            # accidently misspelt
            assert(self.p.data_processing.input_processing_function in ['normalize_images',
                                                                        'resnet50_keras_preprocessing'])

    def preprocess_nn_input(self, raw_data, is_training):
        """
        Pre-process the NN input.
        """
        raw_data = deepcopy(raw_data)

        if is_training:
            # Distort images if required
            if self.p.data_processing.input_processing_function in ['distort_images', 'normalize_distort_images',
                                                                    'resnet50_keras_preprocessing_and_distortion']:
                # Change the field-of-view and tilt if required
                if self.p.data_processing.input_processing_params.version in ['v3']:
                    raw_data['img_nmkd'] = self.image_distortor[1](raw_data['img_nmkd'])
                # Image Augmenter works with uint8, but we want images to be float32 for the network, hence the casting
                # raw_data['img_nmkd'] = \ self.image_distortor[0].augment_images(raw_data['img_nmkd'].astype(np.uint8)).astype(np.float32)
                    raw_data['image'] = \
                        self.image_distortor[0].augment_images(raw_data['image'].astype(np.uint8)).astype(np.float32)
        
        # Normalize images if required
        if self.p.data_processing.input_processing_function in ['normalize_images', 'normalize_distort_images']:
            from training_utils.data_processing.normalize_images import rgb_normalize
            raw_data = rgb_normalize(raw_data)

        if self.p.data_processing.input_processing_function in \
                ['resnet50_keras_preprocessing', 'resnet50_keras_preprocessing_and_distortion']:
            raw_data['img_nmkd'] = tf.keras.applications.resnet50.preprocess_input(raw_data['img_nmkd'], mode='caffe')
            # raw_data['image'] = tf.keras.applications.resnet50.preprocess_input(raw_data['image'], mode='caffe')
            # raw_data['image'] = np.squeeze(raw_data['image'])


        return raw_data


        # return raw_data