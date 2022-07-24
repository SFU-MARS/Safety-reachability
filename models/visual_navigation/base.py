import tensorflow as tf
import numpy as np
from copy import deepcopy

from models.base import BaseModel
from training_utils.architecture.simple_cnn import simple_cnn
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.initializers import glorot_uniform
import numpy as np
from keras import layers
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from models.visual_navigation.resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow

import keras.backend as K
# tf.compat.v1.disable_eager_execution()


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

    def create_nn_inputs_and_outputs(self, raw_data, is_training=None):
        """
        Create the occupancy grid and other inputs for the neural network.
        """
        
        # if self.p.data_processing.input_processing_function is not None:
        #     data = self.preprocess_nn_input(raw_data, is_training)

        # Get the input image (n, m, k, d)
        # batch size n x (m x k pixels) x d channels
        # img_nmkd = raw_data['img_nmkd']
        # raw_data['flatted']=data
        img_nmkd = raw_data['image']


        # ###

        # X_input = Input(img_nmkd.shape())
        #
        # # Zero-Padding
        # X = ZeroPadding2D((3, 3))(X_input)
        #
        # # Stage 1
        # X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
        # X = BatchNormalization(axis=3, name='bn_conv1')(X)
        # X = Activation('relu')(X)
        # X = MaxPooling2D((3, 3), strides=(2, 2))(X)
        #
        # # Stage 2
        # X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
        # X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
        # X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
        #
        # ### START CODE HERE ###
        #
        # # Stage 3 (≈4 lines)
        # X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
        # X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
        # X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
        # X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')
        #
        # # Stage 4 (≈6 lines)
        # X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
        # X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
        # X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
        # X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
        # X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
        # X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
        #
        # # Stage 5 (≈3 lines)
        # X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
        # X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
        # X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')
        #
        # # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
        # X = AveragePooling2D((2, 2), name="avg_pool")(X)



        # output layer
        # X = Flatten()(X)
        # # X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
        #
        # raw_data['flatted']=X

        ###
### END CODE HERE ###
        # Concatenate the goal position in an egocentric frame with vehicle's speed information
        # goal_position = self._goal_position(raw_data)
        # vehicle_controls = self._vehicle_controls(raw_data)
        waypointAction = raw_data['waypointAction']

        vehicle_state = raw_data['start_pose']
        # state_features_n4 = tf.concat([goal_position, vehicle_controls], axis=1)
        # state_features_n4 = np.array(tf.concat([vehicle_state, vehicle_controls], axis=1))
        # state_features_n4 = np.array(vehicle_state)
        state_features_n2= np.squeeze(vehicle_state)
        # Optimal Supervision
        # optimal_labels_n = self._optimal_labels(raw_data)
        optimal_labels_n = raw_data['labels']
        # Prepare and return the data dictionary
        data = {}
        data['inputs'] = [img_nmkd, state_features_n2]
        data['labels'] = optimal_labels_n
        data['Action_waypoint'] = np.squeeze(waypointAction)

        return data
    
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
            # raw_data['img_nmkd'] = tf.keras.applications.resnet50.preprocess_input(raw_data['img_nmkd'], mode='caffe')
            raw_data['image'] = tf.keras.applications.resnet50.preprocess_input(raw_data['image'], mode='caffe')
            # raw_data['image'] = np.squeeze(raw_data['image'])

        # Normalize images if required
        if self.p.data_processing.input_processing_function in ['normalize_images', 'normalize_distort_images']:
            from training_utils.data_processing.normalize_images import rgb_normalize
            raw_data = rgb_normalize(raw_data)

        if self.p.data_processing.input_processing_function in \
                ['resnet50_keras_preprocessing', 'resnet50_keras_preprocessing_and_distortion']:
            raw_data['image'] = tf.keras.applications.resnet50.preprocess_input(raw_data['image'], mode='caffe')

        return raw_data


#Tara
        # ###
        # # tf.compat.v1.disable_eager_execution()
        # img_rows = raw_data['image'].shape[0]
        # img_cols = raw_data['image'].shape[1]
        # Input_shape = (img_rows, img_cols, 3)
        # X_input = Input(Input_shape)
        # print(X_input)
        # # X_input=raw_data['image']
        # # Zero-Padding
        # X = ZeroPadding2D((3, 3))(X_input)
        #
        # # Stage 1
        # X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
        # X = BatchNormalization(axis=3, name='bn_conv1')(X)
        # X = Activation('relu')(X)
        # X = MaxPooling2D((3, 3), strides=(2, 2))(X)
        #
        # # Stage 2
        # X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
        # X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
        # X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
        #
        # ### START CODE HERE ###
        #
        # # Stage 3 (≈4 lines)
        # X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
        # X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
        # X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
        # X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')
        #
        # # Stage 4 (≈6 lines)
        # X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
        # X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
        # X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
        # X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
        # X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
        # X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
        #
        # # Stage 5 (≈3 lines)
        # X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
        # X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
        # X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')
        #
        # # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
        # # X = AveragePooling2D((2, 2), name="avg_pool")(X)
        # X = GlobalMaxPooling2D()(X)
        # ### END CODE HERE ###
        #
        # # output layer
        # # X = Flatten()(X)
        # # data={}
        # raw_data['flattened']=X
        # return raw_data
        return raw_data