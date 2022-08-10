import tensorflow as tf
from training_utils.architecture.simple_mlp import simple_mlp
# from training_utils.architecture.resnet50.resnet_50 import ResNet50
# from training_utils.architecture.resnet50_cnn import resnet50_cnn


class BaseModel(object):
    """
    A base class for an input-output model that can be trained.
    """
    
    def __init__(self, params):
        self.p = params
        self.make_architecture()
        self.make_processing_functions()
        
    def make_architecture(self):
        """
        Create the NN architecture for the model.
        """
        self.arch = simple_mlp(num_inputs=self.p.model.num_inputs,
                               num_outputs=self.p.model.num_outputs,
                               params=self.p.model.arch)
    
    def compute_loss_function(self, raw_data, is_training=None, return_loss_components=False,
                              return_loss_components_and_output=False):
        """
        Compute the loss function for a given dataset.
        """
        # Create the NN inputs and labels
        processed_data  = self.create_nn_inputs_and_outputs(raw_data, is_training=is_training)
        # x=processed_data['inputs'][0]
        #     # Convolutional layer
        # layers = tf.keras.layers
        # x = layers.Conv2D(
        #     filters=self.p.dim_red_conv_2d.num_outputs,
        #     kernel_size=self.p.dim_red_conv_2d.filter_size,
        #     strides=self.p.dim_red_conv_2d.stride,
        #     padding=self.p.dim_red_conv_2d.padding,
        #     # activation=tf.keras.activations.relu)(x)
        #     activation=self.p.hidden_layer_activation_func)(x)
        #     # Max-pooling layer
        # x = layers.MaxPool2D(pool_size=(self.p.dim_red_conv_2d.size_maxpool_filters,
        #                                     self.p.dim_red_conv_2d.size_maxpool_filters),
        #                          padding='valid')(x)
        #
        # # Flatten the image
        # x = layers.Flatten()(x)
        # input_flat=processed_data['input'][1]
        # x = layers.Concatenate(axis=1)([x, input_flat])
        #

        # # Predict the NN output

        nn_output = self.predict_nn_output(processed_data['inputs'], is_training=is_training)
        
        # Compute the regularization loss, prediction loss and the total loss
        regularization_loss = 0.

        # model_variables = self.get_trainable_vars()
        # for model_variable in model_variables:
        #     regularization_loss += tf.nn.l2_loss(model_variable)
        # regularization_loss = self.p.loss.regn * regularization_loss
        
        if self.p.loss.loss_type == 'mse':
            prediction_loss = tf.losses.mean_squared_error(nn_output, processed_data['labels'])
        elif self.p.loss.loss_type == 'l2_loss':
            prediction_loss = tf.nn.l2_loss(nn_output - processed_data['labels'])
        elif self.p.loss.loss_type == 'hinge':
            import numpy as np

            epochs=10
            rate=1
            for epoch in range(1, epochs):
                for n, data in enumerate(processed_data['inputs']):
                    if (processed_data['labels'] * np.dot(processed_data['inputs'], np.array(nn_output))) < 1:
                        nn_output = np.array(nn_output) + rate * ((processed_data['inputs'] * processed_data['labels']) + (-2 * (1 / epoch) * np.array(nn_output)))
                    else:
                        nn_output= np.array(nn_output) + rate * (-2 * (1 / epoch) * np.array(nn_output))

            return nn_output
        # prediction_loss = tf.math.maximum(0, 1- processed_data['labels']*(nn_output[1:3]* Action_waypoint -nn_output[0]) )
        #     prediction_loss = tf.math.maximum(0, 1 - processed_data['labels'] *np.dot(np.concatenate(processed_data['Action_waypoint'],np.ones((self.p.trainer.batch_size,1)), np.transpose(np.array(nn_output)))))
            prediction_loss = tf.reduce_mean(tf.math.maximum(0.,  1.-processed_data['labels']* (np.dot(processed_data['Action_waypoint'],np.transpose(np.array(nn_output))))))
        #     prediction_loss = tf.math.maximum(1, np.transpose(processed_data['labels']) * np.dot(processed_data['Action_waypoint'],
        #                                                                            np.transpose(np.array(nn_output))))


            prediction_loss1=np.sum(np.array(prediction_loss))

            regularization_loss = tf.nn.l2_loss(nn_output)

            C=1 #Penalty parameter of the error term
            total_loss = C*prediction_loss1 + 0.5 * regularization_loss
            # prediction_loss = -(1 - processed_data['labels'] * nn_output*processed_data['flatted'])

            # total_loss = prediction_loss + regularization_loss
        else:
            raise NotImplementedError
        
        # total_loss = prediction_loss + regularization_loss

       
        if return_loss_components_and_output:
            return regularization_loss, prediction_loss, total_loss, nn_output
        elif return_loss_components:
            return regularization_loss, prediction_loss, total_loss
        else:
            return total_loss
    
    def get_trainable_vars(self):
        """
        Get a list of the trainable variables of the model.
        """
        return self.arch.variables
    
    def create_nn_inputs_and_outputs(self, raw_data, is_training=None):
        """
        Create the NN inputs and outputs from the raw data batch. All pre-processing should go here.
        """
        raise NotImplementedError
    
    def predict_nn_output(self, data, is_training=None):
        """
        Predict the NN output to a given input.
        """
        assert is_training is not None
        
        if is_training:
            # Use dropouts
            tf.keras.backend.set_learning_phase(1)
        else:
            # Do not use dropouts
            tf.keras.backend.set_learning_phase(0)
        
        return self.arch.predict_on_batch(data)

    def predict_nn_output_with_postprocessing(self, data, is_training=None):
        """
        Predict the NN output to a given input with an optional post processing function
        applied. By default there is no post processing function applied.
        """
        return self.predict_nn_output(data, is_training=is_training)

    def make_processing_functions(self):
        """
        Initialize the processing functions if required.
        """
        return
