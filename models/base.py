import tensorflow as tf
from training_utils.architecture.simple_mlp import simple_mlp
# from training_utils.architecture.resnet50.resnet_50 import ResNet50
# from training_utils.architecture.resnet50_cnn import resnet50_cnn
# from "@tensorflow/tfjs" import * as tf


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
        nn_output = self.predict_nn_output(processed_data['inputs'], is_training=is_training)
        
        if self.p.loss.loss_type == 'mse':
            prediction_loss = tf.losses.mean_squared_error(nn_output, processed_data['labels'])
        elif self.p.loss.loss_type == 'l2_loss':
            prediction_loss = tf.nn.l2_loss(nn_output - processed_data['labels'])
        elif self.p.loss.loss_type == 'hinge':
            import numpy as np
            nn_output1=np.zeros((60,5))
            prediction_loss1=np.zeros((60,1))
            prediction_loss0 = np.zeros((60, 1))
            epochs=10
            rate=1/epochs
            reg_parm=0.1

            # for epoch in range(1, epochs):
            #     # for n, data in enumerate(processed_data['Action_waypoint'] ):
            #     for i in range (0,60):
            #         if (processed_data['labels'] [0][:,i]* np.dot(tf.concat((tf.transpose(processed_data['Action_waypoint'][0][:,i]), tf.ones(1)),axis=0), nn_output[0])< 1):
            #             # nn_output1[i,:] = np.transpose(np.array(nn_output))[:,i] + rate * ((processed_data['Action_waypoint'][i] * processed_data['labels'][i]) + (-2 * (1 / epoch) * np.transpose(np.array(nn_output))[:,i]))
            #
            #             # nn_output1[i,:] = (1 - rate)*np.transpose(np.array(nn_output))[:,i]+rate*reg_parm*(processed_data['Action_waypoint'][i] * processed_data['labels'][i])
            #             nn_output1[i, :] = (1 - rate) * tf.transpose(np.array(nn_output)[0]) + rate * reg_parm * (
            #                 processed_data['labels'][0][:,i])*tf.concat((tf.transpose(processed_data['Action_waypoint'][0][:,i]), tf.ones(1)) ,axis=0)
            #         else:
            #             # nn_output1[i,:]= np.transpose(np.array(nn_output))[:,i] + rate * (-2 * (1 / epoch) * np.transpose(nn_output)[:,i])
            #             nn_output1[i,:] = (1 - rate) * tf.transpose(nn_output)[0]

            # return nn_output1

        # for i in range(0, 60):
        #     prediction_loss1[i]  = np.maximum(0.,1-np.dot(tf.transpose(np.transpose(processed_data['Action_waypoint'][0])[i], tf.ones(1)),axis=0), np.transpose(nn_output)))
        import numpy as np
        # ywxmax=tf.zeros(8,1)
        output_list=[]
        # for i in range(60):
        #     x = tf.contrib.eager.Variable(tf.random_normal([4, 60], mean=1.0, stddev=0.35))
        #     # tf.assign(x, np.squeeze(processed_data['Action_waypoint'][0:4, :, :]))
        #     tf.assign(x, processed_data['Action_waypoint'][i, :])
        #     x_trans = tf.transpose(x)
        #     x_conc = tf.concat((x_trans, tf.ones((60, 1))), axis=1)
        #     w = tf.transpose(nn_output)[:, i]
        #     z = tf.matmul(tf.reshape(w, (1, 5)), tf.transpose(x_conc))
        #     # y = tf.reshape(tf.convert_to_tensor(processed_data['labels'][i, :]), (60, 1))
        #     y = tf.reshape(tf.convert_to_tensor(processed_data['labels'][i, :,:]), (60, 1))
        #     y1 = tf.cast(y,tf.float32)
        #     output_list.append(tf.maximum(0,1-tf.matmul(z, y1)))
        for i in range(self.p.trainer.batch_size):

            x = tf.contrib.eager.Variable(tf.random_normal([4], mean=1.0, stddev=0.35))
            # tf.assign(x, np.squeeze(processed_data['Action_waypoint'][0:4, :, :]))
            tf.assign(x, processed_data['Action_waypoint'][i, :])
            x1 = tf.reshape(x, (4, 1))
            # x_trans = tf.transpose(x)
            x_conc = tf.concat((x1, tf.ones((1, 1))), axis=0)
            w = nn_output
            z = tf.matmul(w, x_conc)
            # y = tf.reshape(tf.convert_to_tensor(processed_data['labels'][i, :]), (60, 1))
            y = tf.convert_to_tensor(processed_data['labels'][i, :, :])
            y1 = tf.cast(y, tf.float32)
            output_list.append(tf.maximum(0, 1 - tf.matmul(z, y1)))

        ywxmax=tf.stack(output_list)

        # x=tf.ones((60,1))-tf.matmul(tf.matmul(tf.concat([tf.transpose(processed_data['Action_waypoint'][0]), tf.ones((60,1))],axis=1), tf.transpose(nn_output)), processed_data['labels'][0])
        # y=tf.maximum(tf.zeros((60,1)),x)
        prediction_loss1=tf.reduce_sum(ywxmax)

        regularization_loss = tf.nn.l2_loss(nn_output)

        C=1 #Penalty parameter of the error term
        total_loss = C*(prediction_loss1)+ 0.5 * regularization_loss
       
        if return_loss_components_and_output:
            return regularization_loss, prediction_loss1, total_loss, nn_output
        elif return_loss_components:
            return regularization_loss, prediction_loss1, total_loss
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
