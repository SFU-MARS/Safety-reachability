import tensorflow as tf
from training_utils.architecture.simple_mlp import simple_mlp
# from training_utils.architecture.resnet50.resnet_50 import ResNet50
# from training_utils.architecture.resnet50_cnn import resnet50_cnn
# from "@tensorflow/tfjs" import * as tf
import numpy as np

K = tf.keras.backend


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

        processed_data = self.create_nn_inputs_and_outputs(raw_data, is_training=is_training)
        # processed_data = tf.Variable(processed_data)

        nn_output = self.predict_nn_output(processed_data['inputs'], is_training=is_training)

        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        import numpy as np
        x = np.expand_dims(processed_data['Action_waypoint'][0][:, 0], axis=0)
        y = np.expand_dims(processed_data['Action_waypoint'][0][:, 1], axis=0)
        z = np.expand_dims(processed_data['Action_waypoint'][0][:, 2], axis=0)
        normal = np.array(nn_output)[0][:-1]
        C = [.25, .25, 1, 0, .75]
        d = np.array(nn_output)[0][-1]
        xx, yy = np.meshgrid(range(5), range(5))
        z1 = (-normal[0] * xx - normal[1] * yy - d) / normal[2]
        # fig = plt.figure()
        ax = plt.axes(projection="3d")
        # ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(x, y, z, c='red')
        # plt3d = plt.figure().gca(projection='3d')
        ax.plot_surface(xx, yy, z1)
        plt.title("3D Scatter Plot")
        d1 = C[-1]
        z2 = (-C[0] * xx - C[1] * yy - d1) / C[2]
        ax.plot_surface(xx, yy, z2)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        # plt.show()
        # plt3d.plot_surface(xx, yy, z1)
        plt.savefig("demo2.png")

        # regularization_loss = 0.
        # model_variables = self.get_trainable_vars()
        # for model_variable in model_variables:
        #     regularization_loss += tf.nn.l2_loss(model_variable)
        # regularization_loss = self.p.loss.regn * regularization_loss
        # processed_data['labels']=np.tile([.25,.25,0,0,.75], (6,1))
        if self.p.loss.loss_type == 'mse':
            prediction_loss = tf.losses.mean_squared_error(nn_output, processed_data['labels'])
            regularization_loss1 = tf.nn.l2_loss(nn_output)

            C = 1  # Penalty parameter of the error term

            total_loss = prediction_loss + 0.5 * regularization_loss1
        elif self.p.loss.loss_type == 'l2_loss':
            prediction_loss = tf.nn.l2_loss(nn_output - processed_data['labels'])
        elif self.p.loss.loss_type == 'hinge':
            # import numpy as np
            # nn_output1=np.zeros((60,5))
            # prediction_loss1=np.zeros((60,1))
            # prediction_loss0 = np.zeros((60, 1))
            # epochs=10
            # rate=1/epochs
            # reg_parm=0.1
            # output_list=[]
            # countT=0
            #

            # for i in range(self.p.trainer.batch_size):
            #
            #     x = tf.contrib.eager.Variable(tf.random_normal([4], mean=1.0, stddev=0.35))
            #     # tf.assign(x, np.squeeze(processed_data['Action_waypoint'][0:4, :, :]))
            #     tf.assign(x, processed_data['Action_waypoint'][i, :])
            #     x1 = tf.reshape(x, (4, 1))
            #     # x_trans = tf.transpose(x)
            #     x_conc = tf.concat((x1, tf.ones((1, 1))), axis=0)
            #     w = tf.convert_to_tensor(nn_output)
            #     z = tf.matmul(w, x_conc)
            #     new_predicted = np.array([-1 if i == 0 else i for i in z])
            #     # y = tf.reshape(tf.convert_to_tensor(processed_data['labels'][i, :]), (60, 1))
            #     y = tf.convert_to_tensor(processed_data['labels'][i, :, :])
            #     y1 = tf.cast(y, tf.float32)
            #     output_list.append(tf.maximum(0, 1 - tf.matmul(z,  y1)))
            #     if tf.matmul(z, y1) >= 0:
            #         countT+=1
            from tensorflow.python.framework import ops
            # with ops.name_scope(scope, "hinge_loss", (nn_output, processed_data)) as scope:
            # t=np.tile(nn_output,(data[1].shape[0],1,1))
            # data[0]=t.reshape(data[1].shape[0],224,224,3)
            # t = tf.tile(nn_output, (50, 1))
            # processed_data['Action_waypoint'] = processed_data['Action_waypoint'] / np.linalg.norm(processed_data['Action_waypoint'])

            x = K.concatenate(
                (processed_data['Action_waypoint'][0], tf.ones((processed_data['Action_waypoint'][0].shape[0], 1))),
                axis=1)

            predicted = np.transpose(tf.expand_dims(K.dot(x, tf.transpose(nn_output)), axis=0))

            logits = math_ops.to_float(predicted)
            labels = math_ops.to_float(processed_data['labels'])
            logits.get_shape().assert_is_compatible_with(labels.get_shape())
            all_ones = array_ops.ones_like(labels)
            losses = nn_ops.relu(
                math_ops.subtract(all_ones, math_ops.multiply(labels, logits)))
            hinge_loss = math_ops.reduce_sum(losses)

            # hinge_loss = tf.keras.losses.hinge(K.flatten(predicted), K.flatten(processed_data['labels'][:50]))
            # ywxmax=tf.maximum(0, tf.ones(60, 1) - tf.matmul(x, w1))
            prediction_loss = hinge_loss * hinge_loss
            print(prediction_loss)

            # t = [y * wx for y, wx in zip(np.squeeze(processed_data['labels'][:50]), predicted)]
            # threshold = 1
            # elements_gt = tf.math.greater(t, threshold)
            # num_elements_gt = np.mean(tf.cast(elements_gt, tf.int32))
            # print('accuracy:' + str(num_elements_gt))
            # accuracy=num_elements_gt
            # accuracy =tf.reduce_mean(tf.matmul(predicted, processed_data['labels'])>=1)
            # ywxmax=tf.stack(output_list)

            # x=tf.ones((60,1))-tf.matmul(tf.matmul(tf.concat([tf.transpose(processed_data['Action_waypoint'][0]), tf.ones((60,1))],axis=1), tf.transpose(nn_output)), processed_data['labels'][0])
            # y=tf.maximum(tf.zeros((60,1)),x)
            # prediction_loss1 = tf.reduce_sum(ywxmax)

            regularization_loss1 = tf.nn.l2_loss(nn_output[:-1])

            C = 1  # Penalty parameter of the error term

            # total_loss = prediction_loss +  0.5 * regularization_loss1
            # print(total_loss)
            # total_loss = C*(prediction_loss1)+ 0.5 * tf.cast(regularization_loss,dtype=tf.float64)
            total_loss = C * (prediction_loss)
            # accuracy=countT/self.p.trainer.batch_size

        if return_loss_components_and_output:
            return regularization_loss1, prediction_loss, total_loss, nn_output
        elif return_loss_components:
            return regularization_loss1, prediction_loss, total_loss
        else:
            return total_loss
            # return regularization_loss

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
