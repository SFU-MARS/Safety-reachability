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

        processed_data  = self.create_nn_inputs_and_outputs(raw_data, is_training=is_training)
        # processed_data = tf.Variable(processed_data)

        nn_output = self.predict_nn_output(processed_data['inputs'], is_training=is_training)
        print("nn_output: "+str(nn_output.numpy()))
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure()

        ###
        # import numpy as np

        # x = np.expand_dims(processed_data['Action_waypoint'][0][:, 0], axis=0)
        # y = np.expand_dims(processed_data['Action_waypoint'][0][:, 1], axis=0)
        # y = processed_data['Action_waypoint'][0][:, 1]
        # z = np.expand_dims(processed_data['Action_waypoint'][0][:, 2], axis=0)
        # WP=processed_data['Action_waypoint'][0]
        # LABELS=processed_data['labels']
        # normal = np.array(nn_output)[0][:-1]
        # from sklearn.svm import SVC
        # clf = SVC(C=1e5, kernel='linear')
        # clf.fit(WP, LABELS[0])
        # w,b = clf.coef_, clf.intercept_
        # # x_points = np.linspace(-1, 1)  # generating x-points from -1 to 1
        # # y_points = -(w[0] / w[1]) * x_points - b / w[1]  # getting
        # # plt.plot(x_points, y_points, c='r')
        #
        # C = np.concatenate((w,np.reshape(b,(1,1))),axis=1)
        # # C = np.concatenate((w, np.reshape(b, (1))), axis=0)
        # C = np.transpose(C)
        # d = np.array(nn_output)[0][-1]
        #
        #
        # # b=np.array(nn_output)[0][-1]
        # xx, yy = np.meshgrid(range(5), range(5))
        # # y_points1 = -(C[0] / C[1]) * x_points - b / C[1]
        # z0 = (-normal[0] * xx - normal[1] * yy - d) / normal[2]
        # # fig = plt.figure()
        # ax1 = plt.axes(projection="3d")
        # ax1.grid(True)
        #
        # colors = ['red', 'green']
        # # ax = fig.add_subplot(111, projection='3d')
        # import matplotlib
        # ax1.scatter3D(x, y, z, c='red')
        # # ax.scatter2D(x, y, c='red')
        # # ax.scatter3D(x, y, z, c=LABELS[0], cmap=matplotlib.colors.ListedColormap(colors))
        # # plt3d = plt.figure().gca(projection='3d')
        # # plt.plot(x_points, y_points1, c='b')
        #
        # ax1.plot_surface(xx, yy, z0)
        # plt.title("nn vs correct for first")
        # d0 = C[-1]
        # zc1 = (-C[0] * xx - C[1] * yy - d0) / C[2]
        # ax1.plot_surface(xx, yy, zc1)
        #
        # for spine in ax1.spines.values():
        #     spine.set_visible(False)
        #
        # plt.show()
        #
        # normal1 = np.array(nn_output)[1][:-1]
        # d1 = np.array(nn_output)[1][-1]
        # z1 = (-normal1[0] * xx - normal1[1] * yy - d1) / normal1[2]
        # ax2 = plt.axes(projection="3d")
        # # fig = plt.figure()
        # ax2.grid(True)
        # # ax.scatter3D(x, y, z, c=LABELS[0], cmap=matplotlib.colors.ListedColormap(colors))
        # # plt3d = plt.figure().gca(projection='3d')
        # ax2.plot_surface(xx, yy, z1)
        #
        # from sklearn.svm import SVC
        # clf = SVC(C=1e5, kernel='linear')
        # clf.fit(WP, LABELS[1])
        # w1,b1=clf.coef_, clf.intercept_
        # C1 = np.concatenate((w1,np.reshape(b1,(1,1))),axis=1)
        # C1 = np.transpose(C1)
        # zc2 = (-C1[0] * xx - C1[1] * yy - d1) / C1[2]
        # ax1.plot_surface(xx, yy, zc2)
        #
        # ax2.set_xlabel('X Label')
        # ax2.set_ylabel('Y Label')
        # ax2.set_zlabel('Z Label')
        # plt.title("nn vs correct for second")
        # # The fix
        # for spine in ax2.spines.values():
        #     spine.set_visible(False)
        #
        #
        # # plt.tight_layout()
        # plt.show()
        # # plt3d.plot_surface(xx, yy, z1)
        # plt.savefig("demo2.png")

        ###

        # regularization_loss = 0.
        # model_variables = self.get_trainable_vars()
        # for model_variable in model_variables:
        #     regularization_loss += tf.nn.l2_loss(model_variable)
        # regularization_loss = self.p.loss.regn * regularization_loss
        # processed_data['labels']=np.tile([.25,.25,0,0,.75], (6,1))
        if self.p.loss.loss_type == 'mse':
            prediction_loss = tf.losses.mean_squared_error(nn_output, processed_data['labels'])
            regularization_loss1 = tf.nn.l2_loss(nn_output)

            C=1 #Penalty parameter of the error term

            total_loss = prediction_loss +  0.5 * regularization_loss1
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

            x = K.concatenate((processed_data['Action_waypoint'][0], tf.ones((processed_data['Action_waypoint'][0].shape[0], 1))), axis=1)
            # x = K.concatenate((processed_data['Action_waypoint'][:50], tf.ones((nn_output.shape[0], 1, 1))), axis=1)
            # x = K.reshape(x, (50, 5))
            # w = tf.convert_to_tensor(nn_output)
            # w = tf.reduce_mean(nn_output, axis=0)
            # w1 = tf.reshape(w, (5, 1))
            labels1=[]
            labels2 = []
            labels = processed_data['labels']
            labels1 = labels.tolist()
            for label in labels1:
                for l in label:
                    print (l)
                    if l==[1]:
                        l=[1, -1]

                    else:
                        l = [-1, 1]

                    labels2.append(l)
            # labels1[labels1 == -1] = [1, -1]
            # labels1[labels1 == [1]] = [-1, 1]

            # labels1 = tf.cast(processed_data['labels'], dtype=tf.int32)
            # category_indices = labels1
            # unique_category_count = 2
            # y_input = tf.one_hot(category_indices, unique_category_count)
            y_input = np.reshape(np.array(labels2), (2,6,2))
            # nn_output = tf.tile(nn_output, (2,1,1))
            predicted = tf.transpose(tf.expand_dims(K.dot(x, tf.reshape(nn_output, (5,nn_output.shape[0]))),axis=0))
            # predicted = K.dot(x, tf.reshape(nn_output, (5, nn_output.shape[0])))
            # predicted = tf.transpose(predicted)
            num_classes = 2
            # predicted1 = tf.tile(tf.expand_dims(predicted, axis=2),  (1,1,2))
            # print("nn_output:", tf.reduce_mean(nn_output, axis=0).numpy())
            # print()

            # new_predicted = np.array([-1 if i <= 1 else 1 for i in predicted])
            # hinge_loss = np.mean([max(0, 1 - x * y) for x, y in zip(processed_data['labels'], predicted)])
            # hinge_loss = tf.reduce_mean([max(0., 1 - y * wx) for y, wx in zip(np.squeeze(processed_data['labels']), predicted)])
            # hinge_loss = tf.reduce_mean([max(0., 1 - y * wx) for y, wx in
            #                 zip(processed_data['labels'], predicted)])
            # hinge_loss = sum([max(0, 1 - wx * y) for wx, y in
            #  zip(predicted, processed_data['labels'][:50])])
            # hinge_loss =K.sum(1. - K.flatten(tf.cast(predicted, dtype=tf.float64)) * K.flatten(processed_data['labels'][:50]))
            #tf.compat.v1.keras.losses.hinge

            from tensorflow.python.ops import array_ops
            from tensorflow.python.ops import math_ops
            from tensorflow.python.ops import nn_ops


            print("predicted: " + str(predicted))
            logits = math_ops.to_float(predicted)

            batch_size=2
            num_classes=2

            # output = tf.identity(tf.one_hot(tf.sign(tf.cast(processed_data['labels'], dtype=tf.int32)),2))


            regularization_loss = tf.reduce_mean(tf.square(nn_output))
            # output = tf.tile(nn_output, (num_classes,1,1))
            # hinge_loss = tf.reduce_mean(
            #     tf.square(
            #         tf.maximum(
            #             0., 1 - y_input * predicted1
            #         )
            #     )
            # )
            # penalty_parameter=1
            # loss = regularization_loss + penalty_parameter * hinge_loss
            # # logits.get_shape().assert_is_compatible_with(labels.get_shape())
            #
            # output = tf.identity(tf.sign(predicted1))
            # correct_prediction = tf.equal(
            #     tf.argmax(output, 2), tf.argmax(y_input, 2)
            # )
            #
            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # print ("accuracy: "+str(accuracy))

            all_ones = array_ops.ones_like(labels,dtype=tf.float32 )
            # print("logit: "+ str(logits))
            losses = nn_ops.relu(
                math_ops.subtract(all_ones, math_ops.multiply(labels, logits)))
            # cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            #     labels=(labels+1)/2, logits=logits)
            hinge_loss = math_ops.reduce_sum(losses)

        # hinge_loss = tf.keras.losses.hinge(K.flatten(predicted), K.flatten(processed_data['labels'][:50]))
        # ywxmax=tf.maximum(0, tf.ones(60, 1) - tf.matmul(x, w1))

            C1=1
            prediction_loss = C1* hinge_loss #+ cross_entropy_loss
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

            C=.1 #Penalty parameter of the error term

            # total_loss = C* prediction_loss +  0.5 * regularization_loss1
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
