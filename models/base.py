import tensorflow as tf
# from training_utils.architecture.simple_mlp import simple_mlp
from training_utils.architecture.resnet50.resnet_50 import ResNet50
from training_utils.architecture.resnet50_cnn import resnet50_cnn
# from "@tensorflow/tfjs" import * as tf
import numpy as np
K = tf.keras.backend
from sklearn.kernel_approximation import RBFSampler
from sklearn import svm
# from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import learning_curve, GridSearchCV , StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from dotmap import DotMap
from waypoint_grids.projected_image_space_grid import ProjectedImageSpaceGrid

class BaseModel(object):
    """
    A base class for an input-output model that can be trained.
    """
    
    def __init__(self, params):
        self.p = params
        self.make_architecture()
        self.make_processing_functions()
        self.sigma_sq=0.1

    def make_architecture(self):
        """
        Create the NN architecture for the model.
        """
        self.arch = simple_mlp(num_inputs=self.p.model.num_inputs,
                               num_outputs=self.p.model.num_outputs,
                               params=self.p.model.arch)

    def create_params(self):
        p = DotMap()
        p.grid = ProjectedImageSpaceGrid
        # Parameters for the uniform sampling grid
        p.num_waypoints = 3
        p.num_theta_bins = 1
        p.bound_min = [0., 0., -np.pi]
        p.bound_max = [0., 0., 0.]
        # Additional parameters for the projected grid from the image space to the world coordinates
        p.projected_grid_params = DotMap(
            # Focal length in meters
            f=1.,

            # Half-field of view
            fov=np.pi / 4,

            # Height of the camera from the ground in meters
            h=1.,

            # Tilt of the camera
            tilt=0.
        )
        return p
    def compute_loss_function(self, raw_data, param,c,  is_training=None, return_loss_components=False,
                              return_loss_components_and_output=False):
        """
        Compute the loss function for a given dataset.
        """
        # Create the NN inputs and labels

        processed_data  = self.create_nn_inputs_and_outputs(raw_data, is_training=is_training)
        # processed_data = tf.Variable(processed_data)

        nn_output= self.predict_nn_output(processed_data['inputs'], is_training=is_training)
        # nn_output = nn_output1[:, :-1]
        # c_before = tf.expand_dims(nn_output1[:, -1], axis=1)
        import numpy as np
        #
        # cluster_centers = [(1, 0, 0), (1.1, 0, 0)]
        # # cluster_centers = [(2.5, 2.5, 0), (0, -2.5, 0)]
        # num_classes = len(cluster_centers)
        # num_samples_total = 6
        # # X, y = make_circles( n_samples=6, noise=0.07, factor=0.4, random_state=42)
        # # X, y = make_blobs(n_samples=num_samples_total, centers=cluster_centers, n_features=3, cluster_std=0.1,
        # #                   random_state=42)
        # n_samples_1 = 100
        # n_samples_2 = 100
        # centers = [[0.0, 0.0], [2.0, 2.0]]
        # clusters_std = [1.5, 0.5]
        # X, y = make_blobs(
        # n_samples = [n_samples_1, n_samples_2],
        # centers = centers,
        # cluster_std = clusters_std,
        # random_state = 0,
        # shuffle = False,)
        # # import matplotlib.pyplot as plt
        # # fig = plt.figure()
        # # ax = fig.add_subplot(111)
        # # ax.scatter( X[:, 0], X[:, 1], marker='o', c=y )
        # # plt.show()
        # feat_train_sc = np.array(X)
        # feat_train_sc = np.tile(feat_train_sc, (20, 1, 1))
        ## linear svm
        feat_train_sc= processed_data['Action_waypoint']

        # feat_train_sc_1 = tf.concat(
        #     (feat_train_sc,
        #      tf.ones((feat_train_sc.shape[0], feat_train_sc.shape[1], 1))),
        #     axis=2)
        # X = feat_train_sc_1
        # processed_data['Action_waypoint']=feat_train_sc

        # predicted = [np.tanh(predicted1) for predicted1 in predicted_before]
        # hinge_loss = tf.reduce_sum([tf.maximum(0, 1 - wx * y) for wx, y in
        #                      zip(predicted, processed_data['labels'])])
        # y = np.reshape(y, (-1, 1))
        # processed_data['labels'] = np.tile(2 * y - 1, (20, 1, 1))



        # c = (1 / (1 + (np.exp(-c_before)))) * 2 ** 15 + 2 ** (-5)

        # nn_output= nn_output1[:,:-2]
        # import numpy as np
        # c_before= nn_output1[:, -2:-1]
        # c = (1/(1 + (np.exp(-c_before))))*2**15 + 2**(-5)
        # param_before = tf.expand_dims(nn_output1[:, -1], axis=1)
        # param = (1 / (1 + np.exp(-param_before))) * 2 ** 3 +2**(-15)
        # print("nn_output: "+str(nn_output.numpy()))

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure()

        ###
        # import numpy as np

        # x = np.expand_dims(processed_data['Action_waypoint'][0][:, 0], axis=0)
        # y = np.expand_dims(processed_data['Action_waypoint'][0][:, 1], axis=0)
        # y = processed_data['Action_waypoint'][0][:, 1]
        # z = np.expand_dims(processed_data['Action_waypoint'][0][:, 2], axis=0)

        normal = np.array(nn_output)
        # print ("NN's normal: "+ str(normal))


        # clf = SVC(C=1e6, kernel='linear')


        # sample = 100
        # X_40 = [x[::sample, :] for x in processed_data['Action_waypoint']]
        #
        # labels_40 = [x[::sample, :] for x in processed_data['labels']]




        # X = tf.concat(
        #     (processed_data['Action_waypoint'],
        #      tf.ones((processed_data['Action_waypoint'].shape[0], processed_data['Action_waypoint'].shape[1], 1))),
        #     axis=2)
        #
        # X_sampled_1 = [x1[::sample, :] for x1 in x]
        # all_waypoint_sampled = [x[::sample, :] for x in raw_data['all_waypoint']]
        expected_output =[]
        regularization_loss = 0.
        model_variables = self.get_trainable_vars()
        for model_variable in model_variables:
            regularization_loss += tf.nn.l2_loss(model_variable)
        regularization_loss = self.p.loss.regn * regularization_loss

        if self.p.loss.loss_type == 'mse':
            prediction_losses =[]
            C=1
            clf = svm.SVC(kernel='linear', C=C)

            for WP, label, output in zip(processed_data['Action_waypoint'], processed_data['labels'], nn_output):
                # clf = SVC(kernel='linear')

                if np.max(label) == 1 and np.min(label) == -1:
                    clf.fit(WP, label)
                    print (clf.score(WP, label))

                    w, b = clf.coef_, clf.intercept_
                    W = tf.concat((w, np.reshape(b, (1, 1))), axis=1)
                    prediction_losses.append(tf.losses.mean_squared_error(tf.expand_dims(output, axis=0), W))
                else:
                    print ('one group')


            prediction_losses = tf.convert_to_tensor(prediction_losses)
            prediction_loss = tf.reduce_sum(prediction_losses)

            feat_train_sc_1 = tf.concat(
                (feat_train_sc,
                 tf.ones((feat_train_sc.shape[0], feat_train_sc.shape[1], 1))),
                axis=2)
            X = feat_train_sc_1
            predicted = [K.dot(tf.convert_to_tensor(x1, dtype=tf.float32), tf.expand_dims(output, axis=1)) for
                         x1, output in
                         zip(X, nn_output)]

            accuracy_total = []
            prediction_total = []

            for prediction, label in zip(predicted, processed_data['labels']):
                prediction = prediction.numpy()
                prediction[np.where(prediction >= 0)] = 1
                prediction[np.where(prediction < 0)] = -1
                prediction_total.append(prediction)
                accuracy = np.count_nonzero(prediction == label) / np.size(label)
                accuracy_total.append(accuracy)
            # prediction_losses = np.array(prediction_losses)
            # prediction_loss = np.sum(prediction_losses)


            # X = tf.concat(
            #     (processed_data['Action_waypoint'],
            #      tf.ones((processed_data['Action_waypoint'].shape[0], processed_data['Action_waypoint'].shape[1], 1))),
            #     axis=2)







            # prediction_loss = tf.losses.mean_squared_error(nn_output, processed_data['labels'])

        elif self.p.loss.loss_type == 'l2_loss':

            prediction_loss = tf.nn.l2_loss(nn_output - processed_data['labels'])

        elif self.p.loss.loss_type == 'hinge':
            best_params = []
            sigma_sq_all =[]
            C_all =[]

            # feat_train_sc = processed_data['Action_waypoint']

            # feat_train_kr = []
            # for X in feat_train_sc:
            #     fX = [(x2[0] * x2[1], x2[1] * x2[2], x2[0] * x2[2]) for x2 in X]
            #     feat_train_kr.append(np.array(fX))
            # feat_train_kr = np.array(feat_train_kr)

            # feat_train_sc_1 = tf.concat(
            #     (feat_train_kr,
            #      tf.ones((feat_train_kr.shape[0], feat_train_kr.shape[1], 1))),
            #     axis=2)
            feat_train_sc_1 = tf.concat(
                (feat_train_sc,
                 tf.ones((feat_train_sc.shape[0], feat_train_sc.shape[1], 1))),
                axis=2)
            X = feat_train_sc_1


            #svm with kernel
            ##rbf_feature = RBFSampler(gamma=1000, random_state=1, n_components=10)
            ## x_rbf = [rbf_feature.fit_transform(x1) for x1 in X_sampled_1]
            # x_rbf = [self.RBF(x1, gamma= param1) for x1 , param1 in zip(feat_train_sc,param) ]
            # x_rbf = np.array(x_rbf)
            # x_rbf_1 = tf.concat(
            #     (x_rbf,
            #      tf.ones((x_rbf.shape[0], x_rbf.shape[1], 1))),
            #     axis=2)
            # X= x_rbf_1


            # rbf_feature.fit_transform()
            # kerneled_x = [self.gaussianKernelGramMatrixFull(tf.transpose(x1).numpy(), tf.transpose(x1).numpy())  for x1 in x]
            # x = [self.polynomial_kernel(x1, x1) for x1 in x]
            # gamma = 1 / (3 * X.var()) # 3 is the num of features
            # self.sigma_sq = 1/ (2*gamma)
            # sigma_sq_all =[]
            # for x1 in x :
            #     gamma = 1 / (4 * x1.numpy().var())
            #     sigma_sq_all. append(1 / (2 * gamma))

            # for w1 in nn_output:
            #     for x1 in x:
            #         t = tf.convert_to_tensor(x1, dtype=tf.float32)
            #         predicted.append(K.dot(t, tf.expand_dims(w1, axis=1)))

####for testing pca
            predicted = [K.dot(tf.convert_to_tensor(x1, dtype=tf.float32), tf.expand_dims(output, axis=1)) for
                         x1, output in
                         zip(X, nn_output)]

            accuracy_total = []
            prediction_total = []
            precision_total =[]
            recall_total =[]

            for prediction0, label in zip(predicted, processed_data['labels']):
                prediction0 = prediction0.numpy()
                prediction = np.tanh(prediction0)
                prediction[np.where(prediction >= 0)] = 1
                prediction[np.where(prediction < 0)] = -1
                prediction_total.append(prediction)
                accuracy = np.count_nonzero(prediction == label) / np.size(label)
                accuracy1 = accuracy_score(label, prediction)
                precision = precision_score (label, prediction)
                recall = recall_score(label, prediction)
                accuracy_total.append(accuracy)
                precision_total.append(precision)
                recall_total.append(recall)


            # Precision Score = TP / (FP + TP)
            # Recall Score = TP / (FN + TP)
            hinge_losses = [tf.reduce_sum(tf.maximum(0, 1 - wx * y), axis=0) for wx, y in
                                 zip(predicted, processed_data['labels'])]




            ### simple svm for 40 wp

            from mpl_toolkits import mplot3d
            import matplotlib
            from matplotlib.colors import ListedColormap
            import matplotlib.backends.backend_pdf
            pdf = matplotlib.backends.backend_pdf.PdfPages("output_fov_sample500.pdf")
            sample = 50  #600 , 50

            all_waypoint_sampled = [x[::sample, :] for x in raw_data['all_waypoint']]

            p = self.create_params()

            # Initialize and Create a grid
            grid = p.grid(p)
            from waypoint_grids.projected_image_space_grid import ProjectedImageSpaceGrid
            # for WP, prediction, label, C1, image, start_nk3, goal, wp, top in zip(processed_data['Action_waypoint'],
            #                                                                       prediction_total,
            #                                                                       processed_data['labels'],
            #                                                                       nn_output.numpy(),
            #                                                                       raw_data['img_nmkd'][:, :, :, :3],
            #                                                                       raw_data['start_state'],
            #                                                                       raw_data['goal_position_n2'],
            #                                                                       all_waypoint_sampled,
            #                                                                       raw_data['topview']):
            #     # camera_pos_13 = config.heading_nk1()[0]
            #     # camera_grid_world_pos_12 = config.position_nk2()[0] / dx_m
            #     #
            #     # # image of current state
            #     # rgb_image_1mk3 = r._get_rgb_image(camera_grid_world_pos_12, camera_pos_13)
            #     #
            #     # img1 = r._get_topview(camera_grid_world_pos_12, camera_pos_13)
            #
            #     fig = plt.figure()
            #     ax1 = fig.add_subplot(221)
            #     ax1.imshow(image.astype(np.uint8))
            #     x = WP[:, 0:1]
            #     x1 = np.expand_dims(x, axis=2)
            #     y = WP[:, 1:2]
            #     y1 = np.expand_dims(y, axis=2)
            #     t = WP[:, 2:3]
            #     t1 = np.expand_dims(t, axis=2)
            #
            #     # x1= np.expand_dims(np.expand_dims(np.expand_dims(x, axis=0),axis=1), axis=2)
            #     # y1 = np.expand_dims( np.expand_dims(np.expand_dims(y, axis=0), axis=1), axis=2)
            #     # t1 = np.expand_dims( np.expand_dims(np.expand_dims(t, axis=0), axis=1), axis=2)
            #     wp_image = grid.generate_imageframe_waypoints_from_worldframe_waypoints(x1, y1, t1)
            #     wp_image_x = (wp_image[0][:, 0, 0]+1)*224/2
            #     wp_image_y = (wp_image[1][:, 0, 0])*224/1
            #     ax1.scatter(wp_image_x, wp_image_y,  marker="x", color="blue", s=10)
            #     theta = WP[:, 2]  # theta of the arrow
            #     u, v =  1 * (np.cos(theta), np.sin(theta))
            #     q = ax1.quiver(wp_image_x, wp_image_y, u, v)
            #     plt.show()
            #
            #     ax2 = fig.add_subplot(222)
            #     # prediction = prediction.numpy()
            #     # prediction[np.where(prediction >= 0)] = 1
            #     # prediction[np.where(prediction < 0)] = -1
            #     # wrong = WP[np.where(prediction != label)[0]]
            #     # ax2.scatter3D(wrong[:, 0], wrong[:, 1],
            #     #               wrong[:, 2], s=80, edgecolors="k")
            #     # ax2.scatter(wrong[:, 0], wrong[:, 1], s=80, edgecolors="k")
            #
            #     color = ['red' if l == -1 else 'green' for l in label]
            #     # mycmap = ListedColormap(["red", "green"])
            #
            #     # ax2.scatter3D(WP[:, 0], WP[:, 1],
            #     #               WP[:, 2], c=np.squeeze(label), marker='o', alpha=0.6, cmap=mycmap)
            #     # ax2.scatter(WP[:, 0], WP[:, 1]
            #     #               , c=np.squeeze(label), marker='o', alpha=0.6, cmap=mycmap)
            #     ax2.scatter(WP[:, 0], WP[:, 1]
            #                   , marker='o', alpha=0.6, color=color)
            #     # ax2.matplotlib.pyplot.arrow(WP[:, 0], WP[:, 1], math.cos(WP[:, 2]), math.sin(WP[:, 2]))
            #     x = WP[:, 0]
            #     y = WP[:, 1]
            #     theta = WP[:, 2]  # theta of the arrow
            #     u, v =  1 * (np.cos(theta), np.sin(theta))
            #
            #     q = ax2.quiver(x, y, u, v)
            #     ax2.set_title('ground truth')
            #     # plt.xlim(-0.5, len(x[0]) - 0.5)
            #     # plt.ylim(-0.5, len(x) - 0.5)
            #     # plt.xticks(range(len(x[0])))
            #     # plt.yticks(range(len(x)))
            #
            #     plt.show()
            #

            # 2d plots
            for WP, prediction, label, C1, image, start_nk3, goal, wp, top in zip(processed_data['Action_waypoint'], prediction_total, processed_data['labels'],
                                                                             nn_output.numpy(),
                                                                             raw_data['img_nmkd'][:,:,:,:3],
                                                                             raw_data['start_state'],
                                                                             raw_data['goal_position_n2'],
                                                                             all_waypoint_sampled,
                                                                             raw_data['topview']):


                # camera_pos_13 = config.heading_nk1()[0]
                # camera_grid_world_pos_12 = config.position_nk2()[0] / dx_m
                #
                # # image of current state
                # rgb_image_1mk3 = r._get_rgb_image(camera_grid_world_pos_12, camera_pos_13)
                #
                # img1 = r._get_topview(camera_grid_world_pos_12, camera_pos_13)

                plt.imshow(np.squeeze(top))
                plt.show()

                fig = plt.figure()
                ax1 = fig.add_subplot(221)
                ax1.imshow(image.astype(np.uint8))
                plt.grid()


                x = WP[:, 0:1]
                x1 = np.expand_dims(x, axis=2)
                y = WP[:, 1:2]
                y1 = np.expand_dims(y, axis=2)
                t = WP[:, 2:3]
                t1 = np.expand_dims(t, axis=2)

                # x1= np.expand_dims(np.expand_dims(np.expand_dims(x, axis=0),axis=1), axis=2)
                # y1 = np.expand_dims( np.expand_dims(np.expand_dims(y, axis=0), axis=1), axis=2)
                # t1 = np.expand_dims( np.expand_dims(np.expand_dims(t, axis=0), axis=1), axis=2)
                wp_image = grid.generate_imageframe_waypoints_from_worldframe_waypoints(x1, y1, t1)
                wp_image_x = (wp_image[0][:, 0, 0]+1)*224/2
                wp_image_y = (wp_image[1][:, 0, 0])*224/1
                ax1.scatter(wp_image_x, wp_image_y,  marker="x", color="blue", s=10)
                theta = WP[:, 2:3] # theta of the arrow
                u, v =  1 * (np.cos(theta), np.sin(theta))
                q = ax1.quiver(wp_image_x, wp_image_y, u, v)
                plt.show()

                # plt.show()

                # matplotlib.use('Qt4Agg')
                # fig = plt.figure()


                # ax2 = fig.add_subplot(222, projection='3d')
                ax2 = fig.add_subplot(222)
                # prediction = prediction.numpy()
                # prediction[np.where(prediction >= 0)] = 1
                # prediction[np.where(prediction < 0)] = -1
                # wrong = WP[np.where(prediction != label)[0]]
                # ax2.scatter3D(wrong[:, 0], wrong[:, 1],
                #               wrong[:, 2], s=80, edgecolors="k")
                # ax2.scatter(wrong[:, 0], wrong[:, 1], s=80, edgecolors="k")

                color = ['red' if l == -1 else 'green' for l in label]
                # mycmap = ListedColormap(["red", "green"])

                # ax2.scatter3D(WP[:, 0], WP[:, 1],
                #               WP[:, 2], c=np.squeeze(label), marker='o', alpha=0.6, cmap=mycmap)
                # ax2.scatter(WP[:, 0], WP[:, 1]
                #               , c=np.squeeze(label), marker='o', alpha=0.6, cmap=mycmap)
                ax2.scatter(WP[:, 0], WP[:, 1]
                              , marker='o', alpha=0.6, color=color)
                # ax2.matplotlib.pyplot.arrow(WP[:, 0], WP[:, 1], math.cos(WP[:, 2]), math.sin(WP[:, 2]))
                x = WP[:, 0]
                y = WP[:, 1]
                theta = WP[:, 2]  # theta of the arrow
                u, v =  1 * (np.cos(theta), np.sin(theta))

                q = ax2.quiver(x, y, u, v)
                ax2.set_title('ground truth')
                # plt.xlim(-0.5, len(x[0]) - 0.5)
                # plt.ylim(-0.5, len(x) - 0.5)
                # plt.xticks(range(len(x[0])))
                # plt.yticks(range(len(x)))

                plt.show()

                ax4 = fig.add_subplot(224)
                accuracy = np.count_nonzero(prediction == label) / np.size(label)
                color_result = ['red' if l == -1 else 'green' for l in prediction]
                ax4.scatter(x, y
                            , marker='o', alpha=0.6, color=color_result)
                wrong = WP[np.where(prediction != label)[0]]
                ax4.scatter(wrong[:, 0], wrong[:, 1], s=60, edgecolors="k")
                # ax2.scatter3D(wrong[:, 0], wrong[:, 1],
                #               wrong[:, 2], s=80, edgecolors="k")
                # safe = WP[np.where(prediction == 1)[0]]
                # unsafe = WP[np.where(prediction == -1)[0]]
                ax4.set_title('accuracy: ' + str(accuracy))
                # ax4.scatter(WP[:, 0], WP[:, 1]
                #             , c=np.squeeze(prediction), marker='o', alpha=0.6, cmap=mycmap)
                q1 = ax4.quiver(x, y, u, v)

                # ax4.scatter(safe[:, 0], safe[:, 1], s=80, edgecolors="g")
                # ax4.scatter(unsafe[:, 0], unsafe[:, 1], s=80, edgecolors="r")


                plt.show()

                # x = WP[:, 0]
                # x_min, x_max = x.min() - 1, x.max() + 1
                # x = np.linspace(x_min, x_max, 10)
                # y = np.linspace(-2, 2, 10)
                # z = np.linspace(-np.pi / 2, np.pi / 2, 10)
                # xx, zz = np.meshgrid(x, z)
                # y1 = (-C1[0] * xx - C1[2] * zz - C1[3]) / C1[1]
                # y = WP[:, 1]
                # y_min, y_max = y.min() , y.max()
                #
                # ax4 = fig.add_subplot(224, projection='3d')
                #
                # y_filtered = y1[np.where(np.logical_and(y1 >= y_min, y1 <= y_max))]
                # x_filtered = xx[np.where(np.logical_and(y1 >= y_min, y1 <= y_max))]
                # th_filtered = zz[np.where(np.logical_and(y1 >= y_min, y1 <= y_max))]
                # # ax4.plot_surface(np.expand_dims(x_filtered, axis=1), np.expand_dims(y_filtered, axis=1),
                # #                    np.expand_dims(th_filtered, axis=1), alpha=1, color='gray')
                # # plt.show()
                # ax4.scatter3D(np.expand_dims(x_filtered, axis=1), np.expand_dims(y_filtered, axis=1),
                #               np.expand_dims(th_filtered, axis=1), alpha=1, color='gray')
                # plt.show()


                # ax.plot_surface(xx, y1, zz, alpha=1, color='gray', linewidth=0)
                # plt.show()
                # if z< np.pi/2 and z> -np.pi/2:
                #     z1 = z
                # elif z> 3*np.pi/2 and z<2*np.pi:
                #     z1= z - 2*np.pi
                # else:

                # ax.plot_surface(xx, yy,  np.arctan(np.tan(zc2)), alpha=1, color='gray', linewidth=0)
                # ax.plot_surface(xx, yy, z, alpha=1, color='gray', linewidth=0)
                # ax.plot_wireframe(xx, yy, z, alpha=1, color='gray')
                # ax2.plot_wireframe(xx, y1, zz, alpha=1, color='gray')
                # wrongs = WP[np.where(prediction != LABELS1)]
                # ax.scatter3D(wrongs[:, 0], wrongs[:, 1],
                #              wrongs[:, 2], marker='*')

                # plt.show()

                from obstacles.sbpd_map import SBPDMap
                # fig = plt.figure()
                ax3 = fig.add_subplot(223)
                obstacle_map = SBPDMap(self.p.simulator_params.obstacle_map_params)
                obstacle_map.render(ax3)
                start = start_nk3[0]
                ax3.plot(start[0], start[1], 'k*')  # robot
                goal_pos_n2 = goal
                ax3.plot(goal_pos_n2[0], goal_pos_n2[1], 'b*')
                pos_nk2 = wp[:, :2]
                theta =  wp[:, 2]
                # ax3.scatter(pos_nk2[:, 0], pos_nk2[:, 1], c=np.squeeze(label), marker='o', alpha=0.6, cmap=mycmap)
                ax3.scatter(pos_nk2[:, 0], pos_nk2[:, 1], marker='o', alpha=0.6, color=color)
                x = pos_nk2[:, 0]
                y = pos_nk2[:, 1]
              # theta of the arrow
                u, v =  1 * (np.cos(theta), np.sin(theta))

                # q = ax3.quiver(x, y, u, v)
                plt.show()
                theta_world= WP[:, 2]  # theta of the arrow
                # u, v =  1 * (np.cos(theta_world), np.sin(theta_world))
                # q = ax3.quiver(pos_nk2[:, 0], pos_nk2[:, 1], u, v)
                # plt.show()
                pdf.savefig(fig)
            pdf.close()
# end of 2d plot

            accuracy_mean = np.mean(np.array(accuracy_total))
            precision_mean = np.mean(np.array(precision_total))
            recall_mean = np.mean(np.array(recall_total))



            print("correctly predicted in this batch: " + str(accuracy_mean))

            C_all = c
            # prediction_losses= [a*b for a,b in zip(C_all,hinge_losses)]
            prediction_losses = hinge_losses
            # prediction_losses = np.float32(prediction_losses)
            prediction_loss =  tf.reduce_sum( prediction_losses)
            print("prediction_loss: " + str(prediction_loss.numpy()))
            # prediction_loss = C* prediction_loss


            regularization_loss_svm = 0
            regularization_loss_svm = 1/2  * tf.nn.l2_loss(nn_output.numpy()[:,:-1])
            regularization_loss = regularization_loss + regularization_loss_svm
            print("regularization_loss: " + str(regularization_loss.numpy()))

        accuracy_mean = np.mean(np.array(accuracy_total))
        print("correctly predicted in this batch: " + str(accuracy_mean))

        total_loss = prediction_loss + regularization_loss
        # print("total_loss: "+str(total_loss.numpy()))
        print("total_loss: " + str(total_loss))

        if return_loss_components_and_output:
            return regularization_loss, prediction_loss, total_loss, nn_output
        # elif return_loss_components:
        #     return regularization_loss, prediction_loss, total_loss
        elif return_loss_components:
            return regularization_loss, prediction_loss, total_loss, accuracy_mean
        else:
            return total_loss
                # return regularization_loss

    # @staticmethod
    def plot_contours(ax, clf, xx, yy, **params):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out

    # @staticmethod
    def make_meshgrid(self, x, y, h=.2):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        return xx, yy

    def RBF(self, X, gamma):

        # Free parameter gamma
        if gamma == None:
            # gamma = 1.0 / X.shape[1]
            gamma = 1/ (2*np.var (X))
            print ("gamma is: "+str(gamma))

        # RBF kernel Equation
        K = np.exp(-gamma * np.sum((X - X[:, np.newaxis]) ** 2, axis=-1))

        return K

    def gaussian_kernel(self, x1, x):
        m = x.shape[0]
        n = x1.shape[0]
        op = [[self.__similarity(x1[x_index], x[l_index]) for l_index in range(m)] for x_index in range(n)]
        return np.array(op)
    
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

    def __similarity(self,x,l):
        return np.exp(-sum((x-l)**2)/(2*self.sigma_sq))


    @staticmethod
    def gaussianKernelGramMatrixFull( X1, X2, sigma=0.1):
        """(Pre)calculates Gram Matrix K"""

        gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                x1 = x1.flatten()
                x2 = x2.flatten()
                gram_matrix[i, j] = np.exp(- np.sum(np.power((x1 - x2), 2)) / float(2 * (sigma ** 2)))

        return gram_matrix

    # def gaussian_kernel(self,x1,x):
    #     m=x.shape[0]
    #     n=x1.shape[0]
    #     op=[[self.__similarity(x1[x_index],x[l_index]) for l_index in range(m)] for x_index in range(n)]
    #     return tf.convert_to_tensor(op)
    #
    def polynomial_kernel(self, x1, x, p=3):
        m=x.shape[0]
        n=x1.shape[0]
        op = [[(1 + np.dot(x1[x_index],x[l_index]) ** p) for l_index in range(m)] for x_index in range(n)]
        return tf.convert_to_tensor(op, dtype=tf.float32)

    # 2d plots
    from dotmap import DotMap



