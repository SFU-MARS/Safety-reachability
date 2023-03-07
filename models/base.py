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
        WP=processed_data['Action_waypoint']
        LABELS=processed_data['labels']
        normal = np.array(nn_output)
        # print ("NN's normal: "+ str(normal))

        from sklearn.svm import SVC
        clf = SVC(C=1e6, kernel='linear')
        # clf = SVC( kernel='linear')
        # clf.fit(WP, LABELS[0])
        # w,b = clf.coef_, clf.intercept_
        # print ("w&b: "+str(w) + str(b))

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

        regularization_loss = 0.
        model_variables = self.get_trainable_vars()
        for model_variable in model_variables:
            regularization_loss += tf.nn.l2_loss(model_variable)
        regularization_loss = self.p.loss.regn * regularization_loss
        # processed_data['labels']=np.tile([.25,.25,0,0,.75], (6,1))
        if self.p.loss.loss_type == 'mse':
            # prediction_loss = tf.losses.mean_squared_error(nn_output, processed_data['labels'])
            # svm = np.concatenate((w[0], b), axis=0)
            # svm1 = np.tile(svm, (self.p.trainer.batch_size, 1))
            # prediction_loss =  tf.losses.mean_squared_error(nn_output, svm1)
            x = tf.concat(
                (processed_data['Action_waypoint'][0], tf.ones((processed_data['Action_waypoint'][0].shape[0], 1))),
                axis=1)
            predicted = tf.transpose(tf.expand_dims(K.dot(x, tf.reshape(nn_output, (-1,nn_output.shape[0]))),axis=0))#6
            prediction = predicted.numpy()
            prediction[np.where(prediction > 0)] = 1
            prediction[np.where(prediction < 0)] = -1
            labels = processed_data['labels']
            accuracy = np.count_nonzero(prediction == labels) / np.size(labels)
            # print ("correctly predicted: "+str(accuracy))
        elif self.p.loss.loss_type == 'l2_loss':
            prediction_loss = tf.nn.l2_loss(nn_output - processed_data['labels'])
        elif self.p.loss.loss_type == 'hinge':
            best_params = []
            sigma_sq_all =[]
            C_all =[]

            sample =100
            X_40 = [ x[::sample,:] for x in processed_data['Action_waypoint']]
            labels_40 = [x[::sample, :] for x in processed_data['labels']]

            x = tf.concat(
                (processed_data['Action_waypoint'],
                 tf.ones((processed_data['Action_waypoint'].shape[0], processed_data['Action_waypoint'].shape[1], 1))),
                axis=2)

            X_40_1 = [x1[::sample, :] for x1 in x]
            all_waypoint_sampled = [ x[::sample,:] for x in raw_data['all_waypoint']]

            ## kernels:
            # rbf_feature = RBFSampler(gamma=50, random_state=1, n_components=40)
            # x_rbf = [rbf_feature.fit_transform(x1) for x1 in X_40_1]
            # X_40_1 = x_rbf

            # nr_comp=10
            # rbf_feature = RBFSampler(gamma=0.7, random_state=1, n_components=nr_comp)




            # rbf_feature.fit_transform()
            # kerneled_x = [self.gaussianKernelGramMatrixFull(tf.transpose(x1).numpy(), tf.transpose(x1).numpy())  for x1 in x]
            # x = [self.polynomial_kernel(x1, x1) for x1 in x]
            # gamma = 1 / (3 * X.var()) # 3 is the num of features
            # self.sigma_sq = 1/ (2*gamma)
            # sigma_sq_all =[]
            # for x1 in x :
            #     gamma = 1 / (4 * x1.numpy().var())
            #     sigma_sq_all. append(1 / (2 * gamma))


            predicted = []

            # for w1 in nn_output:
            #     for x1 in x:
            #         t = tf.convert_to_tensor(x1, dtype=tf.float32)
            #         predicted.append(K.dot(t, tf.expand_dims(w1, axis=1)))

####for testing pca
            predicted = [K.dot(tf.convert_to_tensor(x1, dtype=tf.float32), tf.expand_dims(w1, axis=1)) for x1, w1 in zip(X_40_1, nn_output)]
            # hinge_loss = tf.reduce_sum([tf.maximum(0, 1 - wx * y) for wx, y in
            #                      zip(predicted, processed_data['labels'])])
            hinge_losses = [tf.reduce_sum(tf.maximum(0, 1 - wx * y), axis=0) for wx, y in
                                 zip(predicted, labels_40)]

            accuracy_total = []
            prediction_total = []


            for prediction, label in zip(predicted, labels_40):
                prediction = prediction.numpy()
                prediction[np.where(prediction >= 0)] = 1
                prediction[np.where(prediction < 0)] = -1
                prediction_total.append(prediction)
                accuracy = np.count_nonzero(prediction == label) / np.size(label)
                accuracy_total.append(accuracy)

####for testing pca

### pca to 2d fro showing contour, 3 nn output ,
            # hinge_losses = []
            # for WP ,label , C1 in zip(X_40, labels_40,nn_output.numpy()):
            #
            #     pca = PCA(n_components=2)
            #     Xreduced = pca.fit_transform(WP)
            #     Xreduced_1 = tf.concat(
            #         (Xreduced,
            #          tf.ones(
            #              (Xreduced.shape[0], 1))),
            #         axis=1)
            #     X0, X1 = Xreduced_1[:, 0], Xreduced_1[:, 1]
            #
            #     xx, yy = self.make_meshgrid(X0, X1)
            #     prediction = K.dot(Xreduced_1, tf.expand_dims(C1, axis=1))
            #     prediction = prediction.numpy()
            #     prediction[np.where(prediction >= 0)] = 1
            #     prediction[np.where(prediction < 0)] = -1
            #     accuracy = np.count_nonzero(prediction == label) / np.size(label)
            #
            #     # hinge_loss = tf.reduce_sum([tf.maximum(0, 1 - wx * y) for wx, y in
            #     #                      zip(predicted, processed_data['labels'])])
            #     hinge_losses = [tf.reduce_sum(tf.maximum(0, 1 - wx * y), axis=0) for wx, y in
            #                          zip(predicted, labels_40)]
            #
            #     #plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
            #     # [K.dot([x, y, 1], tf.expand_dims(C1, axis=1)) for x, y in zip(xx.ravel(), yy.ravel())]
            #     Z_pre = K.dot(np.c_[xx.ravel(), yy.ravel()], tf.expand_dims(C1, axis=1))
            #     Z_pre = Z_pre.numpy()
            #     Z_pre[np.where(Z_pre >= 0)] = 1
            #     Z_pre[np.where(Z_pre < 0)] = -1
            #
            #     # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            #     Z = Z_pre.reshape(xx.shape)
            #     fig, ax = plt.subplots()
            #     ax.contourf(xx, yy, Z)
            #
            #     ax.scatter(X0, X1, c=np.squeeze(label), cmap=plt.cm.coolwarm, s=20, edgecolors='k')
            #     ax.set_ylabel('PC2')
            #     ax.set_xlabel('PC1')
            #     ax.set_xticks(())
            #     ax.set_yticks(())
            #     ax.set_title('using the PCA transformed/projected features, accuracy: ' +str(accuracy))
            #     ax.legend()
            #     plt.show()

###

### simple svm for 40 wp

            from mpl_toolkits import mplot3d
            import matplotlib
            from matplotlib.colors import ListedColormap
            import matplotlib.backends.backend_pdf
            pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")

            for WP, prediction, label , C1, image , start_nk3, goal, wp in zip(X_40, predicted, labels_40,nn_output.numpy(), raw_data['img_nmkd'] , raw_data['start_state'] , raw_data['goal_position_n2'] , all_waypoint_sampled ):

                fig = plt.figure()
                ax1 = fig.add_subplot(221)
                ax1.imshow(image.astype(np.uint8))
                plt.grid()
                # plt.show()


                # matplotlib.use('Qt4Agg')
                # fig = plt.figure()
                ax2 = fig.add_subplot(222, projection='3d')
                prediction = prediction.numpy()
                prediction[np.where(prediction >= 0)] = 1
                prediction[np.where(prediction < 0)] = -1
                accuracy = np.count_nonzero(prediction == label) / np.size(label)
                wrong= WP[np.where(prediction != label)[0]]
                ax2.scatter3D(wrong[:, 0], wrong[:, 1],
                             wrong[:, 2], s=80 , edgecolors="k" )

                accuracy_total.append(accuracy)

                mycmap = ListedColormap(["red", "green"])
                ax2.scatter3D(WP[:, 0], WP[:, 1],
                             WP[:, 2], c=np.squeeze(label), marker='o', alpha=0.6, cmap=mycmap)
                ax2.set_title ('accuracy: ' +str(accuracy))
                x = WP[:, 0]
                x_min, x_max = x.min() - 1, x.max() + 1
                x = np.linspace(x_min, x_max, 10)
                y = np.linspace(-2, 2, 10)
                z = np.linspace(-np.pi/2, np.pi/2, 10)
                xx, zz = np.meshgrid(x,z)
                y1 = (-C1[0] * xx - C1[2] * zz - C1[3]) / C1[1]
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
                ax2.plot_wireframe(xx, y1, zz, alpha=1, color='gray')
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
                ax3.plot(start[0], start[1], 'k*') #robot
                goal_pos_n2 = goal
                ax3.plot(goal_pos_n2[ 0], goal_pos_n2[1], 'b*')
                pos_nk2 = wp[:, :2]
                ax3.scatter(pos_nk2[:, 0], pos_nk2[:, 1] , c=np.squeeze(label), marker='o', alpha=0.6, cmap=mycmap)
                # plt.show()
                pdf.savefig(fig)
            pdf.close()
#
            accuracy_mean = np.mean(np.array(accuracy_total))
            prediction_total = np.array(prediction_total)
            print("correctly predicted in this batch: " + str(accuracy_mean))

            C = 1
            # prediction_losses= [a*b for a,b in zip(C_all,hinge_losses)]
            prediction_losses = hinge_losses
            prediction_loss =  tf.reduce_sum( prediction_losses)
            prediction_loss = C* prediction_loss
            print("prediction_loss: " + str(prediction_loss.numpy()))

            regularization_loss_svm = 0
            regularization_loss_svm = 1/2 * self.p.loss.lam * tf.nn.l2_loss(nn_output.numpy()[:,:-1])
            regularization_loss = regularization_loss + regularization_loss_svm
            print("regularization_loss: " + str(regularization_loss.numpy()))

        total_loss = prediction_loss + regularization_loss
        print("total_loss: "+str(total_loss.numpy()))

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
            gamma = 1.0 / X.shape[1]

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

