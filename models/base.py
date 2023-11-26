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
from sklearn.metrics import balanced_accuracy_score
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.backends.backend_tkagg as tkagg
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from scipy.interpolate import interp1d
from sklearn import metrics
from sbpd.sbpd_renderer import SBPDRenderer
from itertools import combinations_with_replacement as combinations_w_r
from time import time
from systems.dubins_car import DubinsCar
from math import log
from utils import depth_utils
from training_utils.data_processing.distort_images import fov_and_tilt_distortion
import cv2
import matplotlib
from pathlib import Path


class PolynomialFeaturesLayer(tf.keras.layers.Layer):
    def __init__(self, degree):
        super(PolynomialFeaturesLayer, self).__init__()
        self.degree = degree

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        num_features = inputs.shape[-1]

        # Generate powers for all possible combinations of polynomial terms
        power_matrix = tf.tile(tf.expand_dims(tf.range(self.degree, dtype=tf.float32), 0), [num_features, 1])

        # Expand dimensions of inputs for element-wise power operation
        expanded_inputs = tf.expand_dims(inputs, axis=2)

        # Compute polynomial features
        powered_inputs = tf.pow(expanded_inputs, power_matrix)

        # Reshape and concatenate polynomial terms
        polynomial_features = tf.reshape(powered_inputs, [batch_size, -1])
        return polynomial_features


def normalize(X, biases, scales):
    if isinstance(X, np.ndarray):
        X = tf.convert_to_tensor(X)
    # expand for all waypoints
    return tf.expand_dims(scales, 1) * (tf.cast(X, dtype=tf.float32) + tf.expand_dims(biases, 1))


class BaseModel(object):
    """
    A base class for an input-output model that can be trained.
    """
    
    def __init__(self, params):
        self.p = params
        self.make_architecture()
        self.make_processing_functions()
        self.sigma_sq=0.1
        self.poly_layer = PolynomialFeaturesLayer(degree=3)
        self.renderer = SBPDRenderer.get_renderer(self.p.simulator_params.obstacle_map_params.renderer_params)


    def vz_values(self, start_nk3):
        y = np.linspace(-2.5, 2.5, 100)
        x = np.linspace(0, 5, 100)
        theta = np.zeros_like(x)
        xx, yy = np.meshgrid(x, y)
        xy = np.stack((xx, yy), axis=-1).reshape((-1, 2))
        xyt = np.zeros((xy.shape[0], 3))
        xyt[:, :2] = xy
        start = np.tile(start_nk3, (np.shape(xyt)[0], 1))
        with tf.device('cpu'):
            start = tf.convert_to_tensor(np.expand_dims(start, 1), dtype=tf.float32)
            xyt = tf.convert_to_tensor(np.expand_dims(xyt, 1), dtype=tf.float32)
            xyt_world = DubinsCar.convert_position_and_heading_to_ego_coordinates(start, xyt)


    def make_architecture(self):
        """
        Create the NN architecture for the model.
        """
        self.arch = simple_mlp(num_inputs=self.p.model.num_inputs,
                               num_outputs=self.p.model.num_outputs,
                               params=self.p.model.arch)

    def create_params(self, image_width):
        p = DotMap()
        p.grid = ProjectedImageSpaceGrid
        # Parameters for the uniform sampling grid
        p.num_waypoints = 21735
        p.num_theta_bins = 21
        p.bound_min = [2.5, 2.5, 0]
        p.bound_max = [0., -2.5, -np.pi]
        # Additional parameters for the projected grid from the image space to the world coordinates
        p.projected_grid_params = DotMap(
            # Focal length in meters
            f=1,

            # Half-field of view
            fov=0.7853981633974483,

            # Height of the camera from the ground in meters
            h=98/100.0,

            # Tilt of the camera
            tilt=-0.7853981633974483
        )
        p.projected_grid_params.f = fov_to_focal(p.projected_grid_params.fov, image_width)

        return p

    @staticmethod
    def polyfeatures(X, degree=3):
        n_features = X.shape[1]

        combs = []
        for n in range(1, degree+1):
            comb = list(combinations_w_r(range(n_features), n))
            combs.extend(comb)

        features = [tf.ones_like(X[:, 0])]

        for comb in combs:
            feature = tf.ones_like(X[:, 0])
            for i in comb:
                feature *= X[:, i]

            features.append(feature)

        return tf.stack(features, axis=-1)

    @staticmethod
    def cosine_distance(y_true, y_pred, axis=-1):
        y_true = tf.nn.l2_normalize(y_true, axis=axis)
        y_pred = tf.nn.l2_normalize(y_pred, axis=axis)
        return tf.losses.cosine_distance(y_true, y_pred, axis=axis)


    def test_decision_boundary(self, raw_data, param, c, is_training=None, return_loss_components=False,
                              return_loss_components_and_output=False, stamp=None):
        """
        Compute the loss function for a given dataset.
        """
        # Create the NN inputs and labels

        # #read images from file and convert to tensor
        # filenames = [
        #     '/home/ttoufigh/Downloads/left00.jpg',
        #     '/home/ttoufigh/Downloads/left00.jpg',
        #     '/home/ttoufigh/Downloads/left00.jpg',
        #     # '/home/ttoufigh/Downloads/images/3.jpg',
        #     '/home/ttoufigh/Downloads/left00.jpg',
        #     '/home/ttoufigh/Downloads/left00.jpg',
        #
        # ]
        # imgs = []
        # for filename in filenames:
        #     img = cv2.imread(filename)[..., ::-1]
        #     # resize at the end after distortion
        #     cx= 304
        #     cy= 197
        #     cropped_img=img [cy-160:cy+160, cx-160:cx+160, :]
        #     # matplotlib.use('TkAgg')
        #     # plt.imshow(cropped_img)
        #     # plt.show()
        #     img = cv2.resize(cropped_img, (224, 224)).astype(np.float32)
        #     imgs.append(img.astype(np.float32))
        # raw_data['img_nmkd'] = np.stack(imgs, axis=0)

        # # base = real world params
        # base_tilt = np.float32(45.0 * np.pi / 180.0)
        # base_fov_x = np.float32(69.4 / 2 * np.pi / 180.0)
        # base_fov_y = np.float32(42.5 / 2 * np.pi / 180.0)
        # base_f = np.float32(0.01)
        #
        # # simulation FOV is 90
        # new_tilt = np.float32(45.0 * np.pi / 180.0)
        # new_fov_x = np.float32(90.0 / 2 * np.pi / 180.0)
        # new_fov_y = np.float32(90.0 / 2 * np.pi / 180.0)
        #
        # # raw_data['img_nmkd'] = fov_and_tilt_distortion(
        # #     raw_data['img_nmkd'], new_tilt, new_fov_x, new_fov_y,
        # #     base_tilt, base_fov_x, base_fov_y, base_f
        # # )
        #
        # # resize all images
        # # raw_data['img_nmkd'] = np.array([
        # #     cv2.resize(img, (224, 224)).astype(np.float32)
        # #     for img in raw_data['img_nmkd']]
        # # )

        processed_data = self.test_nn_inputs_and_outputs(raw_data, is_training=is_training)

        # processed_data = tf.Variable(processed_data)
        # start_nn = time()
        nn_output, waypoint_scale, waypoint_bias = self.predict_nn_output(
            processed_data['inputs'] + [tf.constant([[1]], dtype=processed_data['inputs'][0].dtype)],
            is_training=is_training
        )
        # end_nn = time()
        # print("time_nn: ", (end_nn - start_nn))

        # print("nn after" + str(nn_output))

        sample = 1  # 600 , 50

        biases = nn_output[:, :4]
        scales = nn_output[:, 4:8]
        kernel_weights = nn_output#[:, 8:]

        # debug
        biases = waypoint_bias  # tf.zeros_like(biases)
        scales = waypoint_scale  # tf.ones_like(scales)

        # X_norm = [normalize(X, bias, scale) for X, bias, scale in zip(feat_train_sc, biases, scales) ]


        LABEL_UNSAFE = 0
        LABEL_SAFE = 1
        def remap_labels(label):
            return (label + 1) / 2


        if True:
            # start_plot = time()
            self.plot_2plots_decision_boundary(biases, kernel_weights, nn_output, normalize, processed_data,
                                        raw_data, remap_labels, sample, scales, stamp=stamp)
            # finish_plot = time()
            # print("time_plot: ", (finish_plot - start_plot))
        return

    def compute_loss_function2(self, raw_data, param, c, is_training=None, return_loss_components=False,
                              return_loss_components_and_output=False):
        """
                Compute the loss function for a given dataset.
                """
        # Create the NN inputs and labels

        processed_data = self.create_nn_inputs_and_outputs(raw_data, is_training=is_training)
        batch_size = raw_data['img_nmkd'].shape[0]
        nn_output, waypoint_scale, waypoint_bias, _ = self.predict_nn_output(
            processed_data['inputs']
            + [tf.constant([[1]], dtype=processed_data['inputs'][0].dtype)]
            + [np.zeros((batch_size, 2304 + self.p.model.num_outputs))],
            is_training=is_training
        )
        biases = waypoint_bias  # tf.zeros_like(biases)
        scales = waypoint_scale  # tf.ones_like(scales)

        waypoint_fc = [i.name for i in self.arch.layers].index('waypoint_fc')
        waypoint_fc = self.arch.layers[waypoint_fc]

        feat_train_sc = processed_data['Action_waypoint']
        num_waypoints = feat_train_sc.shape[1]

        X_norm = normalize(feat_train_sc, biases, scales)
        X_kerneled = self.polyfeatures(
            tf.reshape(X_norm, (batch_size * num_waypoints, -1))
        )
        X_kerneled = tf.reshape(X_kerneled, [batch_size, num_waypoints, -1])

        # create 2nd dimension and tile it with num_waypoints
        nn_output = tf.expand_dims(nn_output, 1)
        nn_output = tf.tile(nn_output, [1, num_waypoints, 1])

        nn_input = tf.concat(
            [nn_output, X_kerneled], axis=-1
        )
        predicted = waypoint_fc(tf.reshape(nn_input, [batch_size * num_waypoints, -1]))
        predicted = tf.reshape(predicted, [batch_size, num_waypoints, -1])
        print(predicted.shape)

        regularization_loss = 0.
        kernel_losses = [0]
        regularization_loss_kernel = 0.
        model_variables = self.get_trainable_vars()
        for model_variable in model_variables:
            regularization_loss += tf.nn.l2_loss(model_variable)  # / model_variable.shape.num_elements()

        regularization_loss = self.p.loss.regn * regularization_loss

        accuracy_total = []
        prediction_total = []
        precision_total = []
        recall_total = []
        output_total = []
        percentage_total = []
        F1_total = []

        sample = 1  # 600 , 50
        sample_weights = []

        X_kerneled = [X_kerneled[i] for i in range(batch_size)]
        predicted = [predicted[i] for i in range(batch_size)]

        for X, y in zip(X_kerneled, processed_data['labels']):

            n_sample0 = np.size(np.where(y == 1)[0])
            n_sample1 = np.size(np.where(y == -1)[0])

            sample0_ratio = n_sample0 / (n_sample1 + n_sample0)
            if np.isnan(sample0_ratio):
                print('sample0_ratio nan')

            print(f'initial sample0_ratio: {sample0_ratio}')
            if sample0_ratio > 0.75:
                sample0_ratio = sample0_ratio ** 2
            elif sample0_ratio < 0.25:
                sample0_ratio = sample0_ratio ** 0.5
            print(f'final sample0_ratio: {sample0_ratio}')
            sample1_ratio = 1 - sample0_ratio

            # debug
            # sample1_ratio = sample0_ratio = 1

            sample_weight = np.array(
                [sample0_ratio if i == -1 else sample1_ratio for i in y])
            if np.all(sample_weight == 0):
                sample_weight[:] = 1.0

            sample_weight = sample_weight / np.sum(sample_weight) * y.shape[0]
            if np.any(np.isnan(sample_weight)):
                print('sample_weight nan')
            # sample_weight = np.ones_like(sample_weight)
            sample_weights.append(sample_weight)

        LABEL_UNSAFE = 0
        LABEL_SAFE = 1

        def remap_labels(label):
            return (label + 1) / 2

        for prediction0, label, sample_weight in zip(predicted, processed_data['labels'], sample_weights):
            # prediction0 = prediction.numpy()

            label = remap_labels(label)

            prediction = tf.sigmoid(prediction0)
            # prediction = prediction0

            output_total.append(prediction)
            prediction = prediction.numpy()
            # reverse for scores calculation
            prediction_binary = np.zeros_like(prediction)
            prediction_binary[np.where(prediction >= 0.5)] = 1
            prediction_binary[np.where(prediction < 0.5)] = 0
            prediction = prediction_binary
            prediction_total.append(prediction)
            print('sample_weight: ', sample_weight)
            print("label: ", label.transpose())
            print("prediction: ", prediction.transpose())
            # print("logits: ", prediction0.numpy().transpose())
            accuracy = np.count_nonzero(prediction == label) / np.size(label)
            # accuracy_total.append(accuracy)
            # prediction_loss1 = tf.losses.mean_squared_error(prediction0,label)
            # prediction_total.append(prediction_loss1)

            POS_LABEL = LABEL_UNSAFE
            NEG_LABEL = LABEL_SAFE

            precision = precision_score(label, prediction, pos_label=POS_LABEL)
            recall = recall_score(label, prediction, pos_label=POS_LABEL)
            precision_total.append(precision)
            recall_total.append(recall)

            # accuracy = balanced_accuracy_score(label, prediction)

            F1 = metrics.f1_score(label, prediction, pos_label=POS_LABEL, zero_division=0)
            if F1 == 0:
                F1 = 1
            F1_total.append(F1)
            if not tf.is_nan(accuracy):
                accuracy_total.append(accuracy)  # look at other metrics maybe auc
            else:
                tn = np.sum((label == NEG_LABEL) and (prediction == NEG_LABEL))
                accuracy_total.append(tn / np.sum(label == NEG_LABEL))
            if (not tf.is_inf(precision)) and not (tf.is_nan(precision)):
                precision_total.append(precision)
            if (not tf.is_inf(recall)) and (not tf.is_nan(recall)):
                recall_total.append(recall)
            # for label1 , prediction1 in zip(label, prediction):
            correct_count = np.sum((label == POS_LABEL) & (prediction == POS_LABEL))
            percentage = correct_count / np.sum(label == POS_LABEL)
            if (not tf.is_inf(percentage)) and (not tf.is_nan(percentage)):
                percentage_total.append(percentage)
        weights = tf.stack(sample_weights)

        if False:
            self.plot_5_decision_boundary(biases, kernel_weights, nn_output, normalize, prediction_total,
                                          processed_data,
                                          raw_data, remap_labels, sample, scales, stimators_coeffs)

        hinge_loss, mse_loss = self.extract_ave_losses(predicted, processed_data, remap_labels, weights=weights)

        if self.p.loss.loss_type == 'mse':
            prediction_loss = mse_loss
        elif self.p.loss.loss_type == 'l2_loss':
            prediction_loss = tf.nn.l2_loss(nn_output - processed_data['labels'])
        elif self.p.loss.loss_type == 'hinge':
            prediction_loss = hinge_loss
        elif self.p.loss.loss_type == 'mse_hinge':
            prediction_loss = mse_loss + hinge_loss

        percentage_mean = np.mean(np.array(percentage_total))
        # print("percentage of unsafe predicted correclty in this batch: " + str(percentage_mean))

        accuracy_mean = np.mean(np.array(accuracy_total))
        # print("accuracy total: " + str(accuracy_total))
        print("accuracy in this batch: " + str(accuracy_mean))

        precision_mean = np.mean(np.array(precision_total))
        print("precision in this batch: " + str(precision_mean))

        recall_mean = np.mean(np.array(recall_total))
        print("recall in this batch: " + str(recall_mean))

        F1_mean = np.mean(np.array(F1_total))
        print("F1 in this batch: " + str(F1_mean))

        total_loss = tf.cast(prediction_loss, dtype=tf.float32) + regularization_loss
        print("regularization_loss: " + str(regularization_loss.numpy()))
        print("prediction_loss: " + str(prediction_loss.numpy()))
        print("log_loss: ", mse_loss.numpy())
        print("hinge_loss: ", hinge_loss.numpy())
        print("bias: ", biases.numpy())
        print("scale: ", scales.numpy())

        regularization_loss_svm = 1e-2 * tf.nn.l2_loss(nn_output.numpy()[:, 1:])  # 1e-1
        regularization_loss = regularization_loss + regularization_loss_svm

        if return_loss_components_and_output:
            return regularization_loss, prediction_loss, total_loss, nn_output  # , grad_dir
        # elif return_loss_components:
        #     return regularization_loss, prediction_loss, total_loss
        elif return_loss_components:
            return regularization_loss, prediction_loss, total_loss, accuracy_mean, precision_mean, recall_mean, percentage_mean, F1_mean
        else:
            return total_loss  # , grad_dir
            # return regularization_loss

    def compute_loss_function(self, raw_data, param, c, is_training=None, return_loss_components=False,
                              return_loss_components_and_output=False):
        """
        Compute the loss function for a given dataset.
        """
        # Create the NN inputs and labels

        processed_data= self.create_nn_inputs_and_outputs(raw_data, is_training=is_training)


        # processed_data = tf.Variable(processed_data)

        nn_output, waypoint_scale, waypoint_bias = self.predict_nn_output(
            processed_data['inputs'] + [tf.constant([[1]], dtype=processed_data['inputs'][0].dtype)],
            is_training=is_training
        )

        print("nn after" + str(nn_output))

        feat_train_sc = processed_data['Action_waypoint']
        regularization_loss = 0.
        regularization_loss_kernel = 0.
        model_variables = self.get_trainable_vars()
        for model_variable in model_variables:
            regularization_loss += tf.nn.l2_loss(model_variable) # / model_variable.shape.num_elements()

        regularization_loss = self.p.loss.regn * regularization_loss

        accuracy_total = []
        prediction_total = []
        precision_total = []
        recall_total = []
        output_total = []
        percentage_total = []
        F1_total = []

        sample = 1  # 600 , 50

        biases = nn_output[:, :4]
        scales = nn_output[:, 4:8]
        kernel_weights = nn_output#[:, 8:]

        # debug
        biases = waypoint_bias  # tf.zeros_like(biases)
        scales = waypoint_scale  # tf.ones_like(scales)

        # X_norm = [normalize(X, bias, scale) for X, bias, scale in zip(feat_train_sc, biases, scales) ]
        X_norm = normalize(feat_train_sc, biases, scales)

        # poly = PolynomialFeatures(3)
        # X_kerneled = [  poly.fit_transform(X).astype(np.float32) for X in X_norm ]
        # X_kerneled = tf.convert_to_tensor(np.array(X_kerneled))

        X_kerneled = [
            self.polyfeatures(X) for X in X_norm
        ]
        X_kerneled = tf.stack(X_kerneled, axis=0)


        stimators_coeffs = []
        sample_weights = []

        kernel_losses = []
        for X, y, output in zip(X_kerneled, processed_data['labels'], kernel_weights):
            try:

                n_sample0 = np.size(np.where(y == 1)[0])
                n_sample1 = np.size(np.where(y == -1)[0])

                sample0_ratio = n_sample0 / (n_sample1 + n_sample0)
                print(f'initial sample0_ratio: {sample0_ratio}')
                if sample0_ratio > 0.75:
                    sample0_ratio = sample0_ratio ** 2
                elif sample0_ratio < 0.25:
                    sample0_ratio = sample0_ratio ** 0.5
                print(f'final sample0_ratio: {sample0_ratio}')
                sample1_ratio = 1 - sample0_ratio

                # debug
                # sample1_ratio = sample0_ratio = 1

                sample_weight = {-1: sample0_ratio, 1: sample1_ratio}
                kernel_regulation = 1e-2
                clf = svm.SVC(kernel='linear', class_weight=sample_weight, C=1.0/kernel_regulation)
                clf.fit(X[:, 1:], np.squeeze(y))
                stimators_coeff = np.concatenate((np.expand_dims(clf.intercept_, axis=1), clf.coef_), axis=1)

                sample_weight = np.array(
                    [sample0_ratio if i == -1 else sample1_ratio for i in y])
                if np.all(sample_weight == 0):
                    sample_weight[:] = 1.0

                sample_weight = sample_weight / np.sum(sample_weight) * y.shape[0]
                # sample_weight = np.ones_like(sample_weight)
                sample_weights.append(sample_weight)


            except ValueError:
                # if tf.reduce_mean([K.dot(tf.convert_to_tensor(tf.expand_dims(x1, axis=0), dtype=tf.float32), tf.expand_dims(output, axis=1)) [0][0] * y
                #                    for x1, output, y in zip(X_kerneled, nn_output,  processed_data['labels'])]) > 0 :
                clf.intercept_ = np.array([-100 if n_sample1 > n_sample0 else 100])
                clf.coef_1 = np.zeros((1, X.shape[1] - 1))
                stimators_coeff = np.concatenate((np.expand_dims(clf.intercept_, axis=1), clf.coef_1), axis=1)
                # else:
                #     stimators_coeff = np.expand_dims(-output, axis=0)
            stimators_coeff = tf.convert_to_tensor(stimators_coeff, dtype=tf.float32)
            stimators_coeffs.append(stimators_coeff)
            kernel_losses.append(self.cosine_distance(stimators_coeff, tf.expand_dims(output, axis=0), axis=-1))
            # kernel_losses.append(tf.nn.l2_loss(stimators_coeff - tf.expand_dims(output, axis=0)))



        predicted = [K.dot(x1, tf.expand_dims(output, axis=1)) for
                     x1, output in
                     zip(X_kerneled, kernel_weights)]

        LABEL_UNSAFE = 0
        LABEL_SAFE = 1
        def remap_labels(label):
            return (label + 1) / 2

        for prediction0, label, sample_weight in zip(predicted, processed_data['labels'], sample_weights):
            # prediction0 = prediction.numpy()

            label = remap_labels(label)

            prediction = tf.sigmoid(prediction0)
            # prediction = prediction0

            output_total.append(prediction)
            prediction = prediction.numpy()
            # reverse for scores calculation
            prediction_binary = np.zeros_like(prediction)
            prediction_binary[np.where(prediction >= 0.5)] = 1
            prediction_binary[np.where(prediction < 0.5)] = 0
            prediction = prediction_binary
            prediction_total.append(prediction)
            print('sample_weight: ', sample_weight)
            print("label: ", label.transpose())
            print("prediction: ", prediction.transpose())
            # print("logits: ", prediction0.numpy().transpose())
            accuracy = np.count_nonzero(prediction == label) / np.size(label)
            # accuracy_total.append(accuracy)
            # prediction_loss1 = tf.losses.mean_squared_error(prediction0,label)
            # prediction_total.append(prediction_loss1)

            POS_LABEL = LABEL_UNSAFE
            NEG_LABEL = LABEL_SAFE

            precision = precision_score(label, prediction, pos_label=POS_LABEL)
            recall = recall_score(label, prediction, pos_label=POS_LABEL)
            precision_total.append(precision)
            recall_total.append(recall)

            # accuracy = balanced_accuracy_score(label, prediction)

            F1 = metrics.f1_score(label, prediction, pos_label=POS_LABEL, zero_division=0)
            if F1 == 0:
                F1 = 1
            F1_total.append(F1)
            if not tf.is_nan(accuracy):
                accuracy_total.append(accuracy)  # look at other metrics maybe auc
            else:
                tn = np.sum((label == NEG_LABEL) and (prediction == NEG_LABEL))
                accuracy_total.append(tn / np.sum(label == NEG_LABEL))
            if (not tf.is_inf(precision)) and not (tf.is_nan(precision)):
                precision_total.append(precision)
            if (not tf.is_inf(recall)) and (not tf.is_nan(recall)):
                recall_total.append(recall)
            # for label1 , prediction1 in zip(label, prediction):
            correct_count = np.sum((label == POS_LABEL) & (prediction == POS_LABEL))
            percentage = correct_count / np.sum(label == POS_LABEL)
            if (not tf.is_inf(percentage)) and (not tf.is_nan(percentage)):
                percentage_total.append(percentage)
        weights = tf.stack(sample_weights)

        if False:
            self.plot_5_decision_boundary(biases, kernel_weights, nn_output, normalize, prediction_total, processed_data,
                                        raw_data, remap_labels, sample, scales, stimators_coeffs)

        hinge_loss, mse_loss= self.extract_ave_losses(predicted, processed_data, remap_labels, weights=weights)

        if self.p.loss.loss_type == 'mse':
            prediction_loss = mse_loss
        elif self.p.loss.loss_type == 'l2_loss':
            prediction_loss = tf.nn.l2_loss(nn_output - processed_data['labels'])
        elif self.p.loss.loss_type == 'hinge':
            prediction_loss = hinge_loss
        elif self.p.loss.loss_type == 'mse_hinge':
            prediction_loss = mse_loss + hinge_loss

        percentage_mean = np.mean(np.array(percentage_total))
        # print("percentage of unsafe predicted correclty in this batch: " + str(percentage_mean))

        accuracy_mean = np.mean(np.array(accuracy_total))
        # print("accuracy total: " + str(accuracy_total))
        print("accuracy in this batch: " + str(accuracy_mean))

        precision_mean = np.mean(np.array(precision_total))
        print("precision in this batch: " + str(precision_mean))

        recall_mean = np.mean(np.array(recall_total))
        print("recall in this batch: " + str(recall_mean))

        F1_mean = np.mean(np.array(F1_total))
        print("F1 in this batch: " + str(F1_mean))

        kernel_loss = 1e-2 * tf.reduce_mean(kernel_losses) # 1e-1
        total_loss = tf.cast(prediction_loss, dtype=tf.float32) + regularization_loss + kernel_loss
        print("kernel_losses: ", kernel_loss.numpy())
        print("regularization_loss: "+str(regularization_loss.numpy()))
        print("prediction_loss: " + str(prediction_loss.numpy()))
        print("log_loss: ", mse_loss.numpy())
        print("hinge_loss: ", hinge_loss.numpy())
        # print("bias: ", waypoint_bias.numpy())
        # print("scale: ", waypoint_scale.numpy())
        print("bias: ", biases.numpy())
        print("scale: ", scales.numpy())


        regularization_loss_svm = 1e-2 * tf.nn.l2_loss(nn_output.numpy()[:, 1:]) #1e-1
        regularization_loss = regularization_loss + regularization_loss_svm


        if return_loss_components_and_output:
            return regularization_loss, prediction_loss, total_loss, nn_output#, grad_dir
        # elif return_loss_components:
        #     return regularization_loss, prediction_loss, total_loss
        elif return_loss_components:
            return regularization_loss, prediction_loss, total_loss, accuracy_mean, precision_mean, recall_mean , percentage_mean, F1_mean
        else:
            return total_loss #, grad_dir
                # return regularization_loss



    def extract_ave_losses(self, predicted, processed_data, remap_labels, weights):
        mse_losses = [tf.reduce_mean(
            tf.losses.log_loss(remap_labels(label), tf.sigmoid(prediction0), tf.cast(weight, tf.float32))) for
            label, prediction0, weight in zip(processed_data['labels'], predicted,
                                              tf.expand_dims(tf.cast(weights, tf.float32),
                                                             axis=-1))]
        mse_loss = tf.reduce_mean(tf.boolean_mask(mse_losses, tf.is_finite(mse_losses)))
        # cce = tf.losses.log_loss ()
        hinge_losses = [tf.reduce_mean(tf.losses.hinge_loss(remap_labels(y), wx, sample_weight)) for
                        wx, y, sample_weight in
                        zip(predicted, processed_data['labels'],
                            tf.expand_dims(tf.cast(weights, tf.float32), axis=-1))]
        hinge_loss = tf.reduce_mean(hinge_losses)
        return hinge_loss, mse_loss


    def plot_2plots_decision_boundary(self, biases, kernel_weights, nn_output, normalize, processed_data,
                               raw_data, remap_labels, sample, scales, stamp=None):
        camera_pos_13 = raw_data['start_state'][:, :, 2:3]
        dx = self.p.simulator_params.reachability_map_params.dMax_avoid_xy
        camera_grid_world_pos_12 = raw_data['start_state'][:, :,
                                   :2] / self.p.simulator_params.reachability_map_params.dMax_avoid_xy
        dx = 0.05
        crop_size = [100, 100]
        if stamp is None:
            stamp = time() / 1e5
            stamp = f'{stamp:.5f}'

        for img_idx, (
         C1, kernel_weight, image, start_nk3, speed, robot_pos, robot_head
       ) in enumerate(zip(
            nn_output.numpy(),
            kernel_weights,
            raw_data['img_nmkd'][:, :, :, :3],
            raw_data['vehicle_state_nk3'],
            processed_data['inputs'][1], camera_grid_world_pos_12, camera_pos_13)):

            # pdf = PdfPages(f"output_all_{stamp}_{img_idx}.pdf")

            # camera_pos_13 = config.heading_nk1()[0]
            # camera_grid_world_pos_12 = config.position_nk2()[0] []/ dx_m

            # image of current state
            # rgb_image_1mk3 = r._get_rgb_image(robot_pos, robot_head)

            # img1 = r._get_topview(robot_pos, robot_head)

            X_grid, xx, yy, hh, ss = get_uniform_grid(crop_size, dx, speed)
            # ax1 = fig.add_subplot(221)
            min_x = X_grid.numpy()[0, :, 0].min()  # min(WP[:, 0].min(), X_grid.numpy()[0, :, 0].min()
            min_y = X_grid.numpy()[0, :, 1].min() # min(WP[:, 1].min(), X_grid.numpy()[0, :, 1].min()
            max_x = X_grid.numpy()[0, :, 0].max() # max(WP[:, 0].max(), X_grid.numpy()[0, :, 0].max()
            max_y = X_grid.numpy()[0, :, 1].max()  # max(WP[:, 1].max(), X_grid.numpy()[0, :, 1].max()

            # add some slack
            min_x -= 0.25
            min_y -= 0.25
            max_x += 0.25
            max_y += 0.25

            min_x, min_y = transform_to_vis_coord(min_x, min_y)
            max_x, max_y = transform_to_vis_coord(max_x, max_y)

            # min and max can be flipped during the transformation
            min_x, max_x = min(min_x, max_x), max(min_x, max_x)
            min_y, max_y = min(min_y, max_y), max(min_y, max_y)


            # X_grid = [normalize(X, bias) for X, bias in zip(X_grid, biases)]
            X_grid_norm = normalize(
                X_grid,
                biases[0:1],
                scales[0:1]
                # biases[img_idx:(img_idx + 1)],
                # scales[img_idx:(img_idx + 1)]
            )

            # X_grid = tf.expand_dims(X_grid,axis=0)
            # X_grid_kerneled = [poly.fit_transform(X).astype(np.float32) for X in X_grid]
            # X_grid_kerneled = np.array(X_grid_kerneled)
            X_grid_kerneled = [
                self.polyfeatures(tf.constant(X)) for X in X_grid_norm
            ]
            X_grid_kerneled = tf.stack(X_grid_kerneled, axis=0)

            # ta = tf.TensorArray(tf.float32, size=np.size(X_kerneled), dynamic_size=True, clear_after_read=False)
            # v = tf.contrib.eager.Variable(1, dtype=tf.float32)
            # for idx, i in X_kerneled:
            #     for jdx, j in i:
            #         for kdx, k in j:
            #             v.assign_add(k)
            #             ta = ta.write([idx, jdx, kdx], v)
            # X_grid_kerneled = ta
            # # X_grid_kerneled = tf.stack([self.poly_layer(tf.cast(X, tf.float32)) for X in X_grid], axis=0)
            # create loop with list of thresholds
            for threshold in np.arange(0.5, 0.8, 0.1):
                Z_pred = [tf.sigmoid(K.dot(tf.cast(x1, tf.float32), tf.expand_dims(output, axis=1))).numpy() > threshold
                          for x1, output in
                          zip(X_grid_kerneled, kernel_weight[np.newaxis, ...])]
                Z_pred = np.array(Z_pred)


                p = self.create_params(image.shape[1])


                # camera to world transformation
                T_world_cam = get_T_world_cam(p.projected_grid_params.tilt, p.projected_grid_params.h)
                # world to camera transformation
                T_cam_world = np.linalg.inv(T_world_cam)

                xyhs = X_grid.numpy()[0, :, :]
                uv = project_to_camera(
                    xyhs[:, 0], xyhs[:, 1],
                    T_cam_world, p.projected_grid_params.f, image.shape
                )

                x = uv[:, 0]
                y = uv[:, 1]

                image_int = image.astype(np.uint8)

                valid_indices_grid = (x >= 0) & (x < image.shape[1]) & (y >= 0) & (y < image.shape[0])
                valid_indices_grid = np.where(valid_indices_grid)[0]
                x = x[valid_indices_grid]
                y = y[valid_indices_grid]

                # tic = time()
                # # plt.imshow(np.squeeze(top))
                # fig = plt.figure()
                #
                # ax6 = fig.add_subplot(111)
                # ax6.imshow(image_int)
                # ax6.scatter(
                #     x.astype(int), y.astype(int),
                #     marker="o", s=15, alpha=0.25,
                #     color=['red' if i == -1 else 'green' for i in np.squeeze(Z_pred)[valid_indices_grid]]
                # )
                #
                # plt.savefig(f"./traj_imgs_area3_batch1/output_all_{stamp}_{img_idx}.png")
                # # pdf.savefig(fig)
                # # pdf.close()
                # plt.close('all')
                # print("time_matplot: ", (time() - tic))

                # tic = time()
                image_labels = np.zeros_like(image_int)
                # draw circles at Z_pred
                for x_, y_, label_ in zip(x.astype(int), y.astype(int), np.squeeze(Z_pred)[valid_indices_grid]):
                    color = (255, 0, 0) if label_ == 0 else (0, 255, 0)
                    image_labels = cv2.circle(image_labels, (x_, y_), 2, color, -1)
                alpha = 0.25
                image_out = cv2.addWeighted(image_int, 1 - alpha, image_labels, alpha, 0)
                image_out = cv2.resize(image_out, (512, 512))
                outdir = f'./traj_imgs_real_8_{threshold:.03f}/'
                Path(outdir).mkdir(exist_ok=True)
                cv2.imwrite(f"./{outdir}/output_all_{stamp}_{img_idx:05d}.png", image_out[..., ::-1])
                # print("time_cv2: ", (time() - tic))

    def plot_5_decision_boundary(self, biases, kernel_weights, nn_output, normalize, prediction_total, processed_data,
                               raw_data, remap_labels, sample, scales, stimators_coeffs):
        all_waypoint_sampled = [x[::sample, :] for x in raw_data['all_waypoint']]
        camera_pos_13 = raw_data['start_state'][:, :, 2:3]
        dx = self.p.simulator_params.reachability_map_params.dMax_avoid_xy
        camera_grid_world_pos_12 = raw_data['start_state'][:, :,
                                   :2] / self.p.simulator_params.reachability_map_params.dMax_avoid_xy

        # 2d plots
        # pdf = PdfPages(f"output_fov_sample40_FRS_4_{stamp:.2f}.pdf")
        # Vc= np.load('optimized_dp-master/V_safe2.npy')
        # V_safe2_wodisturb
        dx = 0.05
        crop_size = [100, 100]
        for img_idx, (
        WP, kernel_weight, prediction, label, C1, stimators_coeff, image, start_nk3, goal, traj, wp, speed, robot_pos, robot_head,
        value) in enumerate(zip(
            processed_data['Action_waypoint'], kernel_weights, prediction_total, processed_data['labels'],
            nn_output.numpy(),
            stimators_coeffs,
            raw_data['img_nmkd'][:, :, :, :3],
            raw_data['start_state'],

            raw_data['goal_position_n2'],
            raw_data['vehicle_state_nk3'],
            all_waypoint_sampled,
            processed_data['inputs'][1], camera_grid_world_pos_12, camera_pos_13, raw_data['value_function'])):

            stamp = time() / 1e5
            pdf = PdfPages(f"output_all_{stamp:.5f}.pdf")

            # camera_pos_13 = config.heading_nk1()[0]
            # camera_grid_world_pos_12 = config.position_nk2()[0] / dx_m

            # image of current state
            # rgb_image_1mk3 = r._get_rgb_image(robot_pos, robot_head)

            # img1 = r._get_topview(robot_pos, robot_head)

            # plt.imshow(np.squeeze(top))
            # plt.show()
            label = remap_labels(label)
            color = ['red' if l == 0 else 'green' for l in label]
            fig = plt.figure()

            # ax1 = fig.add_subplot(221)
            min_x = WP[:, 0].min()  # min(WP[:, 0].min(), X_grid.numpy()[0, :, 0].min()
            min_y = WP[:, 1].min()  # min(WP[:, 1].min(), X_grid.numpy()[0, :, 1].min()
            max_x = WP[:, 0].max()  # max(WP[:, 0].max(), X_grid.numpy()[0, :, 0].max()
            max_y = WP[:, 1].max()  # max(WP[:, 1].max(), X_grid.numpy()[0, :, 1].max()

            # add some slack
            min_x -= 0.25
            min_y -= 0.25
            max_x += 0.25
            max_y += 0.25

            X_grid, xx, yy, hh, ss = get_uniform_grid(crop_size, dx, speed, min_x, min_y, max_x, max_y)

            min_x, min_y = transform_to_vis_coord(min_x, min_y)
            max_x, max_y = transform_to_vis_coord(max_x, max_y)

            # min and max can be flipped during the transformation
            min_x, max_x = min(min_x, max_x), max(min_x, max_x)
            min_y, max_y = min(min_y, max_y), max(min_y, max_y)

            # fig = plt.figure()
            ax3 = fig.add_subplot(233)
            ax3.set_xlim(min_x, max_x)
            ax3.set_ylim(min_y, max_y)

            # obstacle_map = SBPDMap(self.p.simulator_params.obstacle_map_params)
            # obstacle_map.render(ax3)
            # start = start_nk3[0]
            # ax3.plot(start[0], start[1], 'k*')  # robot
            # goal_pos_n2 = goal
            # # ax3.plot(goal_pos_n2[0], goal_pos_n2[1], 'b*')
            # pos_nk2 = wp[:, :2]
            # # ax3.scatter(pos_nk2[:, 0], pos_nk2[:, 1], c=np.squeeze(label), marker='o', alpha=0.6, cmap=mycmap)
            # ax3.scatter(pos_nk2[:, 0], pos_nk2[:, 1], marker='o', alpha=0.6, color=color)

            # x = pos_nk2[:, 0]
            # y = pos_nk2[:, 1]
            # theta = wp[:, 2]
            # theta of the arrow
            # u, v = 1 * (np.cos(theta), np.sin(theta))

            # q = ax3.quiver(x, y, u, v)
            # plt.show()

            # u, v =  1 * (np.cos(theta_world), np.sin(theta_world))
            # q = ax3.quiver(pos_nk2[:, 0], pos_nk2[:, 1], u, v)
            # plt.show()
            ax4 = fig.add_subplot(234)
            ax4.set_xlim(min_x, max_x)
            ax4.set_ylim(min_y, max_y)

            # X_grid = [normalize(X, bias) for X, bias in zip(X_grid, biases)]
            X_grid_norm = normalize(
                X_grid,
                biases[0:1],
                scales[0:1]
                # biases[img_idx:(img_idx + 1)],
                # scales[img_idx:(img_idx + 1)]
            )

            # X_grid = tf.expand_dims(X_grid,axis=0)
            # X_grid_kerneled = [poly.fit_transform(X).astype(np.float32) for X in X_grid]
            # X_grid_kerneled = np.array(X_grid_kerneled)
            X_grid_kerneled = [
                self.polyfeatures(tf.constant(X)) for X in X_grid_norm
            ]
            X_grid_kerneled = tf.stack(X_grid_kerneled, axis=0)

            # ta = tf.TensorArray(tf.float32, size=np.size(X_kerneled), dynamic_size=True, clear_after_read=False)
            # v = tf.contrib.eager.Variable(1, dtype=tf.float32)
            # for idx, i in X_kerneled:
            #     for jdx, j in i:
            #         for kdx, k in j:
            #             v.assign_add(k)
            #             ta = ta.write([idx, jdx, kdx], v)
            # X_grid_kerneled = ta
            # # X_grid_kerneled = tf.stack([self.poly_layer(tf.cast(X, tf.float32)) for X in X_grid], axis=0)
            Z_pred = [np.sign(K.dot(tf.cast(x1, tf.float32), tf.expand_dims(output, axis=1))) for
                      x1, output in
                      zip(X_grid_kerneled, kernel_weight[np.newaxis, ...])]
            Z_pred = np.array(Z_pred)

            ax3.contourf(
                *transform_to_vis_coord(xx + 0, yy),
                np.reshape(np.squeeze(Z_pred), np.shape(xx)),
                cmap=plt.get_cmap("RdBu"), alpha=0.5)
            ax3.scatter(
                *transform_to_vis_coord(WP[:, 0], WP[:, 1]),
                marker='o', alpha=0.6, color=color
            )
            x = WP[:, 0]
            y = WP[:, 1]
            theta = WP[:, 2]  # theta of the arrow
            u, v = 1 * (np.cos(theta), np.sin(theta))
            ax3.quiver(
                *transform_to_vis_coord(x, y), *transform_to_vis_coord(u, v)
            )
            ax3.set_title("Our model")

            Z_svm = [np.sign(K.dot(tf.cast(x1, tf.float32), tf.reshape(output, [-1, 1]))) for
                     x1, output in
                     zip(X_grid_kerneled, stimators_coeffs)]
            Z_svm = np.array(Z_svm)
            ax4.contourf(
                *transform_to_vis_coord(xx + 0, yy),
                np.reshape(np.squeeze(Z_svm), np.shape(xx)),
                cmap=plt.get_cmap("RdBu"), alpha=0.5
            )

            ax4.scatter(*transform_to_vis_coord(WP[:, 0], WP[:, 1]),
                        marker='o', alpha=0.6, color=color)
            ax4.quiver(*transform_to_vis_coord(x, y), *transform_to_vis_coord(u, v))
            ax4.set_title("SVM")

            # commented for better plot
            # x = WP[:, 0]
            # y = WP[:, 1]
            # theta = WP[:, 2]
            # u, v = 1 * (np.cos(theta), np.sin(theta))
            # accuracy = np.count_nonzero(prediction == label) / np.size(label)
            # color_result = ['red' if l == 1 else 'green' for l in prediction]
            # wrong = WP[np.where(prediction != label)[0]]
            # ax4.scatter(wrong[:, 0], wrong[:, 1], s=60, edgecolors="k")
            # ax4.scatter(x, y
            #             , marker='o', alpha=0.6, color=color_result)
            # # ax2.scatter3D(wrong[:, 0], wrong[:, 1],
            # #               wrong[:, 2], s=80, edgecolors="k")
            # # safe = WP[np.where(prediction == 1)[0]]
            # # unsafe = WP[np.where(prediction == -1)[0]]
            # ax4.set_title('accuracy: ' + str(accuracy))
            # # ax4.scatter(WP[:, 0], WP[:, 1]
            # #             , c=np.squeeze(prediction), marker='o', alpha=0.6, cmap=mycmap)
            # q1 = ax4.quiver(x, y, u, v)

            # ax4.scatter(safe[:, 0], safe[:, 1], s=80, edgecolors="g")
            # ax4.scatter(unsafe[:, 0], unsafe[:, 1], s=80, edgecolors="r")

            # plt.show()

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

            ax5 = fig.add_subplot(235)
            x = WP[:, 0:1]
            x1 = np.expand_dims(x, axis=2)
            y = WP[:, 1:2]
            y1 = np.expand_dims(y, axis=2)
            t = WP[:, 2:3]
            t1 = np.expand_dims(t, axis=2)
            #
            # x1= np.expand_dims(np.expand_dims(np.expand_dims(x, axis=0),axis=1), axis=2)
            # y1 = np.expand_dims( np.expand_dims(np.expand_dims(y, axis=0), axis=1), axis=2)
            # t1 = np.expand_dims( np.expand_dims(np.expand_dims(t, axis=0), axis=1), axis=2)
            p = self.create_params(image.shape[1])
            plt.imshow(image.astype(np.uint8))

            # camera to world transformation
            T_world_cam = get_T_world_cam(p.projected_grid_params.tilt, p.projected_grid_params.h)
            # world to camera transformation
            T_cam_world = np.linalg.inv(T_world_cam)

            uv = project_to_camera(x, y, T_cam_world, p.projected_grid_params.f, image.shape)

            x = uv[:, 0]
            y = uv[:, 1]

            valid_indices = (x >= 0) & (x < image.shape[1]) & (y >= 0) & (y < image.shape[0])
            valid_indices = np.where(valid_indices)[0]
            # valid_indices = valid_indices[:10]
            x = x[valid_indices]
            y = y[valid_indices]
            label_valid = label[valid_indices]
            color_valid = ['red' if l == 0 else 'green' for l in label_valid]

            ax5.scatter(x.astype(int), y.astype(int), marker="x", s=10, color=color_valid)
            ax5.set_title("First-person view")

            image_int = image.astype(np.uint8)
            ax5.imshow(image_int)

            theta = traj[valid_indices, -1, 2]
            uv = project_theta_to_cam(theta, T_cam_world)
            ax5.quiver(x, y, 10 * uv[:, 0], 10 * uv[:, 1])

            xyhs = X_grid.numpy()[0, :, :]
            uv = project_to_camera(
                xyhs[:, 0], xyhs[:, 1],
                T_cam_world, p.projected_grid_params.f, image.shape
            )

            x = uv[:, 0]
            y = uv[:, 1]

            valid_indices_grid = (x >= 0) & (x < image.shape[1]) & (y >= 0) & (y < image.shape[0])
            valid_indices_grid = np.where(valid_indices_grid)[0]
            x = x[valid_indices_grid]
            y = y[valid_indices_grid]

            ax6 = fig.add_subplot(236)
            ax6.imshow(image_int)
            ax6.scatter(
                x.astype(int), y.astype(int),
                marker="o", s=1, alpha=0.25,
                color=['red' if i == -1 else 'green' for i in np.squeeze(Z_pred)[valid_indices_grid]]
            )

            # ax5.contourf(
            #     x, y,
            #     Z_svm.reshape((-1,))[valid_indices_grid],
            #     cmap=plt.get_cmap("RdBu"), alpha=0.5
            # )

            ax1 = fig.add_subplot(231)
            top = self.renderer._get_topview(robot_pos, robot_head, crop_size)
            top_cw_90 = cv2.rotate(np.squeeze(top), cv2.ROTATE_90_COUNTERCLOCKWISE)



            ax1.imshow(top_cw_90)

            ## plotting traj
            # list = np.where(label == -1)[0]
            # pdf.savefig(fig)
            plt.grid()
            # plt.show()

            traj_x, traj_y = transform_to_vis_coord(
                traj[valid_indices, :, 0], traj[valid_indices, :, 1]
            )
            traj_x = (traj_x / dx + 0) + ((crop_size[1] - 1) / 2)
            traj_y = (crop_size[0] - 1) - (traj_y / dx)

            theta = (np.pi / 2) + traj[valid_indices, :, 2]
            j = 0
            for i, _ in enumerate(traj_x):
                # s = 1  # Segment length
                u, v = 10 * np.cos(theta[i, -1]), 10 * np.sin(theta[i, -1])

                # print ("value: ", str(value[i,-1]))
                q = ax1.quiver(traj_x[i, -1], traj_y[i, -1], u, v)
                ax1.set_title(f'Top-down, speed:{speed[0]:.3f}')
                # robot is facing right
                # ax1.plot(traj_x[i], traj_y[i])
                # ax1.scatter([traj_x[i][-1]], [traj_y[i][-1]], marker="x", s=10, color=color_valid[i])
                # robot is facing front
                ax1.plot(traj_x[i], traj_y[i])
                ax1.scatter([traj_x[i][-1]], [traj_y[i][-1]], marker="x", s=10, color=color_valid[i])
                # plt.annotate(np.min(value[i, :]), xy=(traj_x[i, -1], traj_y[i, -1] + 0.5))

            theta = np.pi / 2 + WP[valid_indices, 2:3]  # theta of the arrow
            u, v = 1 * (np.cos(theta), np.sin(theta))
            # ax1.set_title('v , w: ' + str(control))
            # plt.savefig('/tmp/plot.png')

            # matplotlib.use('Qt4Agg')
            # fig = plt.figure()

            # ax2 = fig.add_subplot(222, projection='3d')
            ax2 = fig.add_subplot(232)
            ax2.set_xlim(min_x, max_x)
            ax2.set_ylim(min_y, max_y)
            # prediction = prediction.numpy()
            # prediction[np.where(prediction >= 0)] = 1
            # prediction[np.where(prediction < 0)] = -1
            # wrong = WP[np.where(prediction != label)[0]]
            # ax2.scatter3D(wrong[:, 0], wrong[:, 1],
            #               wrong[:, 2], s=80, edgecolors="k")
            # ax2.scatter(wrong[:, 0], wrong[:, 1], s=80, edgecolors="k")

            # mycmap = ListedColormap(["red", "green"])

            # ax2.scatter3D(WP[:, 0], WP[:, 1],
            #               WP[:, 2], c=np.squeeze(label), marker='o', alpha=0.6, cmap=mycmap)
            # ax2.scatter(WP[:, 0], WP[:, 1]
            #               , c=np.squeeze(label), marker='o', alpha=0.6, cmap=mycmap)

            ax2.scatter(
                *transform_to_vis_coord(WP[:, 0], WP[:, 1]),
                marker='o', alpha=0.6, color=color
            )
            # ax2.matplotlib.pyplot.arrow(WP[:, 0], WP[:, 1], math.cos(WP[:, 2]), math.sin(WP[:, 2]))
            x = WP[:, 0]
            y = WP[:, 1]
            theta = WP[:, 2]  # theta of the arrow
            u, v = 1 * (np.cos(theta), np.sin(theta))

            ax2.scatter(
                *transform_to_vis_coord(x[valid_indices], y[valid_indices]),
                s=80, facecolors='none', edgecolors='c'
            )

            q = ax2.quiver(
                *transform_to_vis_coord(x, y),
                *transform_to_vis_coord(u, v)
            )
            ax2.set_title('Ground truth')
            # plt.xlim(-0.5, len(x[0]) - 0.5)
            # plt.ylim(-0.5, len(x) - 0.5)
            # plt.xticks(range(len(x[0])))
            # plt.yticks(range(len(x)))

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

            # plt.show(
            pdf.savefig(fig)
            pdf.close()
        plt.close('all')

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
    @staticmethod
    def sig(x):
        return 1 / (1 + tf.math.exp(-x))

def standardize(X_tr):
    for i in range(np.shape(X_tr)[1]):
        X_tr[:,i] = (X_tr[:,i] - np.mean(X_tr[:,i]))/np.std(X_tr[:,i])
    return X_tr


def bin_cross_entropy(p, q):
    n = len(p)
    return -sum(p[i]*log(q[i]+1e-9) + (1-p[i])*log(1-q[i] + 1e-9) for i in range(n)) / n

def get_T_world_cam(tilt, height):
    T_world_cam = np.eye(4)
    T_world_cam[:3, :3] = depth_utils.get_r_matrix([1., 0., 0.], angle=tilt)
    T_world_cam[1, -1] = height

    return T_world_cam


def transform_grid_to_world(x, y):
    xyzw = np.zeros((x.shape[0], 4))
    # In grid coordinate: X = front, Y = left, Z = up
    # In world coordinate: X = right, Y = up, Z = back
    xyzw[:, 0] = -np.squeeze(y)
    xyzw[:, 2] = -np.squeeze(x)
    xyzw[:, 3] = 1

    return xyzw


def project_to_camera(x, y, T_cam_world, f, image_shape):
    xyzw = transform_grid_to_world(x, y)

    xyz_cam = (T_cam_world @ xyzw.T).T[:, :3]
    uv = xyz_cam[:, :2] / -xyz_cam[:, 2:3]
    # OpenGL coord to OpenCV
    uv[:, 1] *= -1
    uv[:, 0] = (f * uv[:, 0]) + (image_shape[1] - 1) / 2.0
    uv[:, 1] = (f * uv[:, 1]) + (image_shape[0] - 1) / 2.0

    return uv


def project_theta_to_cam(theta, T_cam_world):
    u, v = np.cos(theta), np.sin(theta)
    uvw = transform_grid_to_world(u, v)[:, :3]
    uvw = (T_cam_world[:3, :3] @ uvw.T).T
    uv = uvw[:, :2]
    uv = uv / np.linalg.norm(uv, axis=-1, keepdims=True)

    return uv


def fov_to_focal(fov_half, width):
    return width / 2 / np.tan(fov_half)


def transform_to_vis_coord(x, y):
    # compatible with FPV RGB image (robot front along y-axis instead of x)
    return -y, x


def get_uniform_grid(crop_size, dx, speed, min_x=None, min_y=None, max_x=None, max_y=None, theta=0):
    x_min, x_max = 0, crop_size[0] * dx
    y_min, y_max = -dx * (crop_size[0]) / 2, dx * (crop_size[0]) / 2

    # prevent out of bound from waypoints
    if min_x is not None:
        x_min = max(x_min, min_x)
    if max_x is not None:
        x_max = min(x_max, max_x)
    if min_y is not None:
        y_min = max(y_min, min_y)
    if max_y is not None:
        y_max = min(y_max, max_y)

    # add some slack
    x_min -= 0.25
    y_min -= 0.25
    x_max += 0.25
    y_max += 0.25
    h = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    hh, ss = np.tile(theta, np.shape(xx)), np.tile(speed, np.shape(xx)),
    X_grid = np.c_[xx.ravel(), yy.ravel(), hh.ravel(), ss.ravel()]
    X_grid = tf.expand_dims(X_grid, axis=0)

    return X_grid, xx, yy, hh, ss


if __name__ == '__main__':
    import open3d as o3d

    T_world_cam = get_T_world_cam(np.deg2rad(-45), 0.8)
    T_cam_world = np.linalg.inv(T_world_cam)

    crop_size = [100, 100]
    dx = 0.05
    x_min, x_max = 0, crop_size[0] * dx
    y_min, y_max = -dx * (crop_size[0]) / 2, dx * (crop_size[0]) / 2
    # add some slack
    x_min -= 0.25
    y_min -= 0.25
    x_max += 0.25
    y_max += 0.25
    h = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    xyz = np.zeros((X_grid.shape[0], 3))
    xyz[:, 0] = -X_grid[:, 1]
    xyz[:, 2] = -X_grid[:, 0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud('/tmp/pcd.ply', pcd)

    camera = o3d.geometry.TriangleMesh.create_coordinate_frame(1).transform(T_world_cam)
    o3d.io.write_triangle_mesh('/tmp/camera.ply', camera)

    o3d.visualization.draw_geometries([pcd, camera])
    exit(0)

    tf.enable_eager_execution()
    X = tf.constant(np.random.rand(20, 4).astype(np.float32))
    # X = tf.constant(np.asarray(range(1, 5)).reshape(1, 4).astype(np.float32))
    feats = BaseModel.polyfeatures(X, degree=3)
    print('inp', X.numpy())
    print('out\n', feats.numpy())

    poly = PolynomialFeatures(3)
    X_kerneled = poly.fit_transform(X).astype(np.float32)
    print(f'{X_kerneled}')
    print(X_kerneled - feats.numpy())
    print(np.mean(X_kerneled - feats.numpy()))
