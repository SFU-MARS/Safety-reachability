import tensorflow as tf
from training_utils.architecture.resnet50.resnet_50 import ResNet50
from training_utils.architecture.simple_mlp import simple_mlp

from tensorflow.keras import datasets, layers, models
from tensorflow.keras import Input
layers = tf.keras.layers
from tensorflow.keras.layers import Dense, Dropout , Concatenate, Input
from tensorflow.keras.models import Model

def resnet50_cnn(image_size, num_inputs, num_outputs, params, dtype=tf.float32):
    # Input layers

    input_image = layers.Input(shape=(image_size[0], image_size[1], image_size[2]), dtype=dtype)
    input_flat = layers.Input(shape=(num_inputs,), dtype=dtype)
    x = input_image


    # Load the ResNet50 and restore the imagenet weights
    # Note (Somil): We are initializing ResNet model in this fashion because directly setting the layers.trainable to
    # false is buggy in Keras applications for the Batch Normalization layer. See these issues for details:
    # https://github.com/keras-team/keras/pull/9965
    # http://blog.datumbox.com/the-batch-normalization-layer-of-keras-is-broken/
    with tf.variable_scope('resnet50'):
        resnet50 = ResNet50(data_format='channels_last',
                            name='resnet50',
                            include_top=False,
                            pooling=None)
        resnet50.trainable = False

        # Used to control batch_norm during training vs test time
        is_training = tf.contrib.eager.Variable(True, dtype=tf.bool, name='is_training')
        x = resnet50.call(x, is_training,
                          output_layer=params.resnet_output_layer)

        # for layer in x.layers:
        #     layer.trainable = False
        #
        # for layer in x.layers[-26:]:
        #         layer.trainable = True

    # Optional strided convolution on the output
    # of the Resnet50 to reduce feature dimensionality
    if params.dim_red_conv_2d.use:
        # Convolutional layer
        x = layers.Conv2D(
                    filters=params.dim_red_conv_2d.num_outputs,
                    kernel_size=params.dim_red_conv_2d.filter_size,
                    strides=params.dim_red_conv_2d.stride,
                    padding=params.dim_red_conv_2d.padding,
                    activation=params.hidden_layer_activation_func)(x)
        # Max-pooling layer
        if params.dim_red_conv_2d.use_maxpool:
            x = layers.MaxPool2D(pool_size=(params.dim_red_conv_2d.size_maxpool_filters,
                                            params.dim_red_conv_2d.size_maxpool_filters),
                                 padding='valid')(x)

    # Flatten the image
    x = layers.Flatten()(x)
     # = simple_mlp(input_flat)
    # Concatenate the image and the flat outputs
    x = layers.Concatenate(axis=1)([x, input_flat])

    # Fully connectecd hidden layers
    for i in range(params.num_hidden_layers):
        x = layers.Dense(params.num_neurons_per_layer, activation=params.hidden_layer_activation_func)(x)
        if params.use_dropout:
            x = layers.Dropout(rate=params.dropout_rate)(x)

    # Output layer
    x = layers.Dense(num_outputs, activation=params.output_layer_activation_func)(x)

    # Generate a Keras model


    model = tf.keras.Model(inputs=[input_image, input_flat], outputs=x)
    # model = tf.keras.Model(inputs=[input_image], outputs=x)
    model.load_weights(params.resnet50_weights_path, by_name=True)
    # model.load_weights(params.resnet50_weights_path)
    # model.summary()
    #
    # # Load the Resnet50 weights
    # model.load_weights(params.resnet50_weights_path, by_name=True)
    #
    # resnet50.build(input_shape=(None, 224, 224, 3))
    # # print(resnet50.layers[-1])
    # # resnet50.summary()
    #
    # model1 = models.Sequential()
    #
    # input1=model1.add(layers.InputLayer(input_shape=(224, 224, 180), name="input_layer1"))
    # # input2=model1.add(layers.InputLayer(input_shape=(1, 1, 120), name="input_layer2"))
    # # inputs = tf.keras.layers.Dot(axes=1)([input1, input2])
    # # model1.add(layers.Dense(3,input_shape=(224, 224, 180)))
    # model1.add(layers.Conv2D(90, (3, 3),
    #     strides=(1, 1),
    #     padding='valid',
    #      input_shape=(224, 224, 180)))
    # # model1.add(layers.ZeroPadding2D(padding=(2, 2)))
    # model1.add(layers.Conv2D(20, (3, 3),
    #     strides=(1, 1),
    #     padding='valid'
    #      ))
    # model1.add(layers.Conv2D(3, (3, 3),
    #     strides=(1, 1),
    #     padding='valid'
    #     ))
    #
    # model1.add(layers.ZeroPadding2D(padding=(2, 2)))
    # model1.add(layers.ZeroPadding2D(padding=(1, 1)))
    # # model1.summary()
    #
    # for layer in resnet50.layers[:-1]:
    # # for layer in model.layers[:-15]:
    # #     print(layer)
    #     model1.add(layer)
    #
    #     # model1.summary()
    # # print(len(model1.layers))
    # # print(model1.layers[-1])
    #
    # model1.build(input_shape=(224,224,180))
    #
    # # model1.summary()
    # # model1.compile(optimizer='adam')
    #
    # # model1.add(layers.MaxPooling2D(2, 2))
    #
    #
    # model3 = models.Sequential()
    # input2 = model3.add(layers.InputLayer(input_shape=(120,), name="input_layer2"))
    # model3.add(layers.Dense(60))
    # model3.add(layers.Dense(20))
    # model3.add(layers.Dense(2))
    # model3.build(input_shape=(120,))
    # # model3.summary()
    #
    #
    # input_image1 = layers.Input(shape=(224, 224, 180), dtype=dtype)
    # input_flat1 = layers.Input(shape=(120,), dtype=dtype)

    # input_image1 = Input(shape=(224, 224, 180), dtype=dtype)
    # input_flat1 = Input(shape=(120,), dtype=dtype)

    # x1 = input_image1
    # x1 = model1.call(x1)
    # if params.dim_red_conv_2d.use:
    #     # Convolutional layer
    #     x1 = layers.Conv2D(
    #                 filters=params.dim_red_conv_2d.num_outputs,
    #                 kernel_size=params.dim_red_conv_2d.filter_size,
    #                 strides=params.dim_red_conv_2d.stride,
    #                 padding=params.dim_red_conv_2d.padding,
    #                 activation=params.hidden_layer_activation_func)(x1)
    #     # Max-pooling layer
    #     if params.dim_red_conv_2d.use_maxpool:
    #         x1 = layers.MaxPool2D(pool_size=(params.dim_red_conv_2d.size_maxpool_filters,
    #                                         params.dim_red_conv_2d.size_maxpool_filters),
    #                              padding='valid')(x1)
    #
    # # # model1.add(model)
    # # x1 = model.call(x1)
    # x1 = layers.Flatten()(x1)
    # input_flat2 = model3.call(input_flat1)
    # x1 = layers.Concatenate(axis=1)([x1, input_flat2])
    #
    # for i in range(params.num_hidden_layers):
    #     x1 = layers.Dense(params.num_neurons_per_layer, activation=params.hidden_layer_activation_func)(x1)
    #     if params.use_dropout:
    #         x1 = layers.Dropout(rate=params.dropout_rate)(x1)
    #
    # # Output layer
    # x1 = layers.Dense(num_outputs, activation=params.output_layer_activation_func)(x1)
    #
    # # Generate a Keras model
    #
    # model4 = tf.keras.Model(inputs=[input_image1, input_flat1], outputs=x1)
    # model4.summary()
    # # model4.compile(
    # #     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    # #     loss='mean_absolute_error')

    return model, is_training
