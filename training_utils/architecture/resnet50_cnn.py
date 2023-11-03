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
        resnet50.trainable = params.finetune_resnet_weights # True

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
                    name='dim_red_conv',
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
    x = layers.Flatten(name='img_flatten')(x)
    # input_flat1 = simple_mlp(num_inputs=1, num_outputs=1, params=params)(input_flat)
    # x = layers.Concatenate(axis=1)([x, input_flat1])
     # = simple_mlp(input_flat)
    # Concatenate the image and the flat outputs

    # x = layers.Add(name='add_input_flat')([x,input_flat])
    x = layers.Concatenate(axis=1)(
        [
            x,
            layers.Concatenate(axis=1)([input_flat]*x.shape[-1])
        ]
    )
    num_neurons = [1024, 512, 256, 256, 128, 64]
    # Fully connectecd hidden layers
    for i in range(params.num_hidden_layers):
        x = layers.Dense(
            num_neurons[i], # params.num_neurons_per_layer,
            name=f'fc_{i}',
            activation=params.hidden_layer_activation_func
        )(x)
        if params.use_dropout:
            x = layers.Dropout(rate=params.dropout_rate, name=f'fc_dropout_{i}')(x)

    # Output layer
    x = layers.Dense(num_outputs, activation=params.output_layer_activation_func, name='output_layer')(x)

    # Generate a Keras model


    model = tf.keras.Model(inputs=[input_image, input_flat], outputs=x)
    # model = tf.keras.Model(inputs=[input_image], outputs=x)
    model.load_weights(params.resnet50_weights_path, by_name=True)
    # model.load_weights(params.resnet50_weights_path)
    # model.summary()


    return model, is_training
