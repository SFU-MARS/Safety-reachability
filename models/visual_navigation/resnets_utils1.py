def build(sample, frame, height, width, channels,  classes):
    model = Sequential()
    inputShape = (sample, frame, height, width, channels)
    chanDim = -1

    if K.image_data_format() == "channels_first":
        inputShape = (sample, frame, channels, height, width)
        chanDim = 1


    model.add(Conv3D(32, (3, 3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding="same", data_format="channels_last"))
    model.add(Dropout(0.25))

    model.add(Conv3D(64, (3, 3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding="same", data_format="channels_last"))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))    #(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax")
