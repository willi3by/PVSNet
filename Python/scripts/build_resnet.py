from tensorflow import keras
import tensorflow as tf

class ResidualUnit(keras.layers.Layer):

    def __init__(self, filters=1, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv3D(filters, (3, 3, 3), strides=strides,
                                padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv3D(filters, (3, 3, 3), strides=1,
                                padding="same", use_bias=False),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv3D(filters, (1, 1, 1), strides=strides,
                                    padding="same", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.2)),
                keras.layers.BatchNormalization()]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)

    def get_config(self):
        base_config = super(ResidualUnit, self).get_config()
        return base_config


def build_ResNet(numResNet, input_shape):

    if numResNet == "ResNet50":
        filter_list = [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3
    elif numResNet == "ResNet101":
        filter_list = [64] * 3 + [128] * 4 + [256] * 23 + [512] * 3
    elif numResNet == "ResNet152":
        filter_list = [64] * 3 + [128] * 8 + [256] * 36 + [512] * 3

    model = keras.models.Sequential()
    model.add(keras.layers.Conv3D(64, (7, 7, 7), strides=(2, 2, 2), padding="same",
                                  use_bias=False, input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="same"))
    prev_filters = 64
    for filters in filter_list:  # ResNet 50
        strides = 1 if filters == prev_filters else 2
        model.add(ResidualUnit(filters, strides=strides))
        prev_filters = filters

    model.add(keras.layers.GlobalAveragePooling3D())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.AlphaDropout(0.5))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  metrics=['accuracy', tf.keras.metrics.AUC()])

    return model

def build_ResNet50(input_shape):
    model = build_ResNet("ResNet50", input_shape)
    return model

def build_ResNet101(input_shape):
    model = build_ResNet("ResNet101", input_shape)
    return model

def build_ResNet152(input_shape):
    model = build_ResNet("ResNet152", input_shape)
    return model
