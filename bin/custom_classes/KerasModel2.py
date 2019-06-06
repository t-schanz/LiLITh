from keras.layers import Input, Conv2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
import logging
from functools import partial
import keras.backend as K
from keras.layers.advanced_activations import LeakyReLU
import logging


class ModelStrucure(object):
    def __init__(self):
        pass

    def build_model(self, in_shape):
        nout = 2

        in_layer = Input(in_shape)
        logging.debug(f"Input_shape: {K.int_shape(in_layer)}")

        layer = Conv2D(filters=nout, kernel_size=(3, 3), padding="SAME", activation="linear")(in_layer)
        layer = LeakyReLU()(layer)
        logging.debug(f"After first convolution: {K.int_shape(layer)}")

        go_on = True
        while go_on:
            try:
                t_layer = Conv2D(filters=nout, kernel_size=(3, 3), padding="SAME", activation="linear")(layer)
                t_layer = LeakyReLU()(t_layer)

                t_layer = MaxPooling2D(pool_size=(2, 2))(t_layer)

                nout *= 2
                layer = t_layer
                logging.debug(f"After pooling: {K.int_shape(layer)}")

            except ValueError:
                go_on = False
                logging.debug(f"Final pooling shape: {K.int_shape(layer)}")

        layer = Conv2D(filters=nout, kernel_size=(2, 2), padding="SAME", activation="linear")(layer)
        layer = LeakyReLU()(layer)

        output = Conv2D(filters=1, kernel_size=(1, 1), activation="tanh", padding="SAME", strides=(1, 1))(layer)
        logging.debug(f"Output shape: {K.int_shape(layer)}")

        model = Model(inputs=in_layer, outputs=output)

        return model

    def compile(self, model):
        model.compile(loss="mse", optimizer=self.optimizer(),
                      metrics=['mae'])
        logging.info("Successfully compiled model.")
        return model

    @staticmethod
    def optimizer():
        opt = Adam(lr=0.001)
        return opt


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    MS = ModelStrucure()
    model = MS.build_model((224, 224, 1))
    model = MS.compile(model)
    print(model.summary())

