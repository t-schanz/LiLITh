from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from keras.models import Model, Sequential
from keras.optimizers import Adam
import keras.backend as K
from keras.layers.advanced_activations import LeakyReLU
import logging


class ModelStrucure(object):
    def __init__(self):
        pass

    @staticmethod
    def __build_MLP(in_shape, regress=False):
        model = Sequential()
        model.add(Dense(8, input_dim=in_shape, activation="relu"))
        model.add(Dense(4, activation="relu"))

        if regress:
            model.add(Dense(1, activation="linear"))

        return model

    @staticmethod
    def __build_CNN(in_shape):
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

        layer = Flatten()(layer)
        layer = Dense(nout, kernel_initializer="normal")(layer)
        output = Dense(1, kernel_initializer="normal")(layer)
        logging.debug(f"Output shape: {K.int_shape(layer)}")

        model = Model(inputs=in_layer, outputs=output)

        return model

    def build_model(self, CNN_shape, MLP_shape):
        CNN = self.__build_CNN(CNN_shape)
        MLP = self.__build_MLP(MLP_shape)

        input = concatenate([MLP.output, CNN.output])
        layer = Dense(4, activation="relu")(input)
        output = Dense(1, activation="linear")(layer)

        model = Model(inputs=[MLP.input, CNN.input], outputs=output)

        return model

    def compile(self, model):
        model.compile(loss="mean_absolute_percentage_error", optimizer=self.optimizer(),
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
    model = MS.build_model((224, 224, 1), 4)
    model = MS.compile(model)
    print(model.summary())


    from keras.utils import plot_model

    plot_model(model, to_file="model.png", show_shapes=True)