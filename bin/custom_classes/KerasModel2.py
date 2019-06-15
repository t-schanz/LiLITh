from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, BatchNormalization, Dropout
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
        logging.debug(f"MLP input shape: {in_shape}")
        model = Sequential()
        model.add(Dense(8, input_dim=in_shape, activation="relu"))
        model.add(Dense(5, activation="relu"))

        if regress:
            model.add(Dense(1, activation="linear"))

        return model

    @staticmethod
    def __build_CNN(in_shape, regress=False):
        nout = 2

        in_layer = Input(in_shape)
        logging.debug(f"Input_shape: {K.int_shape(in_layer)}")

        layer = Conv2D(filters=nout, kernel_size=(3, 3), padding="SAME", activation="linear")(in_layer)
        layer = BatchNormalization()(layer)
        layer = LeakyReLU()(layer)
        logging.debug(f"After first convolution: {K.int_shape(layer)}")

        go_on = True
        while go_on:
            try:
                t_layer = Conv2D(filters=nout, kernel_size=(3, 3), padding="SAME", activation="linear")(layer)
                t_layer = BatchNormalization()(t_layer)
                t_layer = LeakyReLU()(t_layer)

                t_layer = MaxPooling2D(pool_size=(2, 2))(t_layer)

                nout *= 2
                layer = t_layer
                logging.debug(f"After pooling: {K.int_shape(layer)}")

            except ValueError:
                go_on = False
                logging.debug(f"Final pooling shape: {K.int_shape(layer)}")

        layer = Conv2D(filters=nout, kernel_size=(2, 2), padding="SAME", activation="linear")(layer)
        layer = BatchNormalization()(layer)
        layer = LeakyReLU()(layer)

        layer = Flatten()(layer)
        layer = Dense(5, activation="relu")(layer)

        if regress:
            layer = Dense(1, activation="linear")(layer)
        logging.debug(f"Output shape: {K.int_shape(layer)}")

        model = Model(inputs=in_layer, outputs=layer)

        return model

    def build_model(self, CNN_shape, MLP_shape):
        CNN = self.__build_CNN(CNN_shape)
        MLP = self.__build_MLP(MLP_shape)

        input = concatenate([CNN.output, MLP.output])
        layer = Dropout(0.5)(input)
        layer = Dense(4, activation="relu")(layer)
        output = Dense(1, activation="linear")(layer)

        model = Model(inputs=[CNN.input, MLP.input], outputs=output)

        return model

    def compile(self, model):
        model.compile(loss="mse", optimizer=self.optimizer(),
                      metrics=['mean_absolute_percentage_error'])
        logging.info("Successfully compiled model.")
        return model

    @staticmethod
    def optimizer():
        opt = Adam(lr=0.01)
        return opt


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    MS = ModelStrucure()
    model = MS.build_model((224, 224, 1), 4)
    model = MS.compile(model)
    print(model.summary())


    from keras.utils import plot_model

    plot_model(model, to_file="model.png", show_shapes=True)
