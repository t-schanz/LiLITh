from keras import backend as K
from keras.models import Model
from keras.layers import Conv2D, Input, Dense
from keras.optimizers import Adam


class KerasModel(object):

    def __init__(self):
        self.model = None

    def create_model_structure(self, image_shape):

        # -----------------------------------------------------------------------------------------------
        nout = 48
        input = Input(shape=(*image_shape,), name="Input")
        previous = Dense(nout)(input)
        output_layer = Conv2D(filters=1, kernel_size=(3, 3), activation="linear", padding="SAME", name="Output")(previous)
        self.model = Model(inputs=input, outputs=output_layer)


    def compile_model(self, lrate=0.01, decay=0):
        adam = Adam(lr=lrate, beta_1=0.9, beta_2=0.99, epsilon=1e-8, decay=decay)
        loss_function = "mse"
        self.model.compile(loss=loss_function, optimizer=adam)

    def train_model(self, epochs, train_gen, valid_gen, shuffle=True):
        self.model.fit_generator(generator=train_gen,
                                 validation_data=valid_gen,
                                 use_multiprocessing=True, workers=2,
                                 epochs=epochs, shuffle=shuffle,
                                )


if __name__ == "__main__":
    train_generator = # Your training generator here
    valid_generator = #your valid generator here
    model = KerasModel()
    model.create_model_structure((60, 60))
    model.compile_model()
    model.train_model(epochs=5, train_generator, valid_generator)