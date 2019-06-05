from keras import layers
from keras.models import Model
from keras.optimizers import Adam
import logging
from functools import partial
import keras.backend as K

class ModelStrucure(object):
    def __init__(self):
        self.conv1x1 = partial(layers.Conv2D, kernel_size=1, activation='relu')
        self.conv3x3 = partial(layers.Conv2D, kernel_size=3, padding='same', activation='relu')
        self.conv5x5 = partial(layers.Conv2D, kernel_size=5, padding='same', activation='relu')

    def inception_module(self, in_tensor, c1, c3_1, c3, c5_1, c5, pp):
        conv1 = self.conv1x1(c1)(in_tensor)
    
        conv3_1 = self.conv1x1(c3_1)(in_tensor)
        conv3 = self.conv3x3(c3)(conv3_1)
    
        conv5_1 = self.conv1x1(c5_1)(in_tensor)
        conv5 = self.conv5x5(c5)(conv5_1)
    
        pool_conv = self.conv1x1(pp)(in_tensor)
        pool = layers.MaxPool2D(3, strides=1, padding='same')(pool_conv)
    
        merged = layers.Concatenate(axis=-1)([conv1, conv3, conv5, pool])
        return merged
    
    def aux_clf(self, in_tensor):
        avg_pool = layers.AvgPool2D(5, 3)(in_tensor)
        conv = self.conv1x1(128)(avg_pool)
        flattened = layers.Flatten()(conv)
        dense = layers.Dense(1024, activation='relu')(flattened)
        dropout = layers.Dropout(0.7)(dense)
        out = layers.Dense(1000, activation='softmax')(dropout)
        return out
    
    def build_model(self, in_shape):
        in_layer = layers.Input(in_shape)
        logging.debug(f"in_layer: {K.int_shape(in_layer)}")
        conv1 = layers.Conv2D(64, 7, strides=2, activation='relu', padding='same')(in_layer)
        logging.debug(f"conv1: {K.int_shape(conv1)}")
        pad1 = layers.ZeroPadding2D()(conv1)
        pool1 = layers.MaxPool2D(3, 2)(pad1)
        conv2_1 = self.conv1x1(64)(pool1)
        conv2_2 = self.conv3x3(192)(conv2_1)
        pad2 = layers.ZeroPadding2D()(conv2_2)
        pool2 = layers.MaxPool2D(3, 2)(pad2)
        logging.debug(f"pool2: {K.int_shape(conv1)}")
    
        inception3a = self.inception_module(pool2, 64, 96, 128, 16, 32, 32)
        inception3b = self.inception_module(inception3a, 128, 128, 192, 32, 96, 64)
        pad3 = layers.ZeroPadding2D()(inception3b)
        pool3 = layers.MaxPool2D(3, 2)(pad3)
        logging.debug(f"pool3: {K.int_shape(pool3)}")
    
        inception4a = self.inception_module(pool3, 192, 96, 208, 16, 48, 64)
        inception4b = self.inception_module(inception4a, 160, 112, 224, 24, 64, 64)
        inception4c = self.inception_module(inception4b, 128, 128, 256, 24, 64, 64)
        inception4d = self.inception_module(inception4c, 112, 144, 288, 32, 48, 64)
        inception4e = self.inception_module(inception4d, 256, 160, 320, 32, 128, 128)
        pad4 = layers.ZeroPadding2D()(inception4e)
        pool4 = layers.MaxPool2D(3, 2)(pad4)
        logging.debug(f"pool4: {K.int_shape(pool4)}")
    
        # aux_clf1 = self.aux_clf(inception4a)
        # aux_clf2 = self.aux_clf(inception4d)
    
        inception5a = self.inception_module(pool4, 256, 160, 320, 32, 128, 128)
        inception5b = self.inception_module(inception5a, 384, 192, 384, 48, 128, 128)
        pad5 = layers.ZeroPadding2D()(inception5b)
        pool5 = layers.MaxPool2D(3, 2)(pad5)
    
        avg_pool = layers.GlobalAvgPool2D()(pool5)
        dropout = layers.Dropout(0.4)(avg_pool)
        preds = layers.Dense(1, activation='linear')(dropout)
    
        model = Model(in_layer, preds)
        model = self.compile(model)
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
    MS = ModelStrucure()
    model = MS.build_model()

