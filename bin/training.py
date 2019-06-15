from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger, LearningRateScheduler
import logging
import matplotlib.pyplot as plt
from .custom_classes.DataRescaler import DataRescaler
from .custom_classes.DataGenerator import DataGenerator
import numpy as np
from tqdm import tqdm

class ModelTrainer(object):

    def __init__(self, model, training_generator, valid_generator, batch_size,
                 epochs, run_id, outpath, workers, shuffling):
        self.model = model
        self.training_generator = training_generator
        self.valid_generator = valid_generator
        self.batch_size = batch_size
        self.epochs = epochs
        self.run_id = run_id
        self.outpath = outpath
        self.workers = workers
        self.shuffle = shuffling

    def train_model(self):
        filepath = self.outpath + str(self.run_id) + "/weights-improvement-{epoch:02d}.hdf5"
        logging.info(f"Will write checkpoints to: {filepath}")
        checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False, save_weights_only=False, mode='min',
                                     monitor='val_loss',)
        callbacks_list = [checkpoint]
        tb_logdir = f"./tb_log/{self.run_id}/"
        logging.debug(f"Tensorboard logdir at: {tb_logdir}")
        callbacks_list.append(TensorBoard(log_dir=tb_logdir, write_graph=True, write_images=True,
                                     write_grads=True, update_freq="batch", batch_size=self.batch_size))
        # callbacks_list.append(ReduceLROnPlateau(patience=3, factor=0.5))
        callbacks_list.append(LearningRateScheduler(schedule=lambda x: 0.01/(x+1)))
        callbacks_list.append(CSVLogger(filename=self.outpath + str(self.run_id) + "/train_log.csv"))

        self.model.fit_generator(generator=self.training_generator,
                                 validation_data=self.valid_generator,
                                 use_multiprocessing=True, workers=self.workers,
                                 epochs=self.epochs, shuffle=self.shuffle,
                                 callbacks=callbacks_list)