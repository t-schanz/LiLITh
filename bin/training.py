from keras.callbacks import ModelCheckpoint, TensorBoard
import logging

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
        filepath = self.outpath + str(self.run_id) + "/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        logging.info(f"Will write checkpoints to: {filepath}")
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
        callbacks_list = [checkpoint]
        tb_logdir = f"./tb_log/{self.run_id}/"
        logging.debug(f"Tensorboard logdir at: {tb_logdir}")
        callbacks_list.append(TensorBoard(log_dir=tb_logdir, histogram_freq=10, write_graph=True, write_images=False,
                                     write_grads=True, update_freq="batch", batch_size=self.batch_size))

        self.model.fit_generator(generator=self.training_generator,
                                 validation_data=self.valid_generator,
                                 use_multiprocessing=True, workers=self.workers,
                                 epochs=self.epochs, shuffle=self.shuffle,
                                 callbacks=callbacks_list)