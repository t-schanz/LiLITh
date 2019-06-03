from keras.callbacks import ModelCheckpoint
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
        filepath = self.outpath + "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
        callbacks_list = [checkpoint]
        self.model.fit_generator(generator=self.training_generator,
                                 validation_data=self.valid_generator,
                                 use_multiprocessing=True, workers=self.workers,
                                 epochs=self.epochs, shuffle=self.shuffle,
                                 callbacks=callbacks_list)