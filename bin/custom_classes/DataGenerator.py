from keras.utils import Sequence
import xarray as xr
import dask.array as da
import os
from datetime import datetime as dt
import numpy as np
from PIL import Image
import logging


class DataGenerator(Sequence):

    def __init__(self, image_files, lidar_files, batch_size, image_dim=None, shuffle=True):
        self.batch_size = batch_size
        self.image_files = image_files
        self.image_dim = image_dim or self.__get_image_dim()
        self.shuffle = shuffle
        self.indices = np.arange(len(image_files))
        self.lidar = self.__get_lidar_ds(lidar_files)

        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            logging.debug("Shuffling indices.")
            np.random.shuffle(self.indices)
        else:
            logging.debug("NO SHUFFLING")


    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        logging.debug(f"batch_indices: {batch_indices}")
        X = np.empty((self.batch_size, *self.image_dim))
        y = np.empty((self.batch_size))

        for i, ID in enumerate(batch_indices):
            X[i, ] = self.__get_image(ID)
            y[i, ] = self.__get_cbh(ID)

        X_norm = self.__data_normalization(X)
        y_norm = self.__data_normalization(y, min=-1, max=10000)
        return X_norm, y_norm

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    @staticmethod
    def __get_lidar_ds(lidar_files):
        ds = xr.open_mfdataset(sorted(lidar_files))
        cbh = ds.cbh.sel(layer=1)
        return cbh

    @staticmethod
    def __get_image_time(image_file):
        filename = os.path.split(image_file)[-1]
        date = dt.strptime(filename[:13], "m%y%m%d%H%M%S")
        logging.debug(f"Got time {date.strftime('%X %x')} from image {filename}")
        return date

    def __match_lidar_to_image(self, image_date):
        matched_lidar_ds = self.lidar.sel(time=image_date, method="nearest")
        matched_time = matched_lidar_ds.time.values
        logging.debug(f"Matching image_date {image_date.strftime('%x %X')} to lidar date {str(matched_time)}")
        return matched_lidar_ds.data

    def __get_cbh(self, ID):
        image_file = self.image_files[ID]
        image_date = self.__get_image_time(image_file)
        logging.debug(f"Got image date: {image_date.strftime('%x %X')}")
        cbh = self.__match_lidar_to_image(image_date)
        return cbh

    def __get_image(self, index):
        im = Image.open(self.image_files[index])
        array = np.array(im)[:, :, :]
        width = array.shape[0]
        height = array.shape[1]
        resized_array = array[int(width/2)-112:int(width/2)+112, int(height/2)-112:int(height/2)+112]
        return resized_array

    def __get_image_dim(self):
        image = self.__get_image(0)
        logging.info(f"Automatically set image dim to: {image.shape}")
        return image.shape

    @staticmethod
    def __data_normalization(array, min=0, max=255) -> np.ndarray:
        """
        Normalizes the values of array to be between -1 and 1.

        Args:
            array: array to be normalized
            min: absolute minimum of all appearing values
            max: absolute maximum of all appearing values

        Returns:
            np.array with normalized values
        """

        return (array - (np.mean((min, max)))) / (max - min) * 1.999

if __name__ == "__main__":
    import glob
    logging.basicConfig(level=logging.DEBUG)
    image_files = sorted(glob.glob("D:/2019_Sonne/THERMAL/mx10-18-202-137/2019/extracted/05/01/*"))
    lidar_files = sorted(glob.glob("D:/2019_Sonne/ceilometer/20190501_RV Sonne_CHM188105_000.nc"))

    DG = DataGenerator(image_files=image_files, lidar_files=lidar_files, batch_size=32)

    for i in range(len(DG))[:1]:
        print(i)
        foo = DG[i]