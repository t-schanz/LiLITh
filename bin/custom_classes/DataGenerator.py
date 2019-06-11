from keras.utils import Sequence
import xarray as xr
import dask.array as da
import os
from datetime import datetime as dt
import numpy as np
from PIL import Image
import logging


class DataGenerator(Sequence):

    def __init__(self, image_files, lidar_files, dship_path, batch_size, image_dim=None, shuffle=True):
        self.batch_size = batch_size
        self.image_files = image_files
        self.image_dim = image_dim or self.__get_image_dim()
        self.shuffle = shuffle
        self.indices = np.arange(len(image_files))
        self.lidar = self.__get_lidar_ds(lidar_files)
        self.dship = self.__get_regression_data(dship_path)
        self.dship_norm = self.__norm_dship_data()

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
        dship = np.empty((self.batch_size, 5))

        for i, ID in enumerate(batch_indices):
            image_file = self.image_files[ID]
            image_date = self.__get_image_time(image_file)
            logging.debug(f"Got image date: {image_date.strftime('%x %X')}")
            X[i, ] = self.__get_image(ID)
            y[i, ] = self.__get_cbh(image_date)
            dship[i, ] = self.__match_dship_to_image(image_date)

        logging.debug(f"X max: {X.max()}")
        X_norm = self.__data_normalization(X)
        y_norm = self.__data_normalization(y, vmin=-1, vmax=10000)
        return [X_norm, dship], y_norm

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    @staticmethod
    def __get_regression_data(file_path):
        with open(file_path, "r") as f:
            dataset = np.genfromtxt(f, delimiter=";", skip_header=3, usecols=[6, 7, 8, 9, 10])

        with open(file_path, "r") as f:
            dataset_dates = np.genfromtxt(f, delimiter=";", skip_header=3, dtype=None, usecols=[0])

        generate_dates = np.vectorize(lambda x: dt.strptime(x.decode(), "%Y%m%dT%H%M%S"))
        dataset_dates = generate_dates(dataset_dates)

        ds_lat = xr.DataArray(dataset[:, 0], dims=["time"], coords={"time": dataset_dates})
        ds_lon = xr.DataArray(dataset[:, 1], dims=["time"], coords={"time": dataset_dates})
        ds_temp = xr.DataArray(dataset[:, 2], dims=["time"], coords={"time": dataset_dates})
        ds_pressure = xr.DataArray(dataset[:, 3], dims=["time"], coords={"time": dataset_dates})
        ds_humid = xr.DataArray(dataset[:, 4], dims=["time"], coords={"time": dataset_dates})

        ds = xr.Dataset({"lat": ds_lat, "lon": ds_lon, "temperature": ds_temp, "pressure": ds_pressure,
                         "humidity": ds_humid})
        return ds

    def __norm_dship_data(self):
        temp = self.__data_normalization(self.dship["temperature"], vmin=-20, vmax=40)
        lat = self.__data_normalization(self.dship["lat"], vmin=-90, vmax=90)
        lon = self.__data_normalization(self.dship["lon"], vmin=-180, vmax=180)
        pressure = self.__data_normalization(self.dship["pressure"], vmin=970, vmax=1100)
        humidity = self.__data_normalization(self.dship["humidity"], vmin=0, vmax=100)

        ds = xr.Dataset({"lat": lat, "lon": lon, "temperature": temp, "pressure": pressure,
                         "humidity": humidity})

        ds = ds.fillna(-1)

        return ds

    @staticmethod
    def __get_lidar_ds(lidar_files):
        ds = xr.open_mfdataset(sorted(lidar_files))
        cbh = ds.cbh.sel(layer=1)
        cbh = cbh.fillna(0)
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

    def __match_dship_to_image(self, image_date):
        matched_dship_ds = self.dship_norm.sel(time=image_date, method="nearest")
        matched_time = matched_dship_ds.time.values
        logging.debug(f"Matching image_date {image_date.strftime('%x %X')} to dship date {str(matched_time)}")

        lat = matched_dship_ds.lat.data
        lon = matched_dship_ds.lon.data
        temp = matched_dship_ds.temperature.data
        pres = matched_dship_ds.pressure.data
        humid = matched_dship_ds.humidity.data

        return lat, lon, temp, pres, humid

    def __get_cbh(self, image_date):

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
    def __data_normalization(array, vmin=0, vmax=255) -> np.ndarray:
        """
        Normalizes the values of array to be between -1 and 1.

        Args:
            array: array to be normalized
            vmin: absolute minimum of all appearing values
            vmax: absolute maximum of all appearing values

        Returns:
            np.array with normalized values
        """
        return (array - (np.mean((vmin, vmax)))) / (vmax - vmin) * 1.999

if __name__ == "__main__":
    import glob
    logging.basicConfig(level=logging.DEBUG)
    image_files = sorted(glob.glob("D:/2019_Sonne/THERMAL/mx10-18-202-137/2019/extracted/05/01/*"))
    lidar_files = sorted(glob.glob("D:/2019_Sonne/ceilometer/20190501_RV Sonne_CHM188105_000.nc"))
    dship_file = "D:/2019_Sonne/DSHIP/DSHIP_WEATHER_5MIN-RES_20181020-20190610/DSHIP_WEATHER_5MIN-RES_20181020-20190610.csv"
    DG = DataGenerator(image_files=image_files, lidar_files=lidar_files, dship_path=dship_file, batch_size=32)

    for i in range(len(DG))[:1]:
        print(i)
        foo = DG[i]