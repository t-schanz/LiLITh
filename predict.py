from bin.custom_classes.Predictor import Prediction
from bin.custom_classes.DataGenerator import DataGenerator
import xarray as xr
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
from datetime import datetime as dt
import os


def images2array(files):
    image_array = np.empty((len(files), 224, 224, 3))
    for i, file in enumerate(tqdm(files)):
        im = Image.open(file)
        array = np.array(im)[:, :, :]
        width = array.shape[0]
        height = array.shape[1]
        image_array[i] = array[int(width / 2) - 112:int(width / 2) + 112, int(height / 2) - 112:int(height / 2) + 112, :]
    return image_array


def get_image_time(image_file):
    filename = os.path.split(image_file)[-1]
    date = dt.strptime(filename[:13], "m%y%m%d%H%M%S")
    # logging.debug(f"Got time {date.strftime('%X %x')} from image {filename}")
    return date


def match_lidar_to_image(image_date, lidar_da):
    matched_lidar_ds = lidar_da.sel(dict(time=image_date, layer=1), method="nearest")
    matched_time = matched_lidar_ds.time.values
    # logging.debug(f"Matching image_date {image_date.strftime('%x %X')} to lidar date {str(matched_time)}")
    return matched_lidar_ds.values


def norm_dship_data(dship):
    temp = data_normalization(dship["temperature"], vmin=-20, vmax=40)
    lat = data_normalization(dship["lat"], vmin=-90, vmax=90)
    lon = data_normalization(dship["lon"], vmin=-180, vmax=180)
    pressure = data_normalization(dship["pressure"], vmin=970, vmax=1100)
    humidity = data_normalization(dship["humidity"], vmin=0, vmax=100)

    ds = xr.Dataset({"lat": lat, "lon": lon, "temperature": temp, "pressure": pressure,
                     "humidity": humidity})

    return ds


def __get_regression_data(file_path):
    with open(file_path, "r") as f:
        dataset = np.genfromtxt(f,
                                delimiter=";",
                                skip_header=3,
                                # dtype=None,
                                usecols=[6, 7, 8, 9, 10]
                                )

    with open(file_path, "r") as f:
        dataset_dates = np.genfromtxt(f,
                                      delimiter=";",
                                      skip_header=3,
                                      dtype=None,
                                      usecols=[0]
                                      )

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


def data_normalization(array, vmin=0, vmax=255) -> np.ndarray:
    return (array - (np.mean((vmin, vmax)))) / (vmax - vmin) * 1.999


def __match_dship_to_image(self, image_date):
    matched_dship_ds = self.dship_norm.sel(time=image_date, method="nearest")
    matched_time = matched_dship_ds.time.values

    lat = matched_dship_ds.lat.data
    lon = matched_dship_ds.lon.data
    temp = matched_dship_ds.temperature.data
    pres = matched_dship_ds.pressure.data
    humid = matched_dship_ds.humidity.data

    return lat, lon, temp, pres, humid

if __name__ == "__main__":
    lidar_files = sorted(glob.glob("D:/2019_Sonne/ceilometer/20190501_RV Sonne_CHM188105_000.nc"))
    image_files = sorted(glob.glob("D:/2019_Sonne/THERMAL/mx10-18-202-137/2019/extracted/05/01/*"))[:1000]
    dship_file = "D:/2019_Sonne/DSHIP/DSHIP_WEATHER_5MIN-RES_20181020-20190610/DSHIP_WEATHER_5MIN-RES_20181020-20190610.csv"

    image_time_array = []
    for i, file in enumerate(image_files):
        image_time_array.append(get_image_time(file))

    image_time_array = np.asarray(image_time_array)

    pred_gen = DataGenerator(image_files=image_files, lidar_files=lidar_files, dship_path=dship_file,
                              batch_size=len(image_files))

    image_array = pred_gen[0][0][0]
    dship_array = pred_gen[0][0][1]
    lidar_array = pred_gen[0][1]

    Pred = Prediction("./checkpoints/5/weights-improvement-11.hdf5")
    X, y = Pred.predict([image_array, dship_array], lidar_array)
    Pred.save(X, y, image_time_array, "./results/CBH_20190501.nc")
