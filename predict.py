from bin.custom_classes.Predictor import Prediction
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


if __name__ == "__main__":
    lidar_files = sorted(glob.glob("D:/2019_Sonne/ceilometer/20190501_RV Sonne_CHM188105_000.nc"))
    image_files = sorted(glob.glob("D:/2019_Sonne/THERMAL/mx10-18-202-137/2019/extracted/05/01/*"))
    ds = xr.open_mfdataset(lidar_files)

    image_array = images2array(image_files)
    image_time_array = []

    for i, file in enumerate(image_files):
        image_time_array.append(get_image_time(file))

    image_time_array = np.asarray(image_time_array)

    lidar_array = match_lidar_to_image(image_time_array, ds.cbh)

    Pred = Prediction("./checkpoints/1/weights-improvement-08.hdf5")
    X, y = Pred.predict(image_array, lidar_array)
    Pred.save(X, y, image_time_array, "./results/CBH_20190501.nc")
