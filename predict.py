from bin.custom_classes.Predictor import Prediction
from bin.custom_classes.DataGenerator import DataGenerator
import xarray as xr
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
from datetime import datetime as dt
import os


def get_image_time(image_file):
    filename = os.path.split(image_file)[-1]
    date = dt.strptime(filename[:13], "m%y%m%d%H%M%S")
    # logging.debug(f"Got time {date.strftime('%X %x')} from image {filename}")
    return date


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

    Pred = Prediction("./checkpoints/9/weights-improvement-05.hdf5")
    X, y, X_raw, y_raw = Pred.predict([image_array, dship_array], lidar_array)
    Pred.save(X, y, image_time_array, "./results/CBH_20190501.nc")
