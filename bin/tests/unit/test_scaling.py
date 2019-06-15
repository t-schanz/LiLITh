import unittest
from bin.custom_classes.DataRescaler import DataRescaler
from bin.custom_classes.DataGenerator import DataGenerator
import glob
import numpy as np

class TestScaling(unittest.TestCase):

    def test_lidar_range(self):
        image_files = sorted(glob.glob("D:/2019_Sonne/THERMAL/mx10-18-202-137/2019/extracted/05/01/*"))[:100]
        lidar_files = sorted(glob.glob("D:/2019_Sonne/ceilometer/20190501_RV Sonne_CHM188105_000.nc"))
        dship_file = "D:/2019_Sonne/DSHIP/DSHIP_WEATHER_5MIN-RES_20181020-20190610/DSHIP_WEATHER_5MIN-RES_20181020-20190610.csv"
        self.DG = DataGenerator(image_files=image_files, lidar_files=lidar_files, dship_path=dship_file,
                                batch_size=1000, shuffle=False)

        less = np.all(np.less_equal(self.DG.lidar.isel(time=slice(0, 1000)).values, 10000))
        greater = np.all(np.greater_equal(self.DG.lidar.isel(time=slice(0, 1000)).values, -1))
        self.assertTrue(less)
        self.assertTrue(greater)

    def test_scaling(self):
        image_files = sorted(glob.glob("D:/2019_Sonne/THERMAL/mx10-18-202-137/2019/extracted/05/01/*"))[:1000]
        lidar_files = sorted(glob.glob("D:/2019_Sonne/ceilometer/20190501_RV Sonne_CHM188105_000.nc"))
        dship_file = "D:/2019_Sonne/DSHIP/DSHIP_WEATHER_5MIN-RES_20181020-20190610/DSHIP_WEATHER_5MIN-RES_20181020-20190610.csv"
        self.DG = DataGenerator(image_files=image_files, lidar_files=lidar_files, dship_path=dship_file,
                                batch_size=1000, shuffle=False)


        batch = self.DG[0]
        image_batch = batch[0][0]
        dship_batch = batch[0][1]
        lidar_batch = batch[1]

        self.assertLessEqual(image_batch.max(), 1)
        self.assertLessEqual(dship_batch.max(), 1)
        self.assertLessEqual(lidar_batch.max(), 1)

        self.assertGreaterEqual(image_batch.min(), -1)
        self.assertGreaterEqual(dship_batch.min(), -1)
        self.assertGreaterEqual(lidar_batch.min(), -1)

        lidar_values = self.DG.lidar.isel(time=slice(0, 1000)).values
        lidar_batch_rescaled = np.round(DataRescaler().rescale(lidar_batch, vmin=-1, vmax=10000), 0).astype(int)

        rescaled_in_lidar = np.isin(lidar_batch_rescaled, lidar_values)

        lidar_test = np.all(rescaled_in_lidar)

        self.assertTrue(lidar_test)


if __name__ == '__main__':
    unittest.main()