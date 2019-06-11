from keras.models import load_model
from .DataRescaler import DataRescaler
import numpy as np
import xarray as xr


class Prediction(object):

    def __init__(self, weight_file):
        self.model = load_model(weight_file)

    def predict(self, X, y):
        # X_norm = self.data_normalization(X[0], min=0, max=255)
        predictions = self.model.predict(X)[:,0]
        preds_rescaled = self.prediction2cbh(predictions)
        y_rescaled = self.prediction2cbh(y)
        return preds_rescaled, y_rescaled

    @staticmethod
    def prediction2cbh(prediction):
        rescaled = DataRescaler().rescale(prediction)
        return rescaled

    @staticmethod
    def save(X, y, time_array, save_file):
        da_X = xr.DataArray(X, dims="time", coords={"time": time_array})
        da_y = xr.DataArray(y, dims="time", coords={"time": time_array})
        ds = xr.Dataset({"X": da_X, "y": da_y})
        ds["X"]["long_name"] = "Predicted Cloud Base Height"
        ds["y"]["long_name"] = "Ceilometer Cloud Base Height"
        ds["X"]["unit"] = "m"
        ds["y"]["unit"] = "m"
        ds.to_netcdf(save_file)

    @staticmethod
    def data_normalization(array, min=-1, max=10000) -> np.ndarray:
        """
        Normalizes the values of array to be between -1 and 1.

        Args:
            array: array to be normalized
            min: absolute minimum of all appearing values
            max: absolute maximum of all appearing values

        Returns:
            np.array with normalized values
        """

        array[np.where(np.isnan(array))] = -32.5
        array[np.where(array <= -32.5)] = min

        return (array - (np.mean((min, max))))/(max - min) * 1.999
