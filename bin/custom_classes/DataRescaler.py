import numpy as np

class DataRescaler(object):

    def __init__(self):
        pass


    def rescale(self, normalized_array:np.ndarray, vmin=-1, vmax=10000)->np.ndarray:
        """
        This method rescales a normed array to its original values.

        Args:
            normalized_array: input numpy.ndarray
            vmin: minimum used for normalization
            vmax: maximum used for normalization

        Returns:
            numpy.ndarray with rescaled values.
        """
        part1 = np.multiply(normalized_array, np.subtract(vmax, vmin))
        array = np.add(np.divide(part1, 1.999), np.mean((vmin, vmax)))
        # array = np.add(np.divide(np.multiply(normalized_array, (np.subtract(vmax,vmin))),1.999), np.mean(vmin,vmax))
        # array[np.where(array <= 0)] = np.nan
        return array