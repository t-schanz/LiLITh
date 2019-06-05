import numpy as np

class DataRescaler(object):

    def __init__(self):
        pass


    def rescale(self, normalized_array:np.ndarray, min=-1, max=10000)->np.ndarray:
        """
        This method rescales a normed array to its original values.

        Args:
            normalized_array: input numpy.ndarray
            min: minimum used for normalization
            max: maximum used for normalization

        Returns:
            numpy.ndarray with rescaled values.
        """
        part1 = np.multiply(normalized_array, np.subtract(max, min))
        array = np.add(np.divide(part1, 1.999),np.mean((min,max)))
        # array = np.add(np.divide(np.multiply(normalized_array, (np.subtract(max,min))),1.999), np.mean(min,max))
        array[np.where(array <= 0)] = np.nan
        return array