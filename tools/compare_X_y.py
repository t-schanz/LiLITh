import xarray as xr
import matplotlib.pyplot as plt

if __name__ == "__main__":
    file = "../results/CBH_20190501.nc"

    ds = xr.open_dataset(file)

    fig, ax = plt.subplots()
    ax.plot(ds.X.values-ds.y.values)
    plt.show()