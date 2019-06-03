from .bin.custom_classes.KerasModel import ModelStrucure
from .bin.custom_classes.DataGenerator import DataGenerator
import argparse
import glob

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--run_id', metavar=1, type=int, help="Set the id of the run.",
                        required=True)

    parser.add_argument('-o', '--outpath', metavar="./checkpoints/", help="Set the path where to save"
                                                                          "the trained model structure and weights.",
                        required=False, default=None)

    parser.add_argument('-e', '--epochs', metavar=100, type=int, help="Set on how many epochs to train.",
                        required=True, default=None)

    parser.add_argument('--gpu', action="store_true", help="If this flag is provided use GPU",
                        required=False, default=False)

    parser.add_argument('--cores', metavar=1, type=int, help="Set how many cpu cores to use (if --gpu is set, then this"
                                                             "will determine the number of gpu cores)",
                        required=False, default=1)

    parser.add_argument('--workers',  metavar=1, type=int, help="Set the number of worker for the DataGenerator.",
                        required=False, default=1)

    parser.add_argument('--batches',  metavar=16, type=int, help="Set the batch size.",
                        required=False, default=16)

    args = vars(parser.parse_args())
    return args

if __name__ == "__main__":
    args = get_args()

    image_files = sorted(glob.glob("D:/2019_Sonne/THERMAL/mx10-18-202-137/2019/extracted/05/01/*"))
    lidar_files = sorted(glob.glob("D:/2019_Sonne/ceilometer/20190501_RV Sonne_CHM188105_000.nc"))

    batch_size = args["batches"]

    train_gen = DataGenerator(image_files=image_files[:-1000], lidar_files=lidar_files, batch_size=batch_size)
    valid_gen = DataGenerator(image_files=image_files[-1000:], lidar_files=lidar_files, batch_size=batch_size)

