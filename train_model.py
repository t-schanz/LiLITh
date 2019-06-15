from bin.custom_classes.KerasModel2 import ModelStrucure
from bin.custom_classes.DataGenerator import DataGenerator
from bin.training import ModelTrainer
import argparse
import glob
import logging
from logging.handlers import RotatingFileHandler

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--run_id', metavar=1, type=int, help="Set the id of the run.",
                        required=True)

    parser.add_argument('-o', '--outpath', metavar="./checkpoints/", help="Set the path where to save"
                                                                          "the trained model structure and weights.",
                        required=False, default="./checkpoints/")

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

    parser.add_argument('-v', '--verbose', metavar="DEBUG",
                        help='Set the level of verbosity [DEBUG, INFO, WARNING, ERROR]',
                        required=False, default="INFO")

    args = vars(parser.parse_args())
    return args


def setup_logging(verbose):
    assert verbose in ["DEBUG", "INFO", "WARNING", "ERROR"]
    logging.basicConfig(
        level=logging.getLevelName(verbose),
        format="%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s",
        handlers=[
            RotatingFileHandler(f"./logs/{__file__}.log", maxBytes=int(1e6), backupCount=5),
            logging.StreamHandler()
        ])


if __name__ == "__main__":
    args = get_args()
    setup_logging("INFO")

    image_files = sorted(glob.glob("D:/2019_Sonne/THERMAL/mx10-18-202-137/2019/extracted/05/*/*"))
    lidar_files = sorted(glob.glob("D:/2019_Sonne/ceilometer/201905*.nc"))
    dship_file = "D:/2019_Sonne/DSHIP/DSHIP_WEATHER_5MIN-RES_20181020-20190610/DSHIP_WEATHER_5MIN-RES_20181020-20190610.csv"

    batch_size = args["batches"]

    train_gen = DataGenerator("TrainGen", image_files=image_files[:-1000], lidar_files=lidar_files, dship_path=dship_file,
                              batch_size=batch_size)
    valid_gen = DataGenerator("ValidGen", image_files=image_files[-1000:], lidar_files=lidar_files, dship_path=dship_file,
                              batch_size=1000, shuffle=False)

    valid_data = valid_gen[0]


    gen0 = train_gen[0]
    logging.debug(f"Image shape: {gen0[0][0][0].shape}")
    logging.debug(f"DSHIP shape: {gen0[0][1][0].shape}")


    Ms = ModelStrucure()
    model = Ms.build_model(CNN_shape=gen0[0][0][0].shape, MLP_shape=gen0[0][1][0].shape[0])
    model = Ms.compile(model)


    Trainer = ModelTrainer(model=model, training_generator=train_gen, valid_generator=valid_data,
                           batch_size=batch_size, epochs=args["epochs"], run_id=args["run_id"],
                           outpath=args["outpath"], workers=args["workers"], shuffling=True)

    Trainer.train_model()
