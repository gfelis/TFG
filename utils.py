# Standard libraries
from argparse import ArgumentParser

# Third party libraries
import cv2
import numpy as np
import pandas as pd

IMG_SHAPE = (4000, 2672)
TRAIN_IMAGES_FOLDER = "data/train_images/"
TEST_IMAGES_FOLDER = "data/test_images/"


def init_hparams():
    parser = ArgumentParser(add_help=False)
    parser.add_argument("-backbone", "--backbone", type=str, default="se_resnext50_32x4d")
    parser.add_argument("-tbs", "--train_batch_size", type=int, default=32 * 1)
    parser.add_argument("-vbs", "--val_batch_size", type=int, default=16 * 1)
    parser.add_argument("--image_size", nargs="+", default=[512, 512])
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--gpus", nargs="+", default=[0])
   
    try:
        hparams = parser.parse_args()
    except:
        hparams = parser.parse_args([])
    print(type(hparams.gpus), hparams.gpus)
    if len(hparams.gpus) == 1:
        hparams.gpus = [int(hparams.gpus[0])]
    else:
        hparams.gpus = [int(gpu) for gpu in hparams.gpus]

    hparams.image_size = [int(size) for size in hparams.image_size]
    return hparams

def load_data(test_size):
    # TO BE DONE: Need to split data into test image folder and test.csv
    data = pd.read_csv("data/train.csv")
    
    return data

def read_image(image_path):
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

def load_model(model_path):
    #TO BE DONE
    return None