# Standard libraries
from argparse import ArgumentParser

# Third party libraries
import cv2
import numpy as np
from numpy.lib.function_base import append
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

def load_split_dataset(frac: float=0.05) -> pd.DataFrame:

    data = pd.read_csv("data/data.csv")
    test = data.sample(frac=frac).reset_index()
    train = data
    for index in test['index'].values:
        train = train.drop([index])
    train = train.reset_index(drop=True)
    test = test.drop(columns=['index'])
    test.to_csv("data/test.csv", index=False)
    train.to_csv("data/train.csv", index=False)
    print(len(data), len(train), len(test))
    return train, test

def read_image(image_path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)


def normalise_from_dataset_disjoint(dataset: pd.DataFrame) -> pd.DataFrame:
    columns = ['image']
    labels = dataset['labels'].value_counts().index.tolist()
        
    columns.extend(labels)
    data = []

    for image, label in zip(dataset['image'], dataset['labels']):
        labelpos = columns.index(label)
        row = [image]
        for _ in labels: row.append(0)
        row[labelpos] =  1
        data.append(row)
    
        return pd.DataFrame(data, columns=columns)
    

def normalise_from_dataset_joint(dataset: pd.DataFrame) -> pd.DataFrame:
    columns = ['image']
    labels = dataset['labels'].value_counts().index.tolist()
    basic_labels = set()   
    for label in labels:
         for word in label.split():
             basic_labels.add(word)

    columns.extend(basic_labels)
    data = []

    for image, labels in zip(dataset['image'], dataset['labels']):

        row = [image]
        real_labels = labels.split()
        for _ in basic_labels: row.append(0)
        for real_label in real_labels:
            labelpos = columns.index(real_label)
        row[labelpos] =  1
        data.append(row)
    
        return pd.DataFrame(data, columns=columns)
    