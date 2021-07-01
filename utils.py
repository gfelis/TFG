# Standard libraries
from argparse import ArgumentParser
import random

# Third party libraries
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

IMG_SHAPE = (4000, 2672)
TRAIN_IMAGES_FOLDER = "./data/train_images/"
TEST_IMAGES_FOLDER = "./data/test_images/"
TRAIN_CSV = "data/train.csv"


def seed_reproducer(seed=2021):
    np.random.seed(seed)
    random.seed(seed)

def load_split_dataset(frac: float=0.1) -> "tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]":
    data = pd.read_csv(TRAIN_CSV)
    seed_reproducer()
    state = random.randint(0, 10000)
    test = data.sample(frac=frac, random_state=state).reset_index()
    train = data
    for index in test['index'].values:
        train = train.drop([index])
    train = train.reset_index(drop=True)
    test = test.drop(columns=['index'])
    return data, train, test

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
    
def load_model(file_name: str)-> tf.keras.Sequential:
    return tf.keras.models.load_model("models/" + file_name) 

def read_log(file_name):
    return pd.read_csv('models/' + file_name, sep=',', engine='python')
