import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import utils

"""
Only use with normalised datasets
"""

OUT_PATH = "out/"

def save_distribution(df: pd.DataFrame, out_file: str) -> None:
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    class_names = list(df.columns)[1:]
    class_counts = []
    for name in class_names:
        class_counts.append((df[name] == 1).sum())
        
    ax.bar(class_names, class_counts)
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.title('Category distribution')

    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        percentage = float(height/len(df))
        x, y = p.get_xy() 
        ax.annotate(f'{percentage:.2%}', (x + width/2, y + height*1.01), ha='center')
        
    plt.savefig(OUT_PATH + out_file, dpi=300, bbox_inches='tight')


def save_training_curves(training, validation, title: str, epochs: int) -> None:

    fig = go.Figure()
        
    fig.add_trace(
        go.Scatter(x=np.arange(1, epochs+1), mode='lines+markers', y=training, marker=dict(color="dodgerblue"),
               name="Train"))
    
    fig.add_trace(
        go.Scatter(x=np.arange(1, epochs+1), mode='lines+markers', y=validation, marker=dict(color="darkorange"),
               name="Val"))
    
    fig.update_layout(title_text=title, yaxis_title="Accuracy", xaxis_title="Epochs", template="plotly_white")

    fig.write_image(OUT_PATH + str(title) + ".jpeg")


def save_sample_leaves(dataset, images, out_file, cond=[0, 0, 0, 0, 0, 0]):
        
    conds = ("frog_eye_leaf_spot == {}".format(cond[0]), 
        "rust == {}".format(cond[1]), 
        "healthy == {}".format(cond[2]),
        "scab == {}".format(cond[3]),
        "powdery_mildew == {}".format(cond[4]),
        "complex == {}".format(cond[5]))
    
    cond_list = []
    for i, col in enumerate(cond):
        if col == 1:
            cond_list.append(conds[i])
    
    data = dataset.loc[:100]
    for cond in cond_list:
        data = data.query(cond)
        
    cols, rows = 6, min([3, len(images)//3])
    
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(30, rows*20/3))
    for col in range(cols):
        for row in range(rows):
            ax[row, col].imshow(images.loc[images.index[row*3+col]])

    plt.savefig(OUT_PATH + out_file, dpi=300, bbox_inches='tight')

def save_samples(labels_df, dataset, out_file):
        
    conds = ("frog_eye_leaf_spot == 1", 
        "rust == 1", 
        "healthy == 1",
        "scab == 1",
        "powdery_mildew == 1",
        "complex == 1")
    
    
    data = dataset.loc[:100]
    images = []
    for cond in conds:
        samples = data.query(cond)
        for image_name in samples["image"][:3]:
            labels = list(labels_df.loc[labels_df["image"] == image_name]["labels"])
            images.append((labels, utils.read_image(utils.TRAIN_IMAGES_FOLDER + image_name)))

    cols, rows = 6, 3
    cols_names = ['Frog eye leaf spot', 'Rust', 'Healthy', 'Scab', 'Powdery mildew', 'Complex']

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(30, 10))

    for ax, col in zip(axes[0], cols_names):
        ax.set_title(col)

    for col in range(cols):
        for row in range(rows):
            image = images[row+col*3][1]
            height, width, _ = image.shape
            axes[row, col].imshow(image)
            axes[row, col].annotate(images[row+col*3][0][0], (width*0.5, height*1.15), annotation_clip=False, ha='center')

    plt.savefig(OUT_PATH + out_file, dpi=300, bbox_inches='tight')


def save_test_results(out_file, correct=[1517, 1506, 11, 0], incorrect=[346, 198, 125, 23]):
    labels = ['Total', '1 label samples', '2 label samples', '3 label samples']

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, correct, width, label='Correctly predicted')
    ax.bar(x + width/2, incorrect, width, label='Incorrectly predicted')


    ax.set_ylabel('Number of samples')
    ax.set_title('Performance by number of labels')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    
    for i in range(len(x)):
        plt.text(i - width, correct[i] + width*1.5, correct[i])
        plt.text(i + width*0.25, incorrect[i] + width*1.5, incorrect[i])

    plt.savefig(OUT_PATH + out_file, dpi=300, bbox_inches='tight')