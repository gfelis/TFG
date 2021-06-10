import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

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

    # Add percentage annotations
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        percentage = float(height/len(df))
        x, y = p.get_xy() 
        ax.annotate(f'{percentage:.2%}', (x + width/2, y + height*1.01), ha='center')
        
    plt.savefig(OUT_PATH + out_file, dpi=300, bbox_inches='tight')


#Need to update validation and training types annotations
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