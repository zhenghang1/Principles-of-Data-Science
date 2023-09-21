import numpy as np
import matplotlib.pyplot as plt

def draw_fig(X,y,name):
    fig, ax = plt.subplots()
    path = 'figure/'+name+'.png'

    for label in np.unique(y):
        ax.scatter(X[y==label, 0], X[y==label, 1], label=label, s=0.2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.savefig(path)