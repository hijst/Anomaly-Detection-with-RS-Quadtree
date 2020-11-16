import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from RSQT_forest import RSQT

start_time = time.time()

# base parameters
DPI = 72  # used for plotting the output
width, height = 600, 600  # height and width of the output
N = 300  # number of points

# random state 1 blob = 0, 2 blobs = 4, 3 blobs = 3, 42
base_coords = make_blobs(n_samples=N, n_features=2, center_box=(0, 300), cluster_std=20, random_state=3)[0]


# base_coords = 60*make_moons(n_samples=N, noise=0.15, random_state=0)[0]


def plot_qt(data):
    fig = plt.figure(figsize=(700 / DPI, 500 / DPI), dpi=DPI)
    ax = plt.subplot()
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    # qtree.draw(ax)

    colors = np.array(['#b80000', '#377eb8'])
    ax.scatter([p.x for p in data], [p.y for p in data], c=colors[[(p.payload + 1) // 2 for p in data]], s=16)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


clf = RSQT()
points = clf.fit_predict(base_coords)[0]

print("--- running time: %s seconds ---" % (round(time.time() - start_time, 2)))

plot_qt(points)
