import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from src.rsqt_forest import RSQT

start_time = time.time()

N = 300  # number of points

# random state 1 blob = 0, 2 blobs = 4, 3 blobs = 3, 42
base_coords = make_blobs(n_samples=N, n_features=2, center_box=(0, 300), cluster_std=20, random_state=3)[0]

# base_coords = 60*make_moons(n_samples=N, noise=0.15, random_state=0)[0]


def plot_qt(data):
    ax = plt.subplot()

    colors = np.array(['#b80000', '#377eb8'])
    ax.scatter([p.x for p in data], [p.y for p in data], c=colors[[(p.is_outlier + 1) // 2 for p in data]], s=16)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


clf = RSQT()
points = clf.fit_predict(base_coords)[0]

print("--- running time: %s seconds ---" % (round(time.time() - start_time, 2)))

plot_qt(points)
