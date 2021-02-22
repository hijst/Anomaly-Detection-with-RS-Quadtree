import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from src.quadtree import Point, Rect, QuadTree

# Number of points, Sample size, Window size, contamination, random shift
N = 1000
S = int(N / 2)
W = 50
con = 0.10
random_shift = 0


def update_points(hl, new_data):
    hl.set_xdata(np.append(hl.get_xdata(), new_data.x))
    hl.set_ydata(np.append(hl.get_ydata(), new_data.y))
    plt.draw()


def plot_qt(data):
    ax = plt.subplot()

    colors = np.array(['#ff7f00', '#377eb8'])
    ax.scatter([p.x for p in data], [p.y for p in data], c=colors[[(p.is_outlier + 1) // 2 for p in data]], s=16)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


# Generate some random data
base_data = make_blobs(n_samples=N, n_features=2, center_box=(0, 300), cluster_std=20, random_state=3)[0]
stream = [point + random_shift for point in base_data]
ds = [Point(*coord) for coord in stream]

start_time = time.time()

domain = Rect(400, 400, 800, 800)
qt = QuadTree(domain, 1)
added_points = []
counter = 0
for point in ds:
    added_points.append(point)
    if len(added_points) > S:
        qt.delete(added_points[0])
        added_points = added_points[1:]

    qt.insert(point)

    counter += 1

    if counter == W:
        counter = 0
        points = qt.points_in()
        cutoff = int(len(points) * con)
        pnts_sorted = sorted(points, key=lambda x: x.anomaly_score)
        anomalies = pnts_sorted[:cutoff]

        for pnt in anomalies:
            pnt.is_outlier = 0
        plot_qt(points)


