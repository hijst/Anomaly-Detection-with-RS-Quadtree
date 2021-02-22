import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from src.quadtree import Point, Rect, QuadTree

# Number of points, Sample size, Window size, contamination, random shift
N = 1000
S = int(N / 2)
W = 100
con = 0.05
random_shift = 0
colors = np.array(['#ff7f00', '#377eb8'])
rng = np.random.RandomState(1)

# Generate some random data
base_data = np.concatenate((make_blobs(n_samples=300, n_features=2, center_box=(0, 100), cluster_std=20, random_state=1)[0],
                            rng.uniform(low=0, high=350, size=(35, 2)),
                            make_blobs(n_samples=300, n_features=2, center_box=(100, 200), cluster_std=20, random_state=11)[0],
                            rng.uniform(low=0, high=350, size=(35, 2)),
                            make_blobs(n_samples=300, n_features=2, center_box=(200, 300), cluster_std=20, random_state=21)[0],
                            rng.uniform(low=0, high=350, size=(35, 2))))
stream = [point + random_shift for point in base_data]
ds = [Point(*coord) for coord in stream]

start_time = time.time()

domain = Rect(400, 400, 800, 800)
qt = QuadTree(domain, 1)
i = 0
cv = 0
fig_nr = 0
for point in ds:
    if len(qt.points_in()) > S:
        points = qt.points_in()
        cutoff = int(len(points) * con)
        pnts_sorted = sorted(points, key=lambda x: x.anomaly_score)
        cv = pnts_sorted[cutoff].anomaly_score
        anomalies = pnts_sorted[:cutoff]
        normal_points = pnts_sorted[cutoff:]
        for pnt in anomalies:
            pnt.is_outlier = 0
        for pnt in normal_points:
            pnt.is_outlier = 1

        for j in range(W):
            index = int(400 * (np.random.rand() ** 4))
            qt.delete(pnts_sorted[index])
        plt.clf()
        plt.scatter([p.x for p in qt.points_in()], [p.y for p in qt.points_in()],
                    c=colors[[(p.is_outlier + 1) // 2 for p in qt.points_in()]], s=32)
        plt.savefig('../output/streaming_' + str(fig_nr) + '.pdf')
        fig_nr += 1
        plt.pause(0.0001)

    qt.insert(point)
    if point.anomaly_score < cv:
        point.is_outlier = 0
    i += 1

    print("points processed: %i" % i)
    plt.clf()
    plt.scatter([p.x for p in qt.points_in()], [p.y for p in qt.points_in()],
                c=colors[[(p.is_outlier + 1) // 2 for p in qt.points_in()]], s=32)
    plt.pause(0.0001)

plt.scatter([p.x for p in qt.points_in()], [p.y for p in qt.points_in()],
            c=colors[[(p.is_outlier + 1) // 2 for p in qt.points_in()]], s=32)
plt.savefig('../output/streaming_' + str(fig_nr) + '.pdf')
plt.show()
