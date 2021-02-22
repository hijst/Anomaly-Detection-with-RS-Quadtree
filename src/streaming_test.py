import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from src.quadtree import Point, Rect, QuadTree

# Number of points, Sample size, Window size, contamination, random shift
N = 1000
S = int(N / 2)
W = 100
con = 0.10
random_shift = 0
colors = np.array(['#ff7f00', '#377eb8'])

# Generate some random data
base_data = make_blobs(n_samples=N, n_features=2, center_box=(0, 300), cluster_std=20, random_state=3)[0]
stream = [point + random_shift for point in base_data]
ds = [Point(*coord) for coord in stream]

start_time = time.time()

domain = Rect(400, 400, 800, 800)
qt = QuadTree(domain, 1)
added_points = []
counter = 0
i = 0
cv = 0
fig_nr = 0
for point in ds:
    added_points.append(point)
    if len(added_points) > S:
        qt.delete(added_points[0])
        added_points = added_points[1:]

    qt.insert(point)
    if point.anomaly_score < cv:
        point.is_outlier = 0
    counter += 1
    i += 1

    if counter == W:
        counter = 0
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
        plt.clf()
        plt.scatter([p.x for p in added_points], [p.y for p in added_points],
                    c=colors[[(p.is_outlier + 1) // 2 for p in added_points]], s=32)
        plt.savefig('../output/streaming_'+str(fig_nr)+'.pdf')
        fig_nr += 1
        plt.pause(0.0001)

    print("points processed: %i" % i)
    plt.clf()
    plt.scatter([p.x for p in added_points], [p.y for p in added_points],
                c=colors[[(p.is_outlier + 1) // 2 for p in added_points]], s=32)
    plt.pause(0.0001)

plt.scatter([p.x for p in added_points], [p.y for p in added_points],
            c=colors[[(p.is_outlier + 1) // 2 for p in added_points]], s=32)
plt.show()
