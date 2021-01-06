import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from src.rsqt_forest import RSQT

start_time = time.time()

N = 300  # number of points

rectangles = []
for i in range(int(N / 12)):
    rectangles.append([4. * (np.random.random() - 0.5), 2])
    rectangles.append([4. * (np.random.random() - 0.5), -2])
    rectangles.append([2, 4. * (np.random.random() - 0.5)])
    rectangles.append([-2, 4. * (np.random.random() - 0.5)])
for i in range(int(N / 6)):
    rectangles.append([8. * (np.random.random() - 0.5), 4])
    rectangles.append([8. * (np.random.random() - 0.5), -4])
    rectangles.append([4, 8. * (np.random.random() - 0.5)])
    rectangles.append([-4, 8. * (np.random.random() - 0.5)])

rng = np.random.RandomState(42)
rectangles = np.concatenate([rectangles, rng.uniform(low=-6, high=6,
                                                     size=(50, 2))], axis=0)
rect1 = []
rect2 = []
rect3 = []

for point in rectangles:
    if -2 < point[0] < 2 and -2 < point[1] < 2:
        rect1.append(point)
    elif -4 < point[0] < 4 and -4 < point[1] < 4:
        rect2.append(point)
    else:
        rect3.append(point)


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
clf.contamination = 0.166

points, y_pred, qtree = clf.fit_predict_qt(rect1)
points2, y_pred2, qtree2 = clf.fit_predict_qt(rect2)
points3, y_pred3, qtree3 = clf.fit_predict_qt(rect3)

plot_qt(points)
plot_qt(points2)
plot_qt(points3)

merged_qt = clf.merge_quadtrees(qtree, qtree2)
merged_qt = clf.merge_quadtrees(merged_qt, qtree3)
points = merged_qt.query(merged_qt.boundary, [])
print("--- running time: %s seconds ---" % (round(time.time() - start_time, 2)))

print(len(points))
print(len(merged_qt))
plot_qt(points)


points = clf.predict(merged_qt)
plot_qt(points)

for point in points:
    if point.is_outlier == 0:
        merged_qt.delete(point)
        print("removed")

points = merged_qt.query(merged_qt.boundary, [])
print(len(points))
print(len(merged_qt))
plot_qt(points)
print(str(merged_qt))
