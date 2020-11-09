import numpy as np
import matplotlib.pyplot as plt
from quadtree import Point, Rect, QuadTree
from sklearn.datasets import make_blobs

DPI = 72
# np.random.seed(80)

width, height = 600, 600

N = 500

random_shift = np.random.rand(2) * 300
print("random horizontal shift:", random_shift[0])
print("random vertical shift:", random_shift[1])

coords = np.random.rand(N, 2) * height / 2 + random_shift
coords = make_blobs(n_samples=500, n_features=2, center_box=(0,300), cluster_std=30)[0] + random_shift
points = [Point(*coord) for coord in coords]


domain = Rect(width / 2, height / 2, width, height)
qtree = QuadTree(domain, 1)
for point in points:
    qtree.insert(point)
    qtree.score(point)
    print(point.payload)
    if point.payload < 650:
        point.payload = 0
    else:
        point.payload = 1
    print(point.payload)

print('Number of points in the domain =', len(qtree))

fig = plt.figure(figsize=(700 / DPI, 500 / DPI), dpi=DPI)
ax = plt.subplot()
ax.set_xlim(0, width)
ax.set_ylim(0, height)
qtree.draw(ax)

colors = np.array(['#377eb8', '#ff7f00'])
ax.scatter([p.x for p in points], [p.y for p in points], c=[p.payload for p in points], s=4)
ax.set_xticks([])
ax.set_yticks([])

region = Rect(140, 190, 150, 150)
found_points = []
qtree.query(region, found_points)
print('Number of found points =', len(found_points))

ax.scatter([p.x for p in found_points], [p.y for p in found_points],
           facecolors='none', edgecolors='r', s=32)

region.draw(ax, c='r')

ax.invert_yaxis()
plt.tight_layout()
plt.show()
