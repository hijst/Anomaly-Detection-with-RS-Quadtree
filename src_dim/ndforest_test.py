import numpy as np
from src_dim.d_quadtree import NDQuadTree, Point, Hypercube
from sklearn.datasets import make_blobs
from src_dim.ndforest import NDForest

big_small_blob = make_blobs(centers=[[0, 0, 0, 0], [4, 0, 0, 0]], n_samples=[1000, 10],
                            n_features=3, cluster_std=[.2, 1.5])[0]

clf = NDForest(contamination=0.01, k=10, points=big_small_blob)
result = clf.fit_predict()
filtered_results = [p for p in result if p.is_outlier == 0]
print(filtered_results)
