from src_dim.d_quadtree import NDQuadTree, Point, Hypercube
import numpy as np

class NDForest:
    """Forest of ND trees, where scores are accumulated to form a measure of anomalousness.

    includes a fit and a predict method, contamination can be set to define number of outliers.
    """

    def __init__(self, contamination=0.1, k=5, domain=None, points=None):

        self.contamination = contamination
        self.k = k
        self.domain = domain
        self.points = points
        if points is not None:
            self.dimensions = len(self.points[0])

    def fill_trees(self):
        """"converts the coordinates to point objects and inserts them into k trees with random shifts."""
        for i in range(self.k):
            random_shift = random_shift = np.random.rand(self.dimensions) * self.domain