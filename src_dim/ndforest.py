from src_dim.d_quadtree import NDQuadTree, Point, Hypercube
import numpy as np


class NDForest:
    """Forest of ND trees, where scores are accumulated to form a measure of anomalousness.

    includes a fit and a predict method, contamination can be set to define number of outliers.
    """

    def __init__(self, contamination=0.1, k=5, points=None):

        self.contamination = contamination
        self.k = k
        self.points = points
        self.trees = []
        self.fitted = False
        if self.points is not None:
            self.dimensions = len(self.points[0])
            mx = np.amax(self.points)
            mn = np.amin(self.points)

            rn = abs(mx - mn)  # range of hypercube
            mxd = mx + rn  # max of hypercube
            cnv = (mn + mxd) / 2  # center value
            cn = [cnv] * self.dimensions  # center coordinates
            self.domain = Hypercube(cn, rn)  # hypercube to contain all (shifted) points
            for i in range(k):
                self.trees.append(NDQuadTree(self.domain))

    def fit(self):
        """converts the coordinates to point objects and inserts them into k trees with random shifts."""

        base_coords = [point for point in self.points]
        base_pts = [Point(base_coord) for base_coord in base_coords]
        for base_pt in base_pts:
            self.trees[0].insert(base_pt)
        for tree in self.trees[1:]:
            random_shift = np.random.rand(self.dimensions) * self.domain.radius
            coords = [point + random_shift for point in self.points]
            pts = [Point(coord) for coord in coords]
            for pt in pts:
                tree.insert(pt)
        self.fitted = True

    def predict(self):
        """combine the scores from all trees and return the points with lowest attached scores"""

        if not self.fitted:
            print("Not fitted, fit the algorithm first")
            return False

        score_points = self.trees[0].points_inside
        for i in range(len(self.points)):
            for tree in self.trees[1:]:
                score_points[i].anomaly_score += tree.points_inside[i].anomaly_score

        cutoff = int(len(score_points) * self.contamination)

        pnts_sorted = sorted(score_points, key=lambda x: x.anomaly_score)
        T = pnts_sorted[cutoff].anomaly_score

        for pnt in score_points:
            if pnt.anomaly_score <= T:
                pnt.is_outlier = 0
            else:
                pnt.is_outlier = 1
        return score_points

    def fit_predict(self):
        self.fit()
        result = self.predict()
        return result
