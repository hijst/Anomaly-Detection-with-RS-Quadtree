from src_dim.rd_quadtree import RDQuadTree, Point, Hypercube
import numpy as np
import time


class RDForest:
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
            mx = np.amax(self.points) + 0.01
            mn = np.amin(self.points) - 0.01  # safety margin to avoid rounding issues

            print("min: ", mn, " max: ", mx)

            rn = abs(mx - mn)  # range of hypercube
            mxd = mx + rn  # max of hypercube
            cnv = (mn + mxd) / 2  # center value
            cn = [cnv] * self.dimensions  # center coordinates
            rns = [rn] * self.dimensions
            self.domain = Hypercube(cn, rns)  # hypercube to contain all (shifted) points
            for i in range(k):
                self.trees.append(RDQuadTree(self.domain, max_points=i+1))

            print("max points per tree: ", [tree.max_points for tree in self.trees])

    def fit(self):
        """converts the coordinates to point objects and inserts them into k trees with random shifts."""

        base_coords = [point for point in self.points]
        base_pts = [Point(base_coord) for base_coord in base_coords]
        for base_pt in base_pts:
            self.trees[0].insert(base_pt)
        print("divisions: ", self.trees[0].divisions)
        for tree in self.trees[1:]:

            random_shift = np.random.rand(self.dimensions) * self.domain.radii[0]
            coords = [point + random_shift for point in self.points]
            pts = [Point(coord) for coord in coords]
            starttime = time.time()
            for pt in pts:
                tree.insert(pt)
            print("time to fit this tree: ", time.time() - starttime)
            print("divisions: ", tree.divisions)
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
