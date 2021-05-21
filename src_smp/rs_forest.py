from src_smp.rs_quadtree import RDQuadTree, Point, Hypercube
import numpy as np
import time
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


def tdplot(data):
    plt.figure()
    ax = plt.axes(projection='3d')
    colors = np.array(['#ff7f00', '#377eb8'])
    xs = [x.coordinates[0] for x in data]
    ys = [y.coordinates[1] for y in data]
    zs = [z.coordinates[2] for z in data]
    ax.scatter3D(xs, ys, zs, c=colors[[(p.is_outlier + 1) // 2 for p in data]])
    # plt.savefig('../output/massive_data_no_noise.pdf')
    plt.show()


class RSForest:
    """Forest of ND trees, where scores are accumulated to form a measure of anomalousness.

    includes a fit and a predict method, contamination can be set to define number of outliers.
    """

    def __init__(self, contamination=0.1, k=5, points=None, granularity=5, sample_size=128, answers=True):

        self.contamination = contamination
        self.k = k
        self.points = points
        self.trees = []
        self.fitted = False
        self.granularity = granularity
        self.sample_size = sample_size
        self.answers = answers
        if self.points is not None:
            # print("self points: ", self.points)
            self.r_sample, self.test_set = train_test_split(self.points, train_size=self.sample_size, random_state=42)
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
            m_depth = 4 * math.ceil(math.log(len(self.r_sample), 2))
            print("max depth: ", m_depth)
            if self.answers:
                dim_order = [np.random.randint(self.dimensions-1, size=m_depth) for i in range(k)]
            else:
                dim_order = [np.random.randint(self.dimensions, size=m_depth) for i in range(k)]
            # dim_order = [i for i in range(self.dimensions-1)] * int(1+(m_depth/(self.dimensions-1)))
            # print("dim order: ", dim_order)
            for i in range(k):
                # print("order: ", dim_order[i])
                self.trees.append(RDQuadTree(self.domain, order=dim_order[i],
                                             max_depth=m_depth, max_points=math.ceil((i + 1) / self.granularity)))

            print("max points per tree: ", [tree.max_points for tree in self.trees])

    def predict_helper(self, t):
        stt = time.time()
        test_set_pnts = [pn + t.random_shift for pn in self.test_set]
        test_set_points = [Point(coord) for coord in test_set_pnts]
        for pt in test_set_points:
            t.clean_insert(pt)
        print("time to predict this tree: ", time.time() - stt)

    def fit(self):
        """converts the coordinates to point objects and inserts them into k trees with random shifts."""

        base_coords = [point for point in self.r_sample]
        base_pts = [Point(base_coord) for base_coord in base_coords]
        for base_pt in base_pts:
            self.trees[0].insert(base_pt)
        # print("divisions: ", self.trees[0].divisions)
        for tree in self.trees[1:]:

            tree.random_shift = np.random.rand(self.dimensions) * self.domain.radii[0]
            coords = [point + tree.random_shift for point in self.r_sample]
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
        base_pnts = [point for point in self.test_set]
        base_pts = [Point(base_coord) for base_coord in base_pnts]
        stt = time.time()
        # Parallel(n_jobs=8, prefer="threads")(delayed(self.trees[0].clean_insert)(base_p) for base_p in base_pts)
        for base_p in base_pts:
            self.trees[0].clean_insert(base_p)
        print("time to predict this tree: ", time.time() - stt)
        Parallel(n_jobs=4, prefer="threads")(delayed(self.predict_helper)(t) for t in self.trees[1:])
        # for t in self.trees[1:]:
            # self.predict_helper(t)
        score_points = self.trees[0].points_inside
        for i in range(len(self.points)):
            for tree in self.trees[1:]:
                score_points[i].anomaly_score += tree.points_inside[i].anomaly_score

        N = len(self.trees)
        m_d = self.trees[0].max_depth
        n = 2 ** m_d
        for i in range(len(self.points)):
            hx = score_points[i].anomaly_score
            Ehx = hx / N
            print("Ehx: ", Ehx)
            cn = 2 * np.log(n-1) + .5772156649 - (2*(n-1)/n)
            print("cn: ", cn)
            score_points[i].anomaly_score = 2 ** - (Ehx / cn)

        cutoff = int(len(score_points) * self.contamination)

        pnts_sorted = sorted(score_points, key=lambda x: x.anomaly_score, reverse=True)
        T = pnts_sorted[cutoff].anomaly_score

        for pnt in score_points:
            if pnt.anomaly_score >= T:
                pnt.is_outlier = -1
            else:
                pnt.is_outlier = 1
        answers = []
        for point in score_points:
            answers.append(point.coordinates[-1])
        print("time to predict this tree: ", time.time() - stt)
        return score_points, answers

    def fit_predict(self):
        self.fit()
        result, answers = self.predict()
        return result, answers
