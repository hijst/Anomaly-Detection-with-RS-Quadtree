import numpy as np
from src.quadtree import Point, Rect, QuadTree
import pyspark as ps


class RSQT:

    def __init__(self, contamination=0.10, ds=0):
        self.points = []
        self.contamination = contamination
        self.ds = ds

    def fill_quadtree(self, data, rs=1):
        """Apply random shift to points, then fill the quadtree with the points and score the points."""

        #mn = np.amin(data)
        #mx = np.amax(data)
        mn = -8
        mx = 8
        self.ds = int((mx - mn)) + 1

        if rs == 0:
            random_shift = [0, 0]
        else:
            random_shift = np.random.rand(2) * self.ds
        print("random horizontal shift:", round(random_shift[0], 2))
        print("random vertical shift:", round(random_shift[1], 2))

        # Add a random shift to the point set and generate the points
        coords = [point + random_shift for point in data]
        pts = [Point(*coord) for coord in coords]

        domain = Rect(mx, mx, 2 * self.ds, 2 * self.ds)
        qt = QuadTree(domain, 1)
        for pt in pts:
            qt.insert(pt)
        for pt in pts:
            qt.score_depth(pt)  # score based on depth in the tree
        print('number of points in the domain =', len(qt))
        return qt, pts

    def merge_quadtrees(self, qt1, qt2, depth=0, max_depth=7, merged_quadtree=None):
        if depth == 0:
            merged_quadtree = qt1
            merged_quadtree.max_points = 1000

        if depth < max_depth:
            merged_quadtree.points = qt1.points + qt2.points
            if qt2.divided:
                if not qt1.divided:
                    qt1.e_divide()
                if not merged_quadtree.divided:
                    merged_quadtree.e_divide()
                self.merge_quadtrees(qt1.nw, qt2.nw, depth=depth + 1, merged_quadtree=merged_quadtree.nw)
                self.merge_quadtrees(qt1.ne, qt2.ne, depth=depth + 1, merged_quadtree=merged_quadtree.ne)
                self.merge_quadtrees(qt1.sw, qt2.sw, depth=depth + 1, merged_quadtree=merged_quadtree.sw)
                self.merge_quadtrees(qt1.se, qt2.se, depth=depth + 1, merged_quadtree=merged_quadtree.se)

        if depth == max_depth:
            print("qt1 points_in: ", qt1.points_in())
            print("qt2 points_in: ", qt2.points_in())
            merged_quadtree.points = qt1.points_in() + qt2.points_in()
            if merged_quadtree.divided:
                del merged_quadtree.nw, merged_quadtree.ne, merged_quadtree.sw, merged_quadtree.se

        if depth == 0:
            return merged_quadtree

    def fit_predict(self, data, k=10):
        cutoff = int(len(data) * self.contamination)

        qtree, pnts = self.fill_quadtree(data, rs=0)

        for i in range(k - 1):
            qt, pts = self.fill_quadtree(data)
            for j in range(len(pnts)):
                pnts[j].anomaly_score += pts[j].anomaly_score

        pnts_sorted = sorted(pnts, key=lambda x: x.anomaly_score)
        T = pnts_sorted[cutoff].anomaly_score

        for pnt in pnts:
            if pnt.anomaly_score < T:
                pnt.is_outlier = 0

        y_pred = []
        for pnt in pnts:
            y_pred.append(pnt.is_outlier)
        return pnts, y_pred

    def fit_predict_qt(self, data, k=10):
        cutoff = int(len(data) * self.contamination)

        qtree, pnts = self.fill_quadtree(data, rs=0)

        for i in range(k - 1):
            qt, pts = self.fill_quadtree(data)
            for j in range(len(pnts)):
                pnts[j].anomaly_score += pts[j].anomaly_score

        pnts_sorted = sorted(pnts, key=lambda x: x.anomaly_score)
        T = pnts_sorted[cutoff].anomaly_score

        for pnt in pnts:
            if pnt.anomaly_score < T:
                pnt.is_outlier = 0

        y_pred = []
        for pnt in pnts:
            y_pred.append(pnt.is_outlier)
        return pnts, y_pred, qtree

    def predict(self, qt):
        pts = qt.query(qt.boundary, [])
        cutoff = int(len(pts) * self.contamination)
        for point in pts:
            qt.score_depth(point)

        pnts_sorted = sorted(pts, key=lambda x: x.anomaly_score)
        T = pnts_sorted[cutoff].anomaly_score

        for pnt in pts:
            if pnt.anomaly_score < T:
                pnt.is_outlier = 0
            else:
                pnt.is_outlier = 1
        return pts