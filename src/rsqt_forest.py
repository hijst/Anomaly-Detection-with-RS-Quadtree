import numpy as np
from src.quadtree import Point, Rect, QuadTree


width, height = 600, 600


class RSQT:

    def __init__(self, contamination=0.10, ds=0):
        self.points = []
        self.contamination = contamination
        self.ds = ds

    def fill_quadtree(self, data, rs=1):
        """Apply random shift to points, then fill the quadtree with the points and score the points."""

        mn = np.amin(data)
        mx = np.amax(data)
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

    def fit_predict(self, data, k=20):
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
