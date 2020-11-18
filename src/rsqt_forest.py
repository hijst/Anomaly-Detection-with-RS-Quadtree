import numpy as np
from src.quadtree import Point, Rect, QuadTree


width, height = 600, 600


class RSQT:

    def __init__(self, contamination=0.10):
        self.points = []
        self.contamination = contamination

    def fill_quadtree(self, data, rs=1):
        """Apply random shift to points, then fill the quadtree with the points and score the points."""

        if rs == 0:
            random_shift = [0, 0]
        else:
            random_shift = np.random.rand(2) * 300
        print("random horizontal shift:", round(random_shift[0], 2))
        print("random vertical shift:", round(random_shift[1], 2))

        # Add a random shift to the point set and generate the points
        coords = [point + random_shift for point in data]
        pts = [Point(*coord) for coord in coords]

        domain = Rect(width / 2, height / 2, width, height)
        qt = QuadTree(domain, 1)
        for pt in pts:
            qt.insert(pt)
        for pt in pts:
            qt.score(pt)  # score based on number of points along a points path from root to leaf
            # qt.score_depth(pt)  # score based on depth in the tree
        print('Number of points in the domain =', len(qt))
        return qt, pts

    def fit_predict(self, data, k=5):
        a = int(len(data) * self.contamination)

        qtree, pnts = self.fill_quadtree(data, rs=0)

        for i in range(k - 1):
            qt, pts = self.fill_quadtree(data)
            for j in range(len(pnts)):
                pnts[j].payload += pts[j].payload

        pnts.sort(key=lambda x: x.payload)
        T = pnts[a].payload

        for pnt in pnts:
            if pnt.payload > T:
                pnt.payload = 1
            else:
                pnt.payload = -1

        y_pred = []
        for pnt in pnts:
            y_pred.append(pnt.payload)
        return pnts, y_pred
