import math


class Point:
    """A point located in d-dimensional space

    Each Point object may be associated with an anomaly score and outlier classification.

    """

    def __init__(self, coordinates, anomaly_score=0, is_outlier=0, container=None):
        self.coordinates = coordinates
        self.anomaly_score = anomaly_score
        self.is_outlier = is_outlier
        self.container = container

    def __repr__(self):
        return 'point {} has anomaly score: {}, is anomaly: {}'.format(str(self.coordinates), repr(self.anomaly_score),
                                                                       repr(self.is_outlier))

    def __str__(self):
        return ("P [" + ', '.join(['%.2f'] * len(self.coordinates)) + "]") % tuple(self.coordinates)


class Hypercube:
    """A Hypercube that possibly has different side lengths in all dimensions"""

    def __init__(self, center, radii):
        self.center = center
        self.radii = radii

    def __str__(self):
        return 'center: ' + str(self.center) + ' radii: ' + str(self.radii)


class RDQuadTree:
    """A QuadTree with an arbitrary number of dimensions, which splits in every dimension on every split

    Therefore the number of data points needs to be much larger than the number of dimensions (2^d) to be of
    any practical use.
    """

    def __init__(self, hc, max_points=1, depth=0, parent=None, max_depth=25, root=None, order=None):
        """Initialize this node of the quadtree.

        hc is the hypercube that defines the boundary of the node

        """

        self.root = root
        self.hc = hc
        self.max_points = max_points
        self.points = []
        self.depth = depth
        self.children = []
        self.parent = parent
        self.divided = False
        self.clean_divided = False
        self.points_inside = []
        self.max_depth = max_depth
        self.divisions = 0
        self.order = order
        self.random_shift = 0
        self.penalty = math.ceil(math.log(self.max_depth, 2))
        if self.depth == 0:
            self.root = self

    def __str__(self):
        s = str(self.hc)
        if len(self.points) > 0:
            s += ' CONTAINS ' + str(self.points[0])
        if not self.divided:
            return s
        s += '\n'
        for child in self.children:
            s += ' ' + str(child)
        return s

    def has_in(self, point):
        for i in range(len(point.coordinates)):
            if abs(point.coordinates[i] - self.hc.center[i]) > self.hc.radii[i]:
                return False
        return True

    def divide(self):
        """Divide this node by spawning 2 children nodes."""

        rd = self.order[self.depth]
        c1 = Hypercube(self.hc.center.copy(), self.hc.radii.copy())
        c1.radii[rd] = c1.radii[rd] / 2
        c1.center[rd] += (c1.radii[rd])
        c2 = Hypercube(self.hc.center.copy(), self.hc.radii.copy())
        c2.radii[rd] = c2.radii[rd] / 2
        c2.center[rd] -= (c2.radii[rd])
        self.children.append(RDQuadTree(hc=c1, depth=self.depth + 1, max_points=self.max_points, parent=self,
                                        root=self.root, order=self.order, max_depth=self.max_depth))
        self.children.append(RDQuadTree(hc=c2, depth=self.depth + 1, max_points=self.max_points, parent=self,
                                        root=self.root, order=self.order, max_depth=self.max_depth))

        if len(self.points) > 0:
            for p in self.points:
                if abs(p.coordinates[rd] - self.children[0].hc.center[rd]) <= self.children[0].hc.radii[rd]:
                    fitting_child = self.children[0]
                else:
                    fitting_child = self.children[1]
                if fitting_child:
                    fitting_child.insert(p)
                else:
                    print("error, none of the children seems to be able to fit the point", p)

        self.divided = True
        self.root.divisions += 1
        return True

    def insert(self, p):
        """Inserting a point with n dimensions into the n-dimensional tree
        We split in all dimensions when there is no room for the point.
        """

        # if not self.has_in(p):
        # print("error: trying to insert point that is outside domain of tree")
        # return False

        if self.depth == 0:
            self.points_inside.append(p)

        if self.depth == self.max_depth:
            self.points.append(p)
            p.anomaly_score = self.depth + math.ceil(math.log(self.max_depth, 2))
            p.container = self
            return True

        if len(self.points) < self.max_points and not self.divided:
            self.points.append(p)
            p.anomaly_score = self.depth
            p.container = self
            return True

        if not self.divided:
            self.divide()

        fitting_child = None
        rd = self.order[self.depth]
        if abs(p.coordinates[rd] - self.children[0].hc.center[rd]) <= self.children[0].hc.radii[rd]:
            fitting_child = self.children[0]
        else:
            fitting_child = self.children[1]
        if fitting_child:
            fitting_child.insert(p)
        else:
            print("error, none of the children seems to be able to fit the point", p, self.depth)
        return True

    def clean_insert(self, p):
        """Inserting a point with n dimensions into the n-dimensional tree
                We split in all dimensions when there is no room for the point.
                """

        if self.depth == 0:
            self.points_inside.append(p)

        if self.depth == self.max_depth:
            p.anomaly_score = self.depth + self.penalty
            return True

        if len(self.points) < self.max_points and not self.divided:
            p.anomaly_score = self.depth
            return True

        if not self.divided and not self.clean_divided:
            self.clean_divide()

        rd = self.order[self.depth]
        if abs(p.coordinates[rd] - self.children[0].hc.center[rd]) <= self.children[0].hc.radii[rd]:
            fitting_child = self.children[0]
        else:
            fitting_child = self.children[1]

        fitting_child.clean_insert(p)

    def clean_divide(self):
        """Divide this node by spawning 2 children nodes."""

        rd = self.order[self.depth]
        c1 = Hypercube(self.hc.center.copy(), self.hc.radii.copy())
        c1.radii[rd] = c1.radii[rd] / 2
        c1.center[rd] += (c1.radii[rd])
        c2 = Hypercube(self.hc.center.copy(), self.hc.radii.copy())
        c2.radii[rd] = c2.radii[rd] / 2
        c2.center[rd] -= (c2.radii[rd])
        self.children.append(RDQuadTree(hc=c1, depth=self.depth + 1, max_points=self.max_points, parent=self,
                                        root=self.root, order=self.order, max_depth=self.max_depth))
        self.children.append(RDQuadTree(hc=c2, depth=self.depth + 1, max_points=self.max_points, parent=self,
                                        root=self.root, order=self.order, max_depth=self.max_depth))

        self.clean_divided = True
        self.root.divisions += 1
        return True
