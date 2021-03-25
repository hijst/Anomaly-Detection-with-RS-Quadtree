import itertools as it
import scipy.spatial.distance as spd


class Point:
    """A point located in d-dimensional space

    Each Point object may be associated with an anomaly score and outlier classification.

    """

    def __init__(self, coordinates, anomaly_score=0, is_outlier=1, container=None):
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
    """A Hypercube that has the same side length in all dimensions"""

    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def __str__(self):
        return 'center: ' + str(self.center) + ' radius: ' + str(self.radius)


class NDQuadTree:
    """A QuadTree with an arbitrary number of dimensions, which splits in every dimension on every split

    Therefore the number of data points needs to be much larger than the number of dimensions (2^d) to be of
    any practical use.
    """

    def __init__(self, hc, max_points=1, depth=0, parent=None, max_depth=25):
        """Initialize this node of the quadtree.

        hc is the hypercube that defines the boundary of the node

        """

        self.hc = hc
        self.max_points = max_points
        self.points = []
        self.depth = depth
        self.children = []
        self.parent = parent
        self.divided = False
        self.points_inside = []
        self.max_depth = max_depth

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
            if spd.cityblock(point.coordinates[i], self.hc.center[i]) > self.hc.radius:
                return False
        return True

    def divide(self):
        """Divide (branch) this node by spawning 2^d children nodes."""

        radius = self.hc.radius / 2
        dimensions = len(self.hc.center)
        combinator = []
        child_centers = []
        for i in range(dimensions):
            combinator.append(i)
        for i in range(dimensions + 1):
            child_centers.extend(it.combinations(combinator, i))
        children_centers = []
        for child_center in child_centers:
            child_c = []
            for i in range(dimensions):
                if i in child_center:
                    child_c.append(self.hc.center[i] + radius)
                else:
                    child_c.append(self.hc.center[i] - radius)
            children_centers.append(child_c)
        for child in children_centers:
            hypercube = Hypercube(child, radius)
            self.children.append(NDQuadTree(hc=hypercube, depth=self.depth + 1, parent=self))

        if len(self.points) > 0:
            for p in self.points:
                fitting_child = next((x for x in self.children if x.has_in(p)), None)
                if fitting_child:
                    fitting_child.insert(p)
                    self.points.remove(p)
                else:
                    print("error, none of the children seems to be able to fit the point", p)

        self.divided = True

    def insert(self, p):
        """Inserting a point with n dimensions into the n-dimensional tree
        We split in all dimensions when there is no room for the point.
        """

        if not self.has_in(p):
            print("error: trying to insert point that is outside domain of tree")
            return False

        if self.depth == 0:
            self.points_inside.append(p)

        if self.depth == self.max_depth:
            self.points.append(p)
            p.anomaly_score = self.depth
            p.container = self
            return True

        if len(self.points) < self.max_points and not self.divided:
            self.points.append(p)
            p.anomaly_score = self.depth
            p.container = self
            return True

        if not self.divided:
            self.divide()

        fitting_child = next((x for x in self.children if x.has_in(p)), None)
        if fitting_child:
            fitting_child.insert(p)
        else:
            print("error, none of the children seems to be able to fit the point", p, self.depth)
        return True
