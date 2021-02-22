import numpy as np


class Point:
    """A point located at (x,y) in 2D space.

    Each Point object may be associated with an anomaly score and outlier classification.

    """

    def __init__(self, x, y, anomaly_score=0, is_outlier=1, container=None):
        self.x, self.y = x, y
        self.anomaly_score = anomaly_score
        self.is_outlier = is_outlier
        self.container = container

    def __repr__(self):
        return '{}: {}'.format(str((self.x, self.y)), repr(self.anomaly_score), repr(self.is_outlier))

    def __str__(self):
        return 'P({:.2f}, {:.2f})'.format(self.x, self.y)

    def distance_to(self, other):
        try:
            other_x, other_y = other.x, other.y
        except AttributeError:
            other_x, other_y = other
        return np.hypot(self.x - other_x, self.y - other_y)


class Rect:
    """A rectangle centred at (cx, cy) with width w and height h."""

    def __init__(self, cx, cy, w, h):
        self.cx, self.cy = cx, cy
        self.w, self.h = w, h
        self.west_edge, self.east_edge = cx - w / 2, cx + w / 2
        self.north_edge, self.south_edge = cy - h / 2, cy + h / 2

    def __repr__(self):
        return str((self.west_edge, self.east_edge, self.north_edge,
                    self.south_edge))

    def __str__(self):
        return '({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(self.west_edge,
                                                         self.north_edge, self.east_edge, self.south_edge)

    def contains(self, point):
        """Is point (a Point object or (x,y) tuple) inside this Rect?"""

        try:
            point_x, point_y = point.x, point.y
        except AttributeError:
            point_x, point_y = point

        return (self.west_edge <= point_x < self.east_edge and
                self.north_edge <= point_y < self.south_edge)

    def intersects(self, other):
        """Does Rect object other intersect this Rect?"""
        return not (other.west_edge > self.east_edge or
                    other.east_edge < self.west_edge or
                    other.north_edge > self.south_edge or
                    other.south_edge < self.north_edge)

    def draw(self, ax, c='k', lw=0.2, **kwargs):
        x1, y1 = self.west_edge, self.north_edge
        x2, y2 = self.east_edge, self.south_edge
        ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], c=c, lw=lw, **kwargs)


class QuadTree:
    """A class implementing a quadtree."""

    def __init__(self, boundary, max_points=1, depth=0, domain=0, parent=None):
        """Initialize this node of the quadtree.

        boundary is a Rect object defining the region from which points are
        placed into this node; max_points is the maximum number of points the
        node can hold before it must divide (branch into four more nodes);
        depth keeps track of how deep into the quadtree this node lies.

        """

        self.boundary = boundary
        self.max_points = max_points
        self.points = []
        self.depth = depth
        self.parent = parent
        # A flag to indicate whether this node has divided (branched) or not.
        self.divided = False

    def __str__(self):
        """Return a string representation of this node, suitably formatted."""
        sp = ' ' * self.depth * 2
        s = str(self.boundary) + '\n'
        s += sp + ', '.join(str(point) for point in self.points)
        if not self.divided:
            return s
        return s + '\n' + '\n'.join([
            sp + 'nw: ' + str(self.nw), sp + 'ne: ' + str(self.ne),
            sp + 'se: ' + str(self.se), sp + 'sw: ' + str(self.sw)])

    def divide(self):
        """Divide (branch) this node by spawning four children nodes."""

        cx, cy = self.boundary.cx, self.boundary.cy
        w, h = self.boundary.w / 2, self.boundary.h / 2
        # The boundaries of the four children nodes are "northwest",
        # "northeast", "southeast" and "southwest" quadrants within the
        # boundary of the current node.

        # for point in self.points:
        #     point.payload += len(self.points)

        self.nw = QuadTree(Rect(cx - w / 2, cy - h / 2, w, h),
                           self.max_points, self.depth + 1, parent=self)

        self.ne = QuadTree(Rect(cx + w / 2, cy - h / 2, w, h),
                           self.max_points, self.depth + 1, parent=self)

        self.se = QuadTree(Rect(cx + w / 2, cy + h / 2, w, h),
                           self.max_points, self.depth + 1, parent=self)

        self.sw = QuadTree(Rect(cx - w / 2, cy + h / 2, w, h),
                           self.max_points, self.depth + 1, parent=self)

        if self.points:
            for point in self.points:
                self.nw.insert(point)
                self.ne.insert(point)
                self.sw.insert(point)
                self.se.insert(point)
                self.points.remove(point)

        self.divided = True

    def e_divide(self):
        """Divide (branch) this node by spawning four children nodes."""

        cx, cy = self.boundary.cx, self.boundary.cy
        w, h = self.boundary.w / 2, self.boundary.h / 2
        # The boundaries of the four children nodes are "northwest",
        # "northeast", "southeast" and "southwest" quadrants within the
        # boundary of the current node.

        # for point in self.points:
        #     point.payload += len(self.points)

        self.nw = QuadTree(Rect(cx - w / 2, cy - h / 2, w, h),
                           self.max_points, self.depth + 1, parent=self)

        self.ne = QuadTree(Rect(cx + w / 2, cy - h / 2, w, h),
                           self.max_points, self.depth + 1, parent=self)

        self.se = QuadTree(Rect(cx + w / 2, cy + h / 2, w, h),
                           self.max_points, self.depth + 1, parent=self)

        self.sw = QuadTree(Rect(cx - w / 2, cy + h / 2, w, h),
                           self.max_points, self.depth + 1, parent=self)

        self.divided = True

    def collapse(self):
        """Undo the dividing of a cell by removing its 4 children nodes and resetting the divided attribute"""
        self.points.extend(self.nw.points)
        self.points.extend(self.ne.points)
        self.points.extend(self.sw.points)
        self.points.extend(self.se.points)
        for point in self.points:
            point.container = self
            point.anomaly_score = self.depth
        del self.nw, self.ne, self.sw, self.se
        self.divided = False

    def insert(self, point):
        """Try to insert Point point into this quadtree."""

        if not self.boundary.contains(point):
            # The point does not lie inside boundary: bail.
            return False
        if not self.divided and len(self.points) < self.max_points:
            # There's room for our point without dividing the QuadTree.
            self.points.append(point)
            point.container = self
            point.anomaly_score = self.depth
            return True

        # No room: divide if necessary, then try the sub-quads.
        if not self.divided:
            self.divide()

        return (self.ne.insert(point) or
                self.nw.insert(point) or
                self.se.insert(point) or
                self.sw.insert(point))

    def delete(self, point):
        """Delete a point from the quadtree and update the regions"""

        if not self.boundary.contains(point):
            return False

        if self.divided:
            self.ne.delete(point)
            self.nw.delete(point)
            self.se.delete(point)
            self.sw.delete(point)

        if point in self.points:
            self.points.remove(point)
            del point

        if self.divided and not (self.nw.divided or self.ne.divided or self.sw.divided or self.se.divided):
            if len(self.nw.points) + len(self.ne.points) + len(self.sw.points) +\
               len(self.se.points) <= self.max_points:
                self.collapse()
                print("collapsed")
                return True

    def score(self, point):
        """Score a point in the quadtree based on how many points are in its regions"""

        if not self.boundary.contains(point):
            return False

        found_points = []
        point.anomaly_score += len(self.query(self.boundary, found_points)) * (4 ** self.depth)

        if self.divided:
            self.ne.score(point)
            self.nw.score(point)
            self.se.score(point)
            self.sw.score(point)

        return point.anomaly_score

    def score_depth(self, point):
        """Score a point in the quadtree based on how deep its leaf lies"""

        if not self.boundary.contains(point):
            return False

        if self.divided:
            self.ne.score(point)
            self.nw.score(point)
            self.se.score(point)
            self.sw.score(point)

        else:
            point.anomaly_score += self.depth

        return point.anomaly_score

    def query(self, boundary, found_points):
        """Find the points in the quadtree that lie within boundary."""

        if not self.boundary.intersects(boundary):
            # If the domain of this node does not intersect the search
            # region, we don't need to look in it for points.
            return False

        # Search this node's points to see if they lie within boundary ...
        for point in self.points:
            if boundary.contains(point):
                found_points.append(point)
        # ... and if this node has children, search them too.
        if self.divided:
            self.nw.query(boundary, found_points)
            self.ne.query(boundary, found_points)
            self.se.query(boundary, found_points)
            self.sw.query(boundary, found_points)
        return found_points

    def query_circle(self, boundary, centre, radius, found_points):
        """Find the points in the quadtree that lie within radius of centre.

        boundary is a Rect object (a square) that bounds the search circle.
        There is no need to call this method directly: use query_radius.

        """

        if not self.boundary.intersects(boundary):
            # If the domain of this node does not intersect the search
            # region, we don't need to look in it for points.
            return False

        # Search this node's points to see if they lie within boundary
        # and also lie within a circle of given radius around the centre point.
        for point in self.points:
            if (boundary.contains(point) and
                    point.distance_to(centre) <= radius):
                found_points.append(point)

        # Recurse the search into this node's children.
        if self.divided:
            self.nw.query_circle(boundary, centre, radius, found_points)
            self.ne.query_circle(boundary, centre, radius, found_points)
            self.se.query_circle(boundary, centre, radius, found_points)
            self.sw.query_circle(boundary, centre, radius, found_points)
        return found_points

    def query_radius(self, centre, radius, found_points):
        """Find the points in the quadtree that lie within radius of centre."""

        # First find the square that bounds the search circle as a Rect object.
        boundary = Rect(*centre, 2 * radius, 2 * radius)
        return self.query_circle(boundary, centre, radius, found_points)

    def __len__(self):
        """Return the number of points in the quadtree."""

        npoints = len(self.points)
        if self.divided:
            npoints += len(self.nw) + len(self.ne) + len(self.se) + len(self.sw)
        return npoints

    def points_in(self):
        return self.query(self.boundary, [])

    def draw(self, ax):
        """Draw a representation of the quadtree on Matplotlib Axes ax."""

        self.boundary.draw(ax)
        if self.divided:
            self.nw.draw(ax)
            self.ne.draw(ax)
            self.se.draw(ax)
            self.sw.draw(ax)
