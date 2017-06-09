import numpy as np
from math import sqrt, pi, cos


def squared_euclidean_distance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def euclidean_distance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return sqrt(dx * dx + dy * dy)


def closest_point(point, points):
    squared_distances = []
    for p in points:
        sd = squared_euclidean_distance(point, p)
        squared_distances.append(sd)
    i = np.argmin(squared_distances)
    return points[i]


# returns indices of the segments that contains the closest point(s)
def indices_closest_segments(point, segments):
    closest_points = []
    for segment in segments:
        cp = closest_point(point, segment)
        closest_points.append(cp)
    squared_distances = []
    for p in closest_points:
        ds = squared_euclidean_distance(point, p)
        squared_distances.append(ds)
    squared_distances = np.array(squared_distances)
    indices, = np.where(squared_distances == squared_distances.min())
    return list(indices)


def get_medoid(points):
    total_dists = {}
    for p1 in points:
        p1 = tuple(p1)
        total_dist = 0
        for p2 in points:
            p2 = tuple(p2)
            total_dist += euclidean_distance(p1, p2)
        total_dists[p1] = total_dist
    return list(min(total_dists, key=total_dists.get))


def is_equal(points1, points2):
    if len(points1) != len(points2):
        return False
    else:
        for i in range(len(points1)):
            x1, y1 = points1[i]
            x2, y2 = points2[i]
            if x1 != x2 or y1 != y2:
                return False
        return True


def meter_per_pixel(lat, zoom):
    # equatorial circumference of the Earth = 40,075 kilometres
    return 40075 * (10 ** 3) * cos(lat * pi / 180) / (2 ** (9 + zoom))
