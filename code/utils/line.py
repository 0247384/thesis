import numpy as np
from scipy.odr import odr
from utils.distance import euclidean_distance
from scipy.stats import linregress
import matplotlib.pyplot as plt


# Bresenham's line algorithm, source: stackoverflow.com/a/29402598
def line_to_pixels(p1, p2):
    x0, y0 = p1
    x1, y1 = p2

    pixels = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1

    if dx > dy:
        err = dx / 2.0
        while x != x1:
            pixels.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            pixels.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

    pixels.append((x, y))
    return pixels


def fill(points, start, end):
    if len(points) == 0:
        points = [start, end]
    if len(points) == 1:
        if points[0] == start or points[0] == end:
            points = [start, end]
        else:
            points = [start, points[0], end]

    result = []
    if points[0] != start:
        l = line_to_pixels(start, points[0])
        result.extend(l)
    for x in range(0, len(points) - 1):
        if len(result) > 0:
            del result[-1]
        l = line_to_pixels(points[x], points[x + 1])
        result.extend(l)
    if result[-1] != end:
        l = line_to_pixels(result[-1], end)
        del result[-1]
        result.extend(l)
    return result


# Source: docs.scipy.org/doc/scipy/reference/odr.html
def linear(b, x):
    return b[0] * x + b[1]


def fit_line(points):
    if len(points) < 2:
        return False
    elif len(points) == 2:
        return Line(points[0], points[1])
    else:
        x = [p[0] for p in points]
        y = [p[1] for p in points]

        switched = False
        if abs(x[0] - x[-1]) < abs(y[0] - y[-1]):
            switched = True
            x_temp = list(x)
            x = list(y)
            y = list(x_temp)
            # print('Switched axes')

        odr_result = odr(linear, [0, 0], y, x, full_output=1)
        m, b = odr_result[0]
        m = round(m, 6)
        b = round(b, 6)
        f = np.poly1d((m, b))

        # if switched:
        #     print('ODR: x = {}y + {}'.format(m, b))
        # else:
        #     print('ODR: y = {}x + {}'.format(m, b))
        #
        # stop_cond = odr_result[3]['info']
        # if stop_cond < 1 or stop_cond > 3:
        #     print('Warning: ODR did not converge to a solution (stop condition {})'.format(stop_cond))
        #
        # ols_result = linregress(x, y)
        # m, b = ols_result[0:2]
        # m = round(m, 6)
        # b = round(b, 6)
        # f_ols = np.poly1d((m, b))
        #
        # if switched:
        #     print('OLS: x = {}y + {}'.format(m, b))
        # else:
        #     print('OLS: y = {}x + {}'.format(m, b))

        x1 = x[0]
        x2 = x[-1]

        if switched:
            p1 = round(f(x1)), x1
            p2 = round(f(x2)), x2
            line = Line(p1, p2)
            # pixels = line.pixels()
            # x_temp = list(x)
            # x = list(y)
            # y = list(x_temp)
        else:
            p1 = x1, round(f(x1))
            p2 = x2, round(f(x2))
            line = Line(p1, p2)
            # pixels = line.pixels()

        # x_l = [p[0] for p in pixels]
        # y_l = [p[1] for p in pixels]
        #
        # t_x = (min(x) - 1, max(x) + 1)
        # t_y = (min(y) - 1, max(y) + 1)
        # t = range(t_x[0], t_x[1] + 1, 1)
        #
        # plt.figure()
        # ax = plt.gca()
        # ax.plot(x, y, 'g', linewidth=3)
        # ax.plot(x_l, y_l, 'b', linewidth=3)
        # ax.plot(t, f(t), 'c--', linewidth=3)
        # ax.plot(t, f_ols(t), 'r--', linewidth=3)
        # ax.set_xlim(t_x)
        # ax.set_ylim(t_y)
        # plt.show()

        return line


# Source: stackoverflow.com/questions/21030391/how-to-normalize-array-numpy
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


class Line:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def pixels(self):
        return line_to_pixels(self.p1, self.p2)

    def vector(self):
        v1 = np.array(self.p1)
        v2 = np.array(self.p2)
        return v2 - v1

    def angle(self, l):
        v1 = self.vector()
        v1 = normalize(v1)
        v2 = -l.vector()
        v2 = normalize(v2)
        dot = np.dot(v1, v2)
        return np.arccos(round(dot, 6))

    def point_on_max_dist_from_p1(self, max_dist):
        points = self.pixels()
        for point in points:
            dist = euclidean_distance(self.p1, point)
            if dist > max_dist:
                break
            previous_point = point
        return previous_point

    # en.wikipedia.org/wiki/Line-line_intersection#Given_two_points_on_each_line
    def intersection(self, l):
        if self.p1 == l.p1 or self.p1 == l.p2:
            return self.p1
        elif self.p2 == l.p1 or self.p2 == l.p2:
            return self.p2

        x1, y1 = self.p1
        x2, y2 = self.p2
        x3, y3 = l.p1
        x4, y4 = l.p2

        d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if d == 0:
            return False

        x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
        y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d

        return x, y
