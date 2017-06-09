import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from skimage.draw import circle
from skimage.filters import gaussian
from rdp import rdp
from scipy.interpolate import splprep, splev
from statsmodels.nonparametric.smoothers_lowess import lowess
from utils.line import fill
from utils.distance import get_medoid, is_equal
from road_extraction.cost import get_cost_path
from road_extraction.search import get_shortest_path

road_width = 9


def straight_line_check(path, cost_map, max_incr_ratio):
    start = path[0]
    goal = path[-1]

    straight_line = fill([], start, goal)
    cost_straight_line = get_cost_path(straight_line, cost_map)
    cost_path = get_cost_path(path, cost_map)

    if cost_straight_line <= max_incr_ratio * cost_path:
        return straight_line
    else:
        return False


def simplify_path(path, e=1, tp_indices=None):
    start = path[0]
    end = path[-1]

    if tp_indices is None:
        points = rdp(path, epsilon=e)
        points = [(x[0], x[1]) for x in points]
        simplified_path = fill(points, start, end)
    else:
        indices = tp_indices + [len(path) - 1]
        points = [start]
        n = 0

        for i in indices:
            points_segment = rdp(path[n:i + 1], epsilon=e)
            points_segment = [(x[0], x[1]) for x in points_segment]
            points.extend(points_segment[1:])
            n = i

        simplified_path = fill(points, start, end)

        del tp_indices[:]
        for i in indices:
            tp_indices.append(simplified_path.index(path[i]))

        if len(tp_indices) > 0:
            del tp_indices[-1]

    return simplified_path, points


def smooth_using_b_spline(path, sc=100):
    degree = 3

    if len(path) <= degree:
        return path

    x = [p[0] for p in path]
    y = [p[1] for p in path]

    tck, u = splprep([x, y], s=sc, k=degree)
    s = splev(u, tck)

    xs = [int(round(x)) for x in s[0]]
    ys = [int(round(y)) for y in s[1]]
    smoothed_path = list(zip(xs, ys))

    margin = min(3 * road_width, int(len(path) / 3))
    smoothed_path = smoothed_path[margin:-margin]
    smoothed_path = fill(smoothed_path, path[0], path[-1])
    return smoothed_path


def smooth_using_moving_centroid(path, s=5, m=15):
    step = s
    margin = m
    centroids = []

    for i in range(margin):
        path.insert(0, path[0])
        path.append(path[-1])

    for i in range(0, len(path), step):
        segment = path[max(0, i - margin):i + margin]
        m = np.mean(segment, axis=0)
        centroid = (int(round(m[0])), int(round(m[1])))
        centroids.append(centroid)

    smoothed_path = fill(centroids, path[0], path[-1])
    return smoothed_path


def smooth_using_moving_medoid(path, s=5, m=15):
    step = s
    margin = m
    medoids = []

    for i in range(margin):
        path.insert(0, path[0])
        path.append(path[-1])

    for i in range(0, len(path), step):
        segment = path[max(0, i - margin):i + margin]
        medoid = get_medoid(segment)
        medoids.append(medoid)

    smoothed_path = fill(medoids, path[0], path[-1])
    return smoothed_path


def smooth_segment_using_lowess(segment):
    dx = abs(segment[0][0] - segment[-1][0])
    dy = abs(segment[0][1] - segment[-1][1])

    x = [x for (x, y) in segment]
    y = [y for (x, y) in segment]

    frac = max(min(50 / len(segment), 0.5), 0.025)

    if dy > dx:
        l = lowess(x, y, frac, it=10, return_sorted=False)
        xs = [int(round(x)) for x in l]
        smoothed_segment = list(zip(xs, y))
    else:
        l = lowess(y, x, frac, it=10, return_sorted=False)
        ys = [int(round(y)) for y in l]
        smoothed_segment = list(zip(x, ys))

    return smoothed_segment


def smooth_using_segmented_lowess(path):
    points = simplify_path(path, e=27)[1]

    indices = []
    for p in points:
        indices.append(path.index(p))
    indices = indices[1:-1]

    segments = []
    n = 0
    for i in indices:
        segment = path[n:i]
        segments.append(segment)
        n = i
    segment = path[n:]
    segments.append(segment)

    smoothed_path = []
    for segment in segments:
        smoothed_segment = smooth_segment_using_lowess(segment)
        smoothed_path.extend(smoothed_segment)

    return smoothed_path


def smooth_using_search(path, cost_map):
    start = path[0]
    end = path[-1]
    cost_map_limited = np.array(cost_map)
    in_buffer = np.full(cost_map_limited.shape, False, dtype='bool')
    not_in_buffer = np.full(cost_map_limited.shape, True, dtype='bool')

    buffer_width = 1.5
    for x, y in path:
        rr, cc = circle(y, x, buffer_width, cost_map_limited.shape[0:2])
        for r, c in zip(rr, cc):
            in_buffer[r][c] = True
            not_in_buffer[r][c] = False

    max_value = max(cost_map_limited[in_buffer])
    cost_map_limited[not_in_buffer] = max_value
    cost_map_limited = gaussian(cost_map_limited, sigma=road_width)
    # cost_map_limited[not_in_buffer] = max_value
    cost_map_limited[not_in_buffer] = 1e9  # INF

    if False:
        fig, ax = plt.subplots(1, 2)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        ax[0].axis('off')
        ax[0].imshow(cost_map, cmap='jet', norm=colors.LogNorm())
        ax[1].axis('off')
        ax[1].imshow(cost_map_limited, cmap='jet', norm=colors.LogNorm())
        plt.show()

    smoothed_path = get_shortest_path(cost_map_limited, start, end)
    return smoothed_path


# updates tp_indices after smoothing if len(list) > 0
def smooth_iteratively(path, cost_map, max_incr_ratio, tp_indices=[], smoothing_mandatory=False):
    if len(tp_indices) > 0:
        indices = tp_indices + [len(path) - 1]
        del tp_indices[:]
        smoothed_path = [path[0]]
        n = 0

        for i in indices:
            segment = path[n:i + 1]
            straight_line = straight_line_check(segment, cost_map, max_incr_ratio)
            if straight_line:
                smoothed_segment = straight_line[1:]
            else:
                smoothed_segment = smooth_iteratively(segment, cost_map, max_incr_ratio)[1:]
            smoothed_path.extend(smoothed_segment)
            tp_indices.append(len(smoothed_path) - 1)
            n = i

        if len(tp_indices) > 0:
            del tp_indices[-1]

        return smoothed_path
    else:
        # s = 100
        margin = 15
        cost_path = get_cost_path(path, cost_map)
        prev = path
        # smoothed = smooth_using_b_spline(path, s)
        smoothed = smooth_using_moving_centroid(path, m=margin)

        i = 0
        never_changed = True
        changed = True
        while i < 15 and (never_changed or changed) and get_cost_path(smoothed, cost_map) < max_incr_ratio * cost_path:
            i += 1
            # s *= 2
            margin += 5
            prev = smoothed
            # smoothed = smooth_using_b_spline(smoothed, s)
            smoothed = smooth_using_moving_centroid(smoothed, m=margin)
            changed = not is_equal(prev, smoothed)
            if never_changed and changed:
                never_changed = False

        if smoothing_mandatory and i == 0:
            return smoothed
        else:
            return prev
