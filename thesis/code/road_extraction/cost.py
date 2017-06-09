import os
import numpy as np
import ctypes as ct
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from skimage.filters import gaussian
from utils.distance import euclidean_distance

# import road_extraction.cost_c as cost_c # cython implementation

language = 'cpp'  # cpp, numpy or python

if language == 'cpp':
    path = os.path.dirname(__file__)
    dll = ct.cdll.LoadLibrary(path + '\lib\cost.dll')

    int = ct.c_int
    double = ct.c_double
    pdouble = ct.POINTER(double)

    compute_cost_map = dll.compute_cost_map
    compute_cost_map.argtypes = (pdouble, pdouble, pdouble, pdouble, int, int, int)
    compute_cost_map.restype = None


def mahalanobis_distance(c1, c2, icm):
    d = c1 - c2
    m = np.dot(np.dot(d, icm), d)
    return np.sqrt(m)


def color_cost(color, seed_colors, icm):
    return sum([mahalanobis_distance(color, seed_color, icm) for seed_color in seed_colors])


def get_cost_map(img, seed_colors, inv_cov_matrix):
    if language == 'cpp':
        img = np.array(img)
        seed_colors = np.array(seed_colors)
        inv_cov_matrix = np.array(inv_cov_matrix)

        rows, cols = img.shape[:2]
        seeds = seed_colors.shape[0]
        cost_map = np.zeros((rows, cols))

        c_cost_map = cost_map.ctypes.data_as(pdouble)
        c_img = img.ctypes.data_as(pdouble)
        c_seed_colors = seed_colors.ctypes.data_as(pdouble)
        c_icm = inv_cov_matrix.ctypes.data_as(pdouble)

        # compute_cost_map fills cost_map
        compute_cost_map(c_cost_map, c_img, c_seed_colors, c_icm, rows, cols, seeds)

    elif language == 'numpy':
        cost_map = np.zeros((len(img), len(img[0])))
        np_img = np.array(img)
        for seed in seed_colors:
            cost_map += np.sqrt(np.einsum('ijk,ijk->ij', np.dot(np_img - seed, inv_cov_matrix), np_img - seed))

    # elif language == 'cython':
    #     cost_map = cost_c.get_cost_map(img, np.array(seed_colors), inv_cov_matrix)

    else:
        cost_map = np.array([[color_cost(color, seed_colors, inv_cov_matrix) for color in row] for row in img])

    cost_map = gaussian(cost_map, sigma=1.5)

    return cost_map / cost_map.max()


def get_cost_pixels(pixels, cost_map):
    cost = 0
    for x, y in pixels:
        cost += cost_map[y][x]
    return cost


def get_cost_path(path, cost_map):
    if len(path) == 0:
        return 0
    elif len(path) == 1:
        x, y = path[0]
        return cost_map[y][x]
    else:
        cost = 0
        previous = None
        for current in path:
            if previous is not None:
                dist_db2 = euclidean_distance(current, previous) / 2
                x, y = current
                x = max(0, x)
                x = min(x, len(cost_map[0]) - 1)
                y = max(0, y)
                y = min(y, len(cost_map) - 1)
                current = x, y
                px, py = previous
                cc = cost_map[y][x]
                cp = cost_map[py][px]
                cost += dist_db2 * cc + dist_db2 * cp
            previous = current
        return cost
