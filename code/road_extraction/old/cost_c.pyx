import cython
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double euclidean_distance(double[:] a, double[:] b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double manhattan_distance(double[:] a, double[:] b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double mahalanobis_distance(double[:] a, double[:] b, double[:, :] inv_cov_matrix):
    cdef double d[3]
    cdef double t[3]
    cdef double r = 0
    cdef int i, j

    d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    t = [0, 0, 0]

    for i in range(3):
        for j in range(3):
            t[i] += d[j] * inv_cov_matrix[i][j]

    for i in range(3):
        r += t[i] * d[i]

    return sqrt(r)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def get_cost_map(double[:, :, :] img, double[:, :] seed_colors, double[:, :] inv_cov_matrix):
    cdef int rows = img.shape[0]
    cdef int cols = img.shape[1]
    cdef int seeds = seed_colors.shape[0]
    cdef double[:, :] cost_map = np.empty((rows, cols))
    cdef double[:] color, seed_color
    cdef double cost, m = 0.000001
    cdef int r, c, s

    for r in range(rows):
        for c in range(cols):
            cost = 0
            color = img[r][c]

            for s in range(seeds):
                seed_color = seed_colors[s]
                cost += mahalanobis_distance(color, seed_color, inv_cov_matrix)

            cost_map[r][c] = cost
            if cost > m:
                m = cost

    return np.array(cost_map) / m
