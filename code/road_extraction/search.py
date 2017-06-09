import os
import numpy as np
import ctypes as ct
import matplotlib.pyplot as plt
from matplotlib import colors
from utils.priority_queue import PriorityQueue

# import road_extraction.search_c as c # cython implementation

language = 'cpp'

if language == 'cpp':
    path = os.path.dirname(__file__)
    dll = ct.cdll.LoadLibrary(path + '\lib\search.dll')

    int = ct.c_int
    pint = ct.POINTER(int)
    double = ct.c_double
    pdouble = ct.POINTER(double)

    compute_shortest_path = dll.compute_shortest_path
    compute_shortest_path.argtypes = (pint, pdouble, int, int, int, int, int, int, pdouble, pdouble)
    compute_shortest_path.restype = int


def get_shortest_path(cost_map, start, goal, img, icm):
    if language == 'cpp':
        rows = len(cost_map)
        cols = len(cost_map[0])
        shortest_path_array = np.zeros((rows * cols * 2), dtype='i')

        c_shortest_path = shortest_path_array.ctypes.data_as(pint)
        c_cost_map = cost_map.ctypes.data_as(pdouble)
        c_img = img.ctypes.data_as(pdouble)
        c_icm = icm.ctypes.data_as(pdouble)

        # compute_shortest_path fills shortest_path_array
        size = compute_shortest_path(c_shortest_path, c_cost_map, start[0], start[1], goal[0], goal[1], rows, cols,
                                     c_img, c_icm)

        shortest_path = []
        for i in range(0, size * 2, 2):
            x = shortest_path_array[i]
            y = shortest_path_array[i + 1]
            shortest_path.append((x, y))

    # elif language == 'cython':
    #     came_from, cost_to = c.shortest_path(cost_map, start, goal)
    #
    #     if False:
    #         max_value = max(cost_to.values())
    #         search_img = np.full(cost_map.shape, max_value + 1)
    #         for x, y in cost_to:
    #             search_img[y][x] = cost_to[(x, y)]
    #
    #         fig, ax = plt.subplots(1, 2)
    #         plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    #         ax[0].axis('off')
    #         ax[0].imshow(cost_map[:-30], cmap='jet', norm=colors.LogNorm())
    #         ax[1].axis('off')
    #         ax[1].imshow(search_img, interpolation='kaiser')
    #         plt.show()
    #
    #     shortest_path = [goal]
    #     pixel = goal
    #     while pixel is not start:
    #         pixel = came_from[pixel]
    #         shortest_path.append(pixel)
    #     shortest_path = shortest_path[::-1]

    return shortest_path

# # uses image object
# def get_shortest_path_py(image, cost_map, start, goal):
#     queue = PriorityQueue()
#     queue.put(start, 0)
#     came_from = {start: None}
#     cost_to = {start: 0}
#
#     while not queue.is_empty():
#         current = queue.get()
#         if current == goal:
#             break
#
#         for neighbor in image.get_neighbors(current):
#             if neighbor not in cost_to:
#                 pixel_cost = cost_map[neighbor[1]][neighbor[0]]
#                 neighbor_cost = cost_to[current] + pixel_cost
#                 cost_to[neighbor] = neighbor_cost
#                 came_from[neighbor] = current
#                 queue.put(neighbor, neighbor_cost)
#
#     return came_from, cost_to


# def shortest_path_old(image, start, goal, cost_map=None):
#     queue = PriorityQueue()
#     queue.put(start, 0)
#     came_from = {start: None}
#     cost_so_far = {start: 0}
#     # Enforce shape:
#     direction_changes = {start: 0}
#     pixels_equal_direction = {start: 0}
#
#     color_start = image.get()[start[1]][start[0]]
#     color_goal = image.get()[goal[1]][goal[0]]
#
#     while not queue.is_empty():
#         current = queue.get()
#         if current == goal:
#             break
#
#         previous = came_from[current]
#         neighbors = image.get_neighbors(current)
#
#         if previous is not None:
#             neighbors.remove(previous)
#             neighbors_previous = set(image.get_neighbors(previous))
#             neighbors = [x for x in neighbors if x not in neighbors_previous]
#
#         for neighbor in neighbors:
#             # if cost_map is not None:
#             #     pixel_cost = cost_map[pixel[1]][pixel[0]]
#             # else:
#             #     color = image.get()[pixel[1]][pixel[0]]
#             #     pixel_cost = color_cost(color, [color_start, color_goal])
#             pixel_cost = cost_map[neighbor[1]][neighbor[0]]
#             neighbor_cost = cost_so_far[current] + pixel_cost
#
#             if neighbor not in cost_so_far or neighbor_cost < cost_so_far[neighbor]:
#                 cost_so_far[neighbor] = neighbor_cost
#                 came_from[neighbor] = current
#                 priority = neighbor_cost
#                 queue.put(neighbor, priority)
#
#     return came_from, cost_so_far
