import cython
import heapq


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef list get_neighbors(int x, int y, int max_x, int max_y):
    cdef list neighbors = []

    if x > 0:
        neighbors.append((x - 1, y))
        if y > 0:
            neighbors.append((x - 1, y - 1))
        if y < max_y:
            neighbors.append((x - 1, y + 1))

    if y > 0:
        neighbors.append((x, y - 1))
        if x < max_x:
            neighbors.append((x + 1, y - 1))

    if x < max_x:
        neighbors.append((x + 1, y))
        if y < max_y:
            neighbors.append((x + 1, y + 1))

    if y < max_y:
        neighbors.append((x, y + 1))

    return neighbors


def shortest_path(double[:, :] cost_map, tuple start, tuple goal):
    queue = []
    heapq.heappush(queue, (0, start))
    came_from = {start: None}
    cost_to = {start: 0}

    cdef int rows = cost_map.shape[0]
    cdef int cols = cost_map.shape[1]
    cdef int max_x = cols - 1
    cdef int max_y = rows - 1
    cdef tuple current, neighbor
    cdef list neighbors
    cdef double cost_current

    while len(queue) > 0:
        cost_current, current = heapq.heappop(queue)
        if current == goal:
            break

        neighbors = get_neighbors(current[0], current[1], max_x, max_y)
        for neighbor in neighbors:
            if neighbor not in cost_to:
                came_from[neighbor] = current
                pixel_cost = cost_map[neighbor[1]][neighbor[0]]
                neighbor_cost = cost_current + pixel_cost
                cost_to[neighbor] = neighbor_cost
                heapq.heappush(queue, (neighbor_cost, neighbor))

    return came_from, cost_to


def shortest_path_old(double[:, :] cost_map, tuple start, tuple goal):
    queue = []
    heapq.heappush(queue, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    cdef int rows = cost_map.shape[0]
    cdef int cols = cost_map.shape[1]
    cdef int max_x = cols - 1
    cdef int max_y = rows - 1
    cdef tuple current, previous, neighbor
    cdef list neighbors
    cdef set neighbors_previous

    while len(queue) > 0:
        cost_current, current = heapq.heappop(queue)
        if current == goal:
            break

        previous = came_from[current]
        neighbors = get_neighbors(current[0], current[1], max_x, max_y)

        if previous is not None:
            neighbors.remove(previous)
            neighbors_previous = set(get_neighbors(previous[0], previous[1], max_x, max_y))
            neighbors = [x for x in neighbors if x not in neighbors_previous]

        for neighbor in neighbors:
            pixel_cost = cost_map[neighbor[1]][neighbor[0]]
            neighbor_cost = cost_current + pixel_cost

            if neighbor not in cost_so_far or neighbor_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = neighbor_cost
                came_from[neighbor] = current
                priority = neighbor_cost
                heapq.heappush(queue, (priority, neighbor))

    return came_from, cost_so_far
