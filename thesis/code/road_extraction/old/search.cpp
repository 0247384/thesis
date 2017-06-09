#define DLLEXPORT extern "C" __declspec(dllexport)
#include <limits>
#include <vector>
#include <queue>

struct XY
{
    int x;
    int y;

    XY() : x(-1), y(-1) {}

    XY(int arg1, int arg2) : x(arg1), y(arg2) {}
};

struct S
{
    double cost;
    int x;
    int y;

    S(double arg1, int arg2, int arg3) : cost(arg1), x(arg2), y(arg3) {}

    bool operator<(const struct S& other) const
    {
        return cost > other.cost;
    }
};

void compute_neighbors(std::vector<XY> & neighbors, int x, int y, int max_x, int max_y) {
    neighbors.clear();

    if (x > 0) {
        neighbors.push_back(XY(x - 1, y));

        if (y > 0) {
            neighbors.push_back(XY(x - 1, y - 1));
        }

        if (y < max_y) {
            neighbors.push_back(XY(x - 1, y + 1));
        }
    }

    if (y > 0) {
        neighbors.push_back(XY(x, y - 1));

        if (x < max_x) {
            neighbors.push_back(XY(x + 1, y - 1));
        }
    }

    if (x < max_x) {
        neighbors.push_back(XY(x + 1, y));

        if (y < max_y) {
            neighbors.push_back(XY(x + 1, y + 1));
        }
    }

    if (y < max_y) {
        neighbors.push_back(XY(x, y + 1));
    }
}

DLLEXPORT int compute_shortest_path(int * shortest_path, double * cost_map, int x_start, int y_start, int x_goal, int y_goal, int rows, int cols) {
    std::vector<XY> shortest_path_vector;
    std::priority_queue<S> pq;
    std::vector<XY> neighbors;
    XY * came_from = new XY[rows * cols];
    double * cost_to = new double[rows * cols];
    double inf = std::numeric_limits<double>::infinity();
    double n_cost;
    int max_x = cols - 1;
    int max_y = rows - 1;
    int x, y, nx, ny, i;
    XY xy;

    for (int i = 0; i < rows * cols; i++) {
        cost_to[i] = inf;
    }

    cost_to[y_start * cols + x_start] = 0;
    pq.push(S(cost_map[y_start * cols + x_start], x_start, y_start));

    while (!pq.empty()) {
        S current = pq.top();
        pq.pop();
        x = current.x;
        y = current.y;

        if (x == x_goal && y == y_goal) {
            break;
        }

        compute_neighbors(neighbors, x, y, max_x, max_y);

        for (XY n : neighbors) {
            nx = n.x;
            ny = n.y;
            i = ny * cols + nx;

            if (cost_to[i] == inf) {
                n_cost = current.cost + cost_map[i];
                cost_to[i] = n_cost;
                pq.push(S(n_cost, nx, ny));
                came_from[i] = XY(x,y);
            }
        }
    }

    x = x_goal;
    y = y_goal;
    xy = XY(x_goal, y_goal);
    shortest_path_vector.insert(shortest_path_vector.begin(), xy);

    while (x != x_start || y != y_start) {
        i = y * cols + x;
        xy = came_from[i];
        shortest_path_vector.insert(shortest_path_vector.begin(), xy);
        x = xy.x;
        y = xy.y;
    }

    for (int i = 0, j = 0; i < shortest_path_vector.size(); i++) {
        xy = shortest_path_vector.at(i);
        shortest_path[j++] = xy.x;
        shortest_path[j++] = xy.y;
    }

    delete came_from;
    delete cost_to;

    return shortest_path_vector.size();
}
