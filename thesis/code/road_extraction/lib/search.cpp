#define DLLEXPORT extern "C" __declspec(dllexport)
#include <cmath>
#include <limits>
#include <vector>
#include <queue>

double const sqrt2d2 = sqrt(2) / 2;

struct XY
{
    int x;
    int y;

    XY() : x(-1), y(-1) {}

    XY(int arg1, int arg2) : x(arg1), y(arg2) {}
};

struct XYD
{
    int x;
    int y;
    double d;

    XYD() : x(-1), y(-1), d(-1) {}

    XYD(int arg1, int arg2, double arg3) : x(arg1), y(arg2), d(arg3) {}
};

struct CXY
{
    double cost;
    int x;
    int y;

    CXY(double arg1, int arg2, int arg3) : cost(arg1), x(arg2), y(arg3) {}

    bool operator<(const struct CXY& other) const
    {
        return cost > other.cost;
    }
};

void compute_neighbors(std::vector<XYD> & neighbors, int x, int y, int max_x, int max_y) {
    neighbors.clear();

    if (x > 0) {
        neighbors.push_back(XYD(x - 1, y, 0.5));

        if (y > 0) {
            neighbors.push_back(XYD(x - 1, y - 1, sqrt2d2));
        }

        if (y < max_y) {
            neighbors.push_back(XYD(x - 1, y + 1, sqrt2d2));
        }
    }

    if (y > 0) {
        neighbors.push_back(XYD(x, y - 1, 0.5));

        if (x < max_x) {
            neighbors.push_back(XYD(x + 1, y - 1, sqrt2d2));
        }
    }

    if (x < max_x) {
        neighbors.push_back(XYD(x + 1, y, 0.5));

        if (y < max_y) {
            neighbors.push_back(XYD(x + 1, y + 1, sqrt2d2));
        }
    }

    if (y < max_y) {
        neighbors.push_back(XYD(x, y + 1, 0.5));
    }
}

double mahalanobis_distance(double a[3], double b[3], double inv_cov_matrix[3][3]) {
    double d[3];
    double t[3] = {0, 0, 0};
    double r = 0;

    d[0] = a[0] - b[0];
    d[1] = a[1] - b[1];
    d[2] = a[2] - b[2];

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            t[i] += d[j] * inv_cov_matrix[i][j];
        }
    }

    for (int i = 0; i < 3; i++) {
        r += t[i] * d[i];
    }

    return sqrt(r);
}

DLLEXPORT int compute_shortest_path(int * shortest_path, double * cost_map, int x_start, int y_start, int x_goal, int y_goal, int rows, int cols, double * img, double inv_cov_matrix[3][3]) {
    std::vector<XY> shortest_path_vector;
    std::priority_queue<CXY> pq;
    std::vector<XYD> neighbors;
    XY * came_from = new XY[rows * cols];
    double * cost_to = new double[rows * cols];
    int * visited = new int[rows * cols];
    double inf = std::numeric_limits<double>::infinity();
    double min_cost = inf;
    int max_x = cols - 1;
    int max_y = rows - 1;

    for (int i = 0; i < rows * cols; i++) {
        cost_to[i] = inf;
        double cost = cost_map[i];
        if (cost < min_cost) {
            min_cost = cost;
        }
    }

    cost_to[y_start * cols + x_start] = 0;
    pq.push(CXY(cost_map[y_start * cols + x_start], x_start, y_start));

    while (!pq.empty()) {
        CXY current = pq.top();
        pq.pop();
        int x = current.x;
        int y = current.y;
        int ci = y * cols + x;

        if (visited[ci] == 1) {
            continue;
        } else {
            visited[ci] = 1;
        }

        if (x == x_goal && y == y_goal) {
            break;
        }

        compute_neighbors(neighbors, x, y, max_x, max_y);

        for (XYD n : neighbors) {
            int ni = n.y * cols + n.x;
            // // double d = abs(cost_map[ci] - cost_map[ni]); // local homogeneity feature
            // int i = y * cols * 3 + x * 3;
            // double cc[3] = {img[i], img[i + 1], img[i + 2]};
            // i = n.y * cols * 3 + n.x * 3;
            // double cn[3] = {img[i], img[i + 1], img[i + 2]};
            // double d = mahalanobis_distance(cc, cn, inv_cov_matrix);
            // n.d == sqrt(2)/2 if diagonal, 1/2 otherwise
            double new_cost = cost_to[ci] + n.d * cost_map[ci] + n.d * cost_map[ni]; // + d / 9;

            if (new_cost < cost_to[ni]) {
                cost_to[ni] = new_cost;
                came_from[ni] = XY(x,y);
                double dx = x_goal - n.x;
                double dy = y_goal - n.y;
                double eucl_dist = sqrt(dx * dx + dy * dy);
                double estimated_cost = new_cost + eucl_dist * min_cost;
                pq.push(CXY(estimated_cost, n.x, n.y));
            }
        }
    }

    int x = x_goal;
    int y = y_goal;
    XY xy = XY(x_goal, y_goal);
    shortest_path_vector.insert(shortest_path_vector.begin(), xy);

    while (x != x_start || y != y_start) {
        int i = y * cols + x;
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
    delete visited;

    return shortest_path_vector.size();
}
