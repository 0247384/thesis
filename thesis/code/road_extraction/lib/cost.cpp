#define DLLEXPORT extern "C" __declspec(dllexport)
#include <cmath>

double euclidean_distance(double a[3], double b[3]) {
    double dx = a[0] - b[0];
    double dy = a[1] - b[1];
    double dz = a[2] - b[2];
    return sqrt(dx * dx + dy * dy + dz * dz);
}

double manhattan_distance(double a[3], double b[3]) {
    double dx = a[0] - b[0];
    double dy = a[1] - b[1];
    double dz = a[2] - b[2];

    if (dx < 0) {
        dx = -dx;
    }
    if (dy < 0) {
        dy = -dy;
    }
    if (dz < 0) {
        dz = -dz;
    }

    return dx + dy + dz;
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

DLLEXPORT void compute_cost_map(double * cost_map, double * img, double seed_colors[][3], double inv_cov_matrix[3][3], int rows, int cols, int seeds) {
    double max = 1e-9;

    for (int r = 0; r < rows; r++) {
        int x = r * cols;

        for (int c = 0; c < cols; c++) {
            int i = x * 3 + c * 3;
            double color[3] = {img[i], img[i + 1], img[i + 2]};
            double cost = 0;

            for (int s = 0; s < seeds; s++) {
                double seed_color[3] = {seed_colors[s][0], seed_colors[s][1], seed_colors[s][2]};
                cost += mahalanobis_distance(color, seed_color, inv_cov_matrix);
            }

            cost_map[x + c] = cost;

            if (cost > max) {
                max = cost;
            }
        }
    }

    for (int i = 0; i < rows * cols; i++) {
        cost_map[i] /= max;
    }
}
