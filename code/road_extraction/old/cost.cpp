#include <iostream>
#include <math.h>
using namespace std;

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

double** get_cost_map(double** img[3], double seed_colors[][3], double inv_cov_matrix[3][3]) {
    int rows = sizeof(img);
    int cols = sizeof(img[0]);
    int seeds = sizeof(seed_colors);
    double** cost_map = new double*[rows];
    double m = 0.000001;

    for (int r = 0; r < rows; r++) {
        cost_map[r] = new double[cols];

        for (int c = 0; c < cols; c++) {
            double cost = 0;
            double color[3] = {img[r][c][0], img[r][c][1], img[r][c][2]};

            for (int s = 0; s < seeds; s++) {
                cout << r;
                cout << c;
                cout << s << endl;
                double seed_color[3] = {seed_colors[s][0], seed_colors[s][1], seed_colors[s][2]};
                cost += mahalanobis_distance(color, seed_color, inv_cov_matrix);
            }

            cost_map[r][c] = cost;

            if (cost > m) {
                m = cost;
            }
        }
    }

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            cost_map[r][c] /= m;
        }
    }

    return cost_map;
}

void main() {
    const int s = 1024;
    double*** img = new double**[s];

    for (int i = 0; i < s; i++) {
        img[i] = new double*[s];

        for (int j = 0; j < s; j++) {
            img[i][j] = new double[3];

            for (int k = 0; k < 3; k++) {
                img[i][j][k] = 0.5;
            }
        }
    }

    double seeds[4][3] = {{0.35, 0.39, 0.39},
                          {0.35, 0.37, 0.35},
                          {0.36, 0.38, 0.37},
                          {0.36, 0.39, 0.38}};
    double icm[3][3] = {{4597.58306945, -3800.90557869, -682.39548799},
                        {-3800.90557869, 6061.35074138, -2395.66081968},
                        {-682.39548799, -2395.66081968, 3183.94857435}};

    double** cost_map = get_cost_map(img, seeds, icm);
    cout << cost_map[0][0] << endl;
}