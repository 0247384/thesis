import os
import gc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt
from scipy import linalg
from skimage import img_as_float
from skimage.io import imread
from skimage.color import rgb2gray, rgb2hsv, rgb2lab, rgb2luv, rgb2ycbcr, rgb2yiq, rgb2ypbpr, rgb2yuv
from skimage.draw import circle
from data.road_collector import get_roads_from_xml_file
from road_extraction.cost import mahalanobis_distance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# matplotlib.rcParams.update({'font.size': 16})
# matplotlib.rcParams.update({'axes.labelpad': 0})

path_road_data = 'C:/Users/V/Documents/Informatica/Thesis/Data/Roads/Leuven.xml'
path_images = 'C:/Users/V/Documents/Informatica/Thesis/Data/Images/Leuven - Bing/'
buffer_width = 0.5


def get_cluster_distances(cluster1, cluster2, title=None):
    c1_distances = []
    c1_mean_dist = 0
    c1_mean = np.mean(cluster1, axis=0)
    for i in range(len(cluster1)):
        c1 = cluster1[i]
        dist = 0
        for j in range(len(c1)):
            # dist += abs(c1_mean[j] - c1[j])
            dist += (c1_mean[j] - c1[j]) ** 2
        dist = sqrt(dist)
        c1_mean_dist += dist
        c1_distances.append(dist)
    c1_mean_dist /= len(cluster1)
    print('Mean distance from c1 to mean c1:', c1_mean_dist)

    c1_c2_distances = []
    c1_c2_mean_dist = 0
    for i in range(len(cluster2)):
        c2 = cluster2[i]
        dist = 0
        for j in range(len(c2)):
            # dist += abs(c1_mean[j] - c2[j])
            dist += (c1_mean[j] - c2[j]) ** 2
        dist = sqrt(dist)
        c1_c2_mean_dist += dist
        c1_c2_distances.append(dist)
    c1_c2_mean_dist /= len(cluster2)
    print('Mean distance from c2 to mean c1:', c1_c2_mean_dist)

    print('Relative Difference:', (c1_c2_mean_dist - c1_mean_dist) / c1_mean_dist)

    c1_distances = np.array(c1_distances)
    c1_c2_distances = np.array(c1_c2_distances)

    highest = max(c1_distances.max(), c1_c2_distances.max())
    c1_distances /= highest
    c1_c2_distances /= highest
    b = 50
    r = (0, 1)
    bins = np.linspace(r[0], r[1], b)

    c1_values, c1_base = np.histogram(c1_distances, bins=b, range=r)
    c1_values = np.array(c1_values / len(cluster1))
    c1_cumsum = np.cumsum(c1_values)
    c1_c2_values, c1_c2_base = np.histogram(c1_c2_distances, bins=b, range=r)
    c1_c2_values = np.array(c1_c2_values / len(cluster2))
    c1_c2_cumsum = np.cumsum(c1_c2_values)

    # total = min(len(cluster1), len(cluster2))
    overlap = 0
    for i in range(b):
        overlap += min(c1_values[i], c1_c2_values[i])
    # overlap /= total
    print('Overlap coefficient:', overlap)

    bc = 0
    for i in range(b):
        bc += sqrt(c1_values[i] * c1_c2_values[i])
    print('Bhattacharyya coefficient:', bc)

    if True:
        weights1 = np.ones_like(c1_distances) / len(c1_distances)
        weights2 = np.ones_like(c1_c2_distances) / len(c1_c2_distances)

        fig, ax = plt.subplots()
        ax.hist(c1_distances, bins, color='g', alpha=0.4, weights=weights1)
        ax.hist(c1_c2_distances, bins, color='b', alpha=0.4, weights=weights2)
        ax.set_xlabel('Afstand', fontsize=20)
        ax.set_ylabel('Frequentie', fontsize=20)
        ax2 = ax.twinx()
        ax2.plot(c1_base[:-1], c1_cumsum, c='g', linewidth=2)
        ax2.plot(c1_c2_base[:-1], c1_c2_cumsum, c='b', linewidth=2)
        ax2.set_ylabel('Cumulatieve frequentie', fontsize=20)
        if title is not None:
            plt.suptitle(title)
        plt.show()


roads = get_roads_from_xml_file(path_road_data)
road_colors = []
nonroad_colors = []
all_colors = []
visited = set()
i = 0

for image_name in os.listdir(path_images):
    i += 1

    if i > 150:
        break

    if 'reference' in image_name or 'extracted' in image_name:
        i -= 1
        continue

    image_name_list = image_name.split('_')
    road_name = image_name_list[0]
    segment_number = None
    if len(image_name_list) == 4:
        segment_number = int(image_name_list[1])
    zoom_level = int(image_name_list[-2][1:])

    if road_name in visited:
        i -= 1
        continue
    else:
        visited.add(road_name)

    current_road = None
    for road in roads:
        if road.name == road_name:
            current_road = road
            if segment_number is None:
                center = road.center()
            else:
                center = road.segments[segment_number - 1].center()
            break

    if current_road is None or center is None:
        print('Unknown road')
        i -= 1
        continue
    else:
        img = img_as_float(imread(path_images + image_name))
        # img = gaussian(img, sigma=0.5, multichannel=True)
        size = len(img)

        all_road_pixels = set()
        for road in roads:
            road_pixels = road.pixels(size, zoom_level, center=center)
            for (x, y) in road_pixels:
                if 0 <= x < size and 0 <= y < size:
                    all_road_pixels.add((x, y))

        buffered_all_road_pixels = set()
        for x, y in all_road_pixels:
            rr, cc = circle(y, x, 4.5, (size, size))
            for r, c in zip(rr, cc):
                buffered_all_road_pixels.add((c, r))
                # img[r][c] = (1, 0, 0)

        current_road_pixels = current_road.pixels(size, zoom_level, center=center)

        rc = []
        for x, y in current_road_pixels:
            rr, cc = circle(y, x, buffer_width, (size, size))
            for r, c in zip(rr, cc):
                rc.append(np.array(img[r][c]))
                # img[r][c] = (1, 1, 0)

        road_colors.extend(rc)
        rc_mean = np.mean(rc, axis=0)

        print('-----')
        print('{}) {}'.format(i, road_name))
        print('{} road colors added'.format(len(rc)))
        print('Mean road color:', rc_mean)

        nrc = []
        for r in range(0, len(img), 32):
            for c in range(0, len(img[0]), 32):
                all_colors.append(np.array(img[r][c]))
                if (c, r) not in buffered_all_road_pixels:
                    nrc.append(np.array(img[r][c]))

        nonroad_colors.extend(nrc)
        nrc_mean = np.mean(nrc, axis=0)

        print('{} nonroad colors added'.format(len(nrc)))
        print('Mean nonroad color:', nrc_mean)
        print('Color difference:', rc_mean - nrc_mean)
        print('-----')

        if False:
            ax = plt.gca()
            im = ax.imshow(img, interpolation='kaiser')
            plt.show()

        img = []
        gc.collect()

print('{} road colors collected'.format(len(road_colors)))
print('{} nonroad colors collected'.format(len(nonroad_colors)))

rc_mean = np.mean(road_colors, axis=0)
print('Mean road color:', rc_mean)

nrc_mean = np.mean(nonroad_colors, axis=0)
print('Mean nonroad color:', nrc_mean)

# cov_matrix = np.cov(road_colors, rowvar=False)
# print('Covariance matrix:', cov_matrix)
# inv_cov_matrix = np.linalg.inv(cov_matrix)
# print('Inverse covariance matrix:', inv_cov_matrix)

# ------
#  RGB
# ------

rgb_mean = np.mean(road_colors, axis=0)
rgb_road_colors = road_colors  # - rgb_mean
rgb_nonroad_colors = nonroad_colors  # - rgb_mean

# std_rgb = np.std(rgb_road_colors, axis=0)
# rgb_road_colors = rgb_road_colors / std_rgb
# rgb_nonroad_colors = rgb_nonroad_colors / std_rgb

rgb_mean = np.mean(rgb_road_colors, axis=0)

print('-----')
print('RGB')
get_cluster_distances(rgb_road_colors, rgb_nonroad_colors, 'RGB')
print('-----')

red_road = [color[0] for color in rgb_road_colors]
green_road = [color[1] for color in rgb_road_colors]
blue_road = [color[2] for color in rgb_road_colors]

red_nonroad = [color[0] for color in rgb_nonroad_colors]
green_nonroad = [color[1] for color in rgb_nonroad_colors]
blue_nonroad = [color[2] for color in rgb_nonroad_colors]

if False:
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(red_nonroad, blue_nonroad, green_nonroad, c='b', alpha=0.05)
    ax.scatter(red_road, blue_road, green_road, c='g', alpha=0.05)

    ax.set_xlabel('Rood')
    ax.set_ylabel('Blauw')
    ax.set_zlabel('Groen')

    plt.show()

if False:
    fig, ax = plt.subplots(2, 3)

    ax[0][0].plot(red_nonroad, green_nonroad, 'bo', alpha=0.05)
    ax[0][0].plot(red_road, green_road, 'go', alpha=0.05)
    ax[0][0].plot(rgb_mean[0], rgb_mean[1], 'ro', alpha=1)
    ax[0][0].set_xlabel('Rood', fontsize=20)
    ax[0][0].set_ylabel('Groen', fontsize=20)
    ax[0][0].set_xlim([0, 1])
    ax[0][0].set_ylim([0, 1])

    ax[0][1].plot(red_nonroad, blue_nonroad, 'bo', alpha=0.05)
    ax[0][1].plot(red_road, blue_road, 'go', alpha=0.05)
    ax[0][1].plot(rgb_mean[0], rgb_mean[2], 'ro', alpha=1)
    ax[0][1].set_xlabel('Rood', fontsize=20)
    ax[0][1].set_ylabel('Blauw', fontsize=20)
    ax[0][1].set_xlim([0, 1])
    ax[0][1].set_ylim([0, 1])

    ax[0][2].plot(blue_nonroad, green_nonroad, 'bo', alpha=0.05)
    ax[0][2].plot(blue_road, green_road, 'go', alpha=0.05)
    ax[0][2].plot(rgb_mean[2], rgb_mean[1], 'ro', alpha=1)
    ax[0][2].set_xlabel('Blauw', fontsize=20)
    ax[0][2].set_ylabel('Groen', fontsize=20)
    ax[0][2].set_xlim([0, 1])
    ax[0][2].set_ylim([0, 1])

    bins = np.linspace(0, 1, 50)
    ticks = np.arange(0, 1.1, 0.2)

    ax[1][0].hist(red_nonroad, bins, color='b', alpha=0.55)
    ax[1][0].hist(red_road, bins, color='g', alpha=0.6)
    ax[1][0].set_xlabel('Rood', fontsize=20)
    # ax[1][0].set_ylabel('Frequentie', fontsize=20)
    ax[1][0].set_xticks(ticks)

    ax[1][1].hist(green_nonroad, bins, color='b', alpha=0.55)
    ax[1][1].hist(green_road, bins, color='g', alpha=0.6)
    ax[1][1].set_xlabel('Groen', fontsize=20)
    ax[1][1].set_xticks(ticks)

    ax[1][2].hist(blue_nonroad, bins, color='b', alpha=0.55)
    ax[1][2].hist(blue_road, bins, color='g', alpha=0.6)
    ax[1][2].set_xlabel('Blauw', fontsize=20)
    ax[1][2].set_xticks(ticks)

    fig.subplots_adjust(left=0.15, right=0.85, top=0.96, bottom=0.09)
    plt.show()

# ------------
#  PCA on RGB
# ------------

# center data
ac_mean = np.mean(all_colors, axis=0)
all_colors_centered = all_colors - ac_mean
# rc_mean = np.mean(road_colors, axis=0)
road_colors_centered = road_colors - ac_mean
nonroad_colors_centered = nonroad_colors - ac_mean

# compute principal components and order them
cov_matrix = np.cov(all_colors_centered, rowvar=False)
eigvals, eigvecs = linalg.eigh(cov_matrix)
idx = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, idx]
eigvals = eigvals[idx]

# transform data to PCA space
all_colors_pca = np.dot(eigvecs.T, all_colors_centered.T).T
road_colors_pca = np.dot(eigvecs.T, road_colors_centered.T).T
nonroad_colors_pca = np.dot(eigvecs.T, nonroad_colors_centered.T).T

# standardize data: divide by standard deviation in each PC direction
std_xyz = np.std(all_colors_pca, axis=0)
road_colors_pca_standardized = road_colors_pca / std_xyz
nonroad_colors_pca_standardized = nonroad_colors_pca / std_xyz

print('-----')
print('PCA RGB')
get_cluster_distances(road_colors_pca_standardized, nonroad_colors_pca_standardized, 'PCA RGB')
print('-----')

xr = [vector[0] for vector in road_colors_pca_standardized]
yr = [vector[1] for vector in road_colors_pca_standardized]
zr = [vector[2] for vector in road_colors_pca_standardized]

xnr = [vector[0] for vector in nonroad_colors_pca_standardized]
ynr = [vector[1] for vector in nonroad_colors_pca_standardized]
znr = [vector[2] for vector in nonroad_colors_pca_standardized]

xr_rgb = np.array(xr)
yr_rgb = np.array(yr)
zr_rgb = np.array(zr)

xnr_rgb = np.array(xnr)
ynr_rgb = np.array(ynr)
znr_rgb = np.array(znr)

# # test
# x1, y1, z1 = road_colors_pca_standardized[0]
# x2, y2, z2 = nonroad_colors_pca_standardized[0]
# dx = x1 - x2
# dy = y1 - y2
# dz = z1 - z2
# eucl_dist = sqrt(dx * dx + dy * dy + dz * dz)
# print(eucl_dist)
#
# cov_matrix = np.cov(road_colors, rowvar=False)
# inv_cov_matrix = np.linalg.inv(cov_matrix)
# mahal_dist = mahalanobis_distance(road_colors[0], nonroad_colors[0], inv_cov_matrix)
# print(mahal_dist)

if False:
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(xr, yr, zr, c='g', alpha=0.05)
    ax.scatter(xnr, ynr, znr, c='b', alpha=0.05)

    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])

    ax.plot([-5, 5], [0, 0], [0, 0], 'r')
    ax.plot([0, 0], [-5, 5], [0, 0], 'r')
    ax.plot([0, 0], [0, 0], [-5, 5], 'r')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

if False:
    fig, ax = plt.subplots(2, 3)
    ticks = np.arange(-9, 10, 3)

    ax[0][0].plot(xnr, ynr, 'bo', alpha=0.05)
    ax[0][0].plot(xr, yr, 'go', alpha=0.05)
    ax[0][0].plot(0, 0, 'ro', alpha=1)
    ax[0][0].set_xlabel('PC1', fontsize=20)
    ax[0][0].set_ylabel('PC2', fontsize=20)
    ax[0][0].set_xlim([-9, 9])
    ax[0][0].set_ylim([-9, 9])
    ax[0][0].set_xticks(ticks)
    ax[0][0].set_yticks(ticks)

    ax[0][1].plot(xnr, znr, 'bo', alpha=0.05)
    ax[0][1].plot(xr, zr, 'go', alpha=0.05)
    ax[0][1].plot(0, 0, 'ro', alpha=1)
    ax[0][1].set_xlabel('PC1', fontsize=20)
    ax[0][1].set_ylabel('PC3', fontsize=20)
    ax[0][1].set_xlim([-9, 9])
    ax[0][1].set_ylim([-9, 9])
    ax[0][1].set_xticks(ticks)
    ax[0][1].set_yticks(ticks)

    ax[0][2].plot(znr, ynr, 'bo', alpha=0.05)
    ax[0][2].plot(zr, yr, 'go', alpha=0.05)
    ax[0][2].plot(0, 0, 'ro', alpha=1)
    ax[0][2].set_xlabel('PC3', fontsize=20)
    ax[0][2].set_ylabel('PC2', fontsize=20)
    ax[0][2].set_xlim([-9, 9])
    ax[0][2].set_ylim([-9, 9])
    ax[0][2].set_xticks(ticks)
    ax[0][2].set_yticks(ticks)

    bins = np.linspace(-9, 9, 50)
    ticks = np.arange(-9, 10, 3)

    ax[1][0].hist(xnr, bins, color='b', alpha=0.55)
    ax[1][0].hist(xr, bins, color='g', alpha=0.6)
    ax[1][0].set_xlabel('PC1', fontsize=20)
    # ax[1][0].set_ylabel('Frequentie', fontsize=20)
    ax[1][0].set_xticks(ticks)

    ax[1][1].hist(ynr, bins, color='b', alpha=0.55)
    ax[1][1].hist(yr, bins, color='g', alpha=0.6)
    ax[1][1].set_xlabel('PC2', fontsize=20)
    ax[1][1].set_xticks(ticks)

    ax[1][2].hist(znr, bins, color='b', alpha=0.55)
    ax[1][2].hist(zr, bins, color='g', alpha=0.6)
    ax[1][2].set_xlabel('PC3', fontsize=20)
    ax[1][2].set_xticks(ticks)

    fig.subplots_adjust(left=0.15, right=0.85, top=0.96, bottom=0.09)
    plt.show()

# ------------
#  HSV & HLS
# ------------

hsv_all_colors = rgb2hsv([all_colors])[0]
hsv_road_colors = rgb2hsv([road_colors])[0]
hsv_nonroad_colors = rgb2hsv([nonroad_colors])[0]

hsv_mean = np.mean(hsv_all_colors, axis=0)
hsv_all_colors -= hsv_mean
hsv_road_colors -= hsv_mean
hsv_nonroad_colors -= hsv_mean

std_hsv = np.std(hsv_all_colors, axis=0)
hsv_road_colors /= std_hsv
hsv_nonroad_colors /= std_hsv

hsv_mean = np.mean(hsv_all_colors, axis=0)

print('-----')
print('HSV')
get_cluster_distances(hsv_road_colors, hsv_nonroad_colors, 'HSV')
print('-----')

hue_road = [color[0] for color in hsv_road_colors]
sat_road = [color[1] for color in hsv_road_colors]
val_road = [color[2] for color in hsv_road_colors]

hue_nonroad = [color[0] for color in hsv_nonroad_colors]
sat_nonroad = [color[1] for color in hsv_nonroad_colors]
val_nonroad = [color[2] for color in hsv_nonroad_colors]

# hls_road_colors = [rgb_to_hls(r, g, b) for r, g, b in road_colors]
# hls_nonroad_colors = [rgb_to_hls(r, g, b) for r, g, b in nonroad_colors]
#
# hu_road = [color[0] for color in hls_road_colors]
# li_road = [color[1] for color in hls_road_colors]
# sa_road = [color[2] for color in hls_road_colors]
#
# hu_nonroad = [color[0] for color in hls_nonroad_colors]
# li_nonroad = [color[1] for color in hls_nonroad_colors]
# sa_nonroad = [color[2] for color in hls_nonroad_colors]

if False:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(hue_nonroad, val_nonroad, sat_nonroad, c='b', alpha=0.05)
    ax.scatter(hue_road, val_road, sat_road, c='g', alpha=0.05)

    ax.set_xlabel('Hue')
    ax.set_ylabel('Value')
    ax.set_zlabel('Saturation')

    plt.show()

if False:
    fig, ax = plt.subplots(2, 3)
    ticks = np.arange(-9, 10, 3)

    ax[0][0].plot(hue_nonroad, sat_nonroad, 'bo', alpha=0.05)
    ax[0][0].plot(hue_road, sat_road, 'go', alpha=0.05)
    ax[0][0].plot(hsv_mean[0], hsv_mean[1], 'ro', alpha=1)
    # ax[0][0].set_xlim(min_hsv, max_hsv)
    # ax[0][0].set_ylim(min_hsv, max_hsv)
    ax[0][0].set_xlabel('Hue', fontsize=20)
    ax[0][0].set_ylabel('Saturation', fontsize=20)
    ax[0][0].set_xlim([-9, 9])
    ax[0][0].set_ylim([-9, 9])
    ax[0][0].set_xticks(ticks)
    ax[0][0].set_yticks(ticks)

    ax[0][1].plot(hue_nonroad, val_nonroad, 'bo', alpha=0.05)
    ax[0][1].plot(hue_road, val_road, 'go', alpha=0.05)
    ax[0][1].plot(hsv_mean[0], hsv_mean[2], 'ro', alpha=1)
    # ax[0][1].set_xlim(min_hsv, max_hsv)
    # ax[0][1].set_ylim(min_hsv, max_hsv)
    ax[0][1].set_xlabel('Hue', fontsize=20)
    ax[0][1].set_ylabel('Value', fontsize=20)
    ax[0][1].set_xlim([-9, 9])
    ax[0][1].set_ylim([-9, 9])
    ax[0][1].set_xticks(ticks)
    ax[0][1].set_yticks(ticks)

    ax[0][2].plot(val_nonroad, sat_nonroad, 'bo', alpha=0.05)
    ax[0][2].plot(val_road, sat_road, 'go', alpha=0.05)
    ax[0][2].plot(hsv_mean[2], hsv_mean[1], 'ro', alpha=1)
    # ax[0][2].set_xlim(min_hsv, max_hsv)
    # ax[0][2].set_ylim(min_hsv, max_hsv)
    ax[0][2].set_xlabel('Value', fontsize=20)
    ax[0][2].set_ylabel('Saturation', fontsize=20)
    ax[0][2].set_xlim([-9, 9])
    ax[0][2].set_ylim([-9, 9])
    ax[0][2].set_xticks(ticks)
    ax[0][2].set_yticks(ticks)

    bins = np.linspace(-9, 9, 50)
    ticks = np.arange(-9, 10, 3)

    ax[1][0].hist(hue_nonroad, bins, color='b', alpha=0.55)
    ax[1][0].hist(hue_road, bins, color='g', alpha=0.6)
    ax[1][0].set_xlabel('Hue', fontsize=20)
    # ax[1][0].set_ylabel('Frequentie', fontsize=20)
    ax[1][0].set_xticks(ticks)

    ax[1][1].hist(sat_nonroad, bins, color='b', alpha=0.55)
    ax[1][1].hist(sat_road, bins, color='g', alpha=0.6)
    ax[1][1].set_xlabel('Saturation', fontsize=20)
    ax[1][1].set_xticks(ticks)

    ax[1][2].hist(val_nonroad, bins, color='b', alpha=0.55)
    ax[1][2].hist(val_road, bins, color='g', alpha=0.6)
    ax[1][2].set_xlabel('Value', fontsize=20)
    ax[1][2].set_xticks(ticks)

    fig.subplots_adjust(left=0.15, right=0.85, top=0.96, bottom=0.09)
    plt.show()

# ------------
#  PCA on HSV
# ------------

hsv_all_colors = rgb2hsv([all_colors])[0]
hsv_road_colors = rgb2hsv([road_colors])[0]
hsv_nonroad_colors = rgb2hsv([nonroad_colors])[0]

# center data
ac_mean = np.mean(hsv_all_colors, axis=0)
all_colors_centered = hsv_all_colors - ac_mean
#rc_mean = np.mean(hsv_road_colors, axis=0)
road_colors_centered = hsv_road_colors - ac_mean
nonroad_colors_centered = hsv_nonroad_colors - ac_mean

# compute principal components and order them
cov_matrix = np.cov(all_colors_centered, rowvar=False)
eigvals, eigvecs = linalg.eigh(cov_matrix)
idx = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, idx]
eigvals = eigvals[idx]

# transform data to PCA space
all_colors_pca = np.dot(eigvecs.T, all_colors_centered.T).T
road_colors_pca = np.dot(eigvecs.T, road_colors_centered.T).T
nonroad_colors_pca = np.dot(eigvecs.T, nonroad_colors_centered.T).T

# standardize data: divide by standard deviation in each PC direction
std_xyz = np.std(all_colors_pca, axis=0)
road_colors_pca_standardized = road_colors_pca / std_xyz
nonroad_colors_pca_standardized = nonroad_colors_pca / std_xyz

print('-----')
print('PCA HSV')
get_cluster_distances(road_colors_pca_standardized, nonroad_colors_pca_standardized, 'PCA HSV')
print('-----')

xr = [vector[0] for vector in road_colors_pca_standardized]
yr = [vector[1] for vector in road_colors_pca_standardized]
zr = [vector[2] for vector in road_colors_pca_standardized]

xnr = [vector[0] for vector in nonroad_colors_pca_standardized]
ynr = [vector[1] for vector in nonroad_colors_pca_standardized]
znr = [vector[2] for vector in nonroad_colors_pca_standardized]

if False:
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(xr, yr, zr, c='g', alpha=0.05)
    ax.scatter(xnr, ynr, znr, c='b', alpha=0.05)

    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])

    ax.plot([-5, 5], [0, 0], [0, 0], 'r')
    ax.plot([0, 0], [-5, 5], [0, 0], 'r')
    ax.plot([0, 0], [0, 0], [-5, 5], 'r')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

if False:
    fig, ax = plt.subplots(2, 3)
    ticks = np.arange(-9, 10, 3)

    ax[0][0].plot(xnr, ynr, 'bo', alpha=0.05)
    ax[0][0].plot(xr, yr, 'go', alpha=0.05)
    ax[0][0].plot(0, 0, 'ro', alpha=1)
    ax[0][0].set_xlabel('PC1', fontsize=20)
    ax[0][0].set_ylabel('PC2', fontsize=20)
    ax[0][0].set_xlim([-9, 9])
    ax[0][0].set_ylim([-9, 9])
    ax[0][0].set_xticks(ticks)
    ax[0][0].set_yticks(ticks)

    ax[0][1].plot(xnr, znr, 'bo', alpha=0.05)
    ax[0][1].plot(xr, zr, 'go', alpha=0.05)
    ax[0][1].plot(0, 0, 'ro', alpha=1)
    ax[0][1].set_xlabel('PC1', fontsize=20)
    ax[0][1].set_ylabel('PC3', fontsize=20)
    ax[0][1].set_xlim([-9, 9])
    ax[0][1].set_ylim([-9, 9])
    ax[0][1].set_xticks(ticks)
    ax[0][1].set_yticks(ticks)

    ax[0][2].plot(znr, ynr, 'bo', alpha=0.05)
    ax[0][2].plot(zr, yr, 'go', alpha=0.05)
    ax[0][2].plot(0, 0, 'ro', alpha=1)
    ax[0][2].set_xlabel('PC3', fontsize=20)
    ax[0][2].set_ylabel('PC2', fontsize=20)
    ax[0][2].set_xlim([-9, 9])
    ax[0][2].set_ylim([-9, 9])
    ax[0][2].set_xticks(ticks)
    ax[0][2].set_yticks(ticks)

    bins = np.linspace(-9, 9, 50)
    ticks = np.arange(-9, 10, 3)

    ax[1][0].hist(xnr, bins, color='b', alpha=0.55)
    ax[1][0].hist(xr, bins, color='g', alpha=0.6)
    ax[1][0].set_xlabel('PC1', fontsize=20)
    # ax[1][0].set_ylabel('Frequentie', fontsize=20)
    ax[1][0].set_xticks(ticks)

    ax[1][1].hist(ynr, bins, color='b', alpha=0.55)
    ax[1][1].hist(yr, bins, color='g', alpha=0.6)
    ax[1][1].set_xlabel('PC2', fontsize=20)
    ax[1][1].set_xticks(ticks)

    ax[1][2].hist(znr, bins, color='b', alpha=0.55)
    ax[1][2].hist(zr, bins, color='g', alpha=0.6)
    ax[1][2].set_xlabel('PC3', fontsize=20)
    ax[1][2].set_xticks(ticks)

    fig.subplots_adjust(left=0.15, right=0.85, top=0.96, bottom=0.09)
    plt.show()

# --------
#  L*a*b*
# --------

lab_all_colors = rgb2lab([all_colors])[0]
lab_road_colors = rgb2lab([road_colors])[0]
lab_nonroad_colors = rgb2lab([nonroad_colors])[0]

lab_mean = np.mean(lab_all_colors, axis=0)
lab_all_colors -= lab_mean
lab_road_colors -= lab_mean
lab_nonroad_colors -= lab_mean

std_lab = np.std(lab_all_colors, axis=0)
lab_road_colors /= std_lab
lab_nonroad_colors /= std_lab

lab_mean = np.mean(lab_all_colors, axis=0)

print('-----')
print('Lab')
get_cluster_distances(lab_road_colors, lab_nonroad_colors, 'L*a*b*')
print('-----')

l_road = [color[0] for color in lab_road_colors]
a_road = [color[1] for color in lab_road_colors]
b_road = [color[2] for color in lab_road_colors]

l_nonroad = [color[0] for color in lab_nonroad_colors]
a_nonroad = [color[1] for color in lab_nonroad_colors]
b_nonroad = [color[2] for color in lab_nonroad_colors]

if False:
    fig, ax = plt.subplots(2, 3)
    ticks100 = np.arange(0, 101, 25)
    ticks50 = np.arange(-50, 51, 25)
    # ticks = np.arange(-9, 10, 3)

    ax[0][0].plot(l_nonroad, a_nonroad, 'bo', alpha=0.05)
    ax[0][0].plot(l_road, a_road, 'go', alpha=0.05)
    ax[0][0].plot(lab_mean[0], lab_mean[1], 'ro', alpha=1)
    ax[0][0].set_xlabel('L*', fontsize=20)
    ax[0][0].set_ylabel('a*', fontsize=20)
    ax[0][0].set_xlim([0, 100])
    ax[0][0].set_ylim([-50, 50])
    ax[0][0].set_xticks(ticks100)
    ax[0][0].set_yticks(ticks50)
    # ax[0][0].set_xlim([-9, 9])
    # ax[0][0].set_ylim([-9, 9])
    # ax[0][0].set_xticks(ticks)
    # ax[0][0].set_yticks(ticks)

    ax[0][1].plot(l_nonroad, b_nonroad, 'bo', alpha=0.05)
    ax[0][1].plot(l_road, b_road, 'go', alpha=0.05)
    ax[0][1].plot(lab_mean[0], lab_mean[2], 'ro', alpha=1)
    ax[0][1].set_xlabel('L*', fontsize=20)
    ax[0][1].set_ylabel('b*', fontsize=20)
    ax[0][1].set_xlim([0, 100])
    ax[0][1].set_ylim([-50, 50])
    ax[0][1].set_xticks(ticks100)
    ax[0][1].set_yticks(ticks50)
    # ax[0][1].set_xlim([-9, 9])
    # ax[0][1].set_ylim([-9, 9])
    # ax[0][1].set_xticks(ticks)
    # ax[0][1].set_yticks(ticks)

    ax[0][2].plot(b_nonroad, a_nonroad, 'bo', alpha=0.05)
    ax[0][2].plot(b_road, a_road, 'go', alpha=0.05)
    ax[0][2].plot(lab_mean[2], lab_mean[1], 'ro', alpha=1)
    ax[0][2].set_xlabel('b*', fontsize=20)
    ax[0][2].set_ylabel('a*', fontsize=20)
    ax[0][2].set_xlim([-50, 50])
    ax[0][2].set_ylim([-50, 50])
    ax[0][2].set_xticks(ticks50)
    ax[0][2].set_yticks(ticks50)
    # ax[0][2].set_xlim([-9, 9])
    # ax[0][2].set_ylim([-9, 9])
    # ax[0][2].set_xticks(ticks)
    # ax[0][2].set_yticks(ticks)

    # bins = np.linspace(-9, 9, 50)
    # ticks = np.arange(-9, 10, 3)
    bins = np.linspace(0, 100, 50)

    ax[1][0].hist(l_nonroad, bins, color='b', alpha=0.55)
    ax[1][0].hist(l_road, bins, color='g', alpha=0.6)
    ax[1][0].set_xlabel('L*', fontsize=20)
    # ax[1][0].set_ylabel('Frequentie', fontsize=20)
    ax[1][0].set_xticks(ticks100)

    bins = np.linspace(-50, 50, 50)

    ax[1][1].hist(a_nonroad, bins, color='b', alpha=0.55)
    ax[1][1].hist(a_road, bins, color='g', alpha=0.6)
    ax[1][1].set_xlabel('a*', fontsize=20)
    ax[1][1].set_xticks(ticks50)

    ax[1][2].hist(b_nonroad, bins, color='b', alpha=0.55)
    ax[1][2].hist(b_road, bins, color='g', alpha=0.6)
    ax[1][2].set_xlabel('b*', fontsize=20)
    ax[1][2].set_xticks(ticks50)

    fig.subplots_adjust(left=0.15, right=0.85, top=0.96, bottom=0.09)
    plt.show()

# ------------
#  PCA on Lab
# ------------

lab_all_colors = rgb2lab([all_colors])[0]
lab_road_colors = rgb2lab([road_colors])[0]
lab_nonroad_colors = rgb2lab([nonroad_colors])[0]

# center data
ac_mean = np.mean(lab_all_colors, axis=0)
all_colors_centered = lab_all_colors - ac_mean
#rc_mean = np.mean(lab_road_colors, axis=0)
road_colors_centered = lab_road_colors - ac_mean
nonroad_colors_centered = lab_nonroad_colors - ac_mean

# compute principal components and order them
cov_matrix = np.cov(all_colors_centered, rowvar=False)
eigvals, eigvecs = linalg.eigh(cov_matrix)
idx = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, idx]
eigvals = eigvals[idx]

# transform data to PCA space
all_colors_pca = np.dot(eigvecs.T, all_colors_centered.T).T
road_colors_pca = np.dot(eigvecs.T, road_colors_centered.T).T
nonroad_colors_pca = np.dot(eigvecs.T, nonroad_colors_centered.T).T

# standardize data: divide by standard deviation in each PC direction
std_xyz = np.std(all_colors_pca, axis=0)
road_colors_pca_standardized = road_colors_pca / std_xyz
nonroad_colors_pca_standardized = nonroad_colors_pca / std_xyz

print('-----')
print('PCA Lab')
get_cluster_distances(road_colors_pca_standardized, nonroad_colors_pca_standardized, 'PCA L*a*b*')
print('-----')

xr = [vector[0] for vector in road_colors_pca_standardized]
yr = [vector[1] for vector in road_colors_pca_standardized]
zr = [vector[2] for vector in road_colors_pca_standardized]

xnr = [vector[0] for vector in nonroad_colors_pca_standardized]
ynr = [vector[1] for vector in nonroad_colors_pca_standardized]
znr = [vector[2] for vector in nonroad_colors_pca_standardized]

xr_lab = np.array(xr)
yr_lab = np.array(yr)
zr_lab = np.array(zr)

xnr_lab = np.array(xnr)
ynr_lab = np.array(ynr)
znr_lab = np.array(znr)

if False:
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(xr, yr, zr, c='g', alpha=0.05)
    ax.scatter(xnr, ynr, znr, c='b', alpha=0.05)

    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])

    ax.plot([-5, 5], [0, 0], [0, 0], 'r')
    ax.plot([0, 0], [-5, 5], [0, 0], 'r')
    ax.plot([0, 0], [0, 0], [-5, 5], 'r')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

if False:
    fig, ax = plt.subplots(2, 3)
    ticks = np.arange(-9, 10, 3)

    ax[0][0].plot(xnr, ynr, 'bo', alpha=0.05)
    ax[0][0].plot(xr, yr, 'go', alpha=0.05)
    ax[0][0].plot(0, 0, 'ro', alpha=1)
    ax[0][0].set_xlabel('PC1', fontsize=20)
    ax[0][0].set_ylabel('PC2', fontsize=20)
    ax[0][0].set_xlim([-9, 9])
    ax[0][0].set_ylim([-9, 9])
    ax[0][0].set_xticks(ticks)
    ax[0][0].set_yticks(ticks)

    ax[0][1].plot(xnr, znr, 'bo', alpha=0.05)
    ax[0][1].plot(xr, zr, 'go', alpha=0.05)
    ax[0][1].plot(0, 0, 'ro', alpha=1)
    ax[0][1].set_xlabel('PC1', fontsize=20)
    ax[0][1].set_ylabel('PC3', fontsize=20)
    ax[0][1].set_xlim([-9, 9])
    ax[0][1].set_ylim([-9, 9])
    ax[0][1].set_xticks(ticks)
    ax[0][1].set_yticks(ticks)

    ax[0][2].plot(znr, ynr, 'bo', alpha=0.05)
    ax[0][2].plot(zr, yr, 'go', alpha=0.05)
    ax[0][2].plot(0, 0, 'ro', alpha=1)
    ax[0][2].set_xlabel('PC3', fontsize=20)
    ax[0][2].set_ylabel('PC2', fontsize=20)
    ax[0][2].set_xlim([-9, 9])
    ax[0][2].set_ylim([-9, 9])
    ax[0][2].set_xticks(ticks)
    ax[0][2].set_yticks(ticks)

    bins = np.linspace(-9, 9, 50)
    ticks = np.arange(-9, 10, 3)

    ax[1][0].hist(xnr, bins, color='b', alpha=0.55)
    ax[1][0].hist(xr, bins, color='g', alpha=0.6)
    ax[1][0].set_xlabel('PC1', fontsize=20)
    # ax[1][0].set_ylabel('Frequentie', fontsize=20)
    ax[1][0].set_xticks(ticks)

    ax[1][1].hist(ynr, bins, color='b', alpha=0.55)
    ax[1][1].hist(yr, bins, color='g', alpha=0.6)
    ax[1][1].set_xlabel('PC2', fontsize=20)
    ax[1][1].set_xticks(ticks)

    ax[1][2].hist(znr, bins, color='b', alpha=0.55)
    ax[1][2].hist(zr, bins, color='g', alpha=0.6)
    ax[1][2].set_xlabel('PC3', fontsize=20)
    ax[1][2].set_xticks(ticks)

    fig.subplots_adjust(left=0.15, right=0.85, top=0.96, bottom=0.09)
    plt.show()

# ----
#  MC
# ----

# ones = np.ones(len(road_colors))
# mc_road_colors = list(zip(xr_lab, yr_lab, zr_lab, np.array(sat_road)))
# # xr_rgb, yr_rgb, zr_rgb, xr_hsv, yr_hsv, zr_hsv, xr_lab, yr_lab, zr_lab
# mc_nonroad_colors = list(zip(xnr_lab, ynr_lab, znr_lab, np.array(sat_nonroad)))
# # xnr_rgb, ynr_rgb, znr_rgb, xnr_hsv, ynr_hsv, znr_hsv, xnr_lab, ynr_lab, znr_lab
#
# print('-----')
# print('MC')
# get_cluster_distances(mc_road_colors, mc_nonroad_colors)
# print('-----')
#
# if False:
#     fig, ax = plt.subplots(2, 3)
#
#     ax[0][0].plot(sat_nonroad, a_nonroad, 'bo', alpha=0.05)
#     ax[0][0].plot(sat_road, a_road, 'go', alpha=0.05)
#     ax[0][0].set_xlabel('Sat', fontsize=20)
#     ax[0][0].set_ylabel('a*', fontsize=20)
#
#     ax[0][1].plot(sat_nonroad, b_nonroad, 'bo', alpha=0.05)
#     ax[0][1].plot(sat_road, b_road, 'go', alpha=0.05)
#     ax[0][1].set_xlabel('Sat', fontsize=20)
#     ax[0][1].set_ylabel('b*', fontsize=20)
#
#     ax[0][2].plot(b_nonroad, a_nonroad, 'bo', alpha=0.05)
#     ax[0][2].plot(b_road, a_road, 'go', alpha=0.05)
#     ax[0][2].set_xlabel('b*', fontsize=20)
#     ax[0][2].set_ylabel('a*', fontsize=20)
#
#     lowest = min(min(sat_road), min(sat_nonroad))
#     highest = max(max(sat_road), max(sat_nonroad))
#     bins = np.linspace(lowest, highest, 50)
#
#     ax[1][0].hist(sat_nonroad, bins, color='b', alpha=0.5)
#     ax[1][0].hist(sat_road, bins, color='g', alpha=0.5)
#     ax[1][0].set_xlabel('Sat', fontsize=20)
#     # ax[1][0].set_ylabel('Frequentie', fontsize=20)
#
#     lowest = min(min(a_road), min(a_nonroad))
#     highest = max(max(a_road), max(a_nonroad))
#     bins = np.linspace(lowest, highest, 50)
#
#     ax[1][1].hist(a_nonroad, bins, color='b', alpha=0.5)
#     ax[1][1].hist(a_road, bins, color='g', alpha=0.5)
#     ax[1][1].set_xlabel('a*', fontsize=20)
#
#     lowest = min(min(b_road), min(b_nonroad))
#     highest = max(max(b_road), max(b_nonroad))
#     bins = np.linspace(lowest, highest, 50)
#
#     ax[1][2].hist(b_nonroad, bins, color='b', alpha=0.5)
#     ax[1][2].hist(b_road, bins, color='g', alpha=0.5)
#     ax[1][2].set_xlabel('b*', fontsize=20)
#
#     fig.subplots_adjust(left=0.15, right=0.85, top=0.96, bottom=0.09)
#     plt.show()

# ----
#  LDA
# ----

len_road_colors = len(road_colors)
len_nonroad_colors = len(nonroad_colors)
lda = LDA(n_components=1)
data = np.append(np.array(road_colors), np.array(nonroad_colors), axis=0)
labels = np.append(np.ones(len_road_colors), np.zeros(len_nonroad_colors))
data_lda = lda.fit_transform(data, labels)

roads_lda = data_lda[0:len_road_colors]
nonroads_lda = data_lda[len_road_colors:]

print('-----')
print('LDA')
get_cluster_distances(roads_lda, nonroads_lda, 'LDA')
print('-----')
