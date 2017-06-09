import time
import numpy as np
from skimage.draw import circle
from road_extraction.cost import get_cost_map
from road_extraction.search import get_shortest_path
from road_extraction.smoothing import straight_line_check, simplify_path
from road_extraction.smoothing import smooth_iteratively, smooth_using_moving_centroid, smooth_using_moving_medoid
from road_extraction.smoothing import smooth_using_b_spline, smooth_using_segmented_lowess, smooth_using_search
from road_extraction.turning_point import correct_sharp_turns


def extract_road(img, start, goal, seed_colors_extra=None, post_processing=True):
    start_time_total = time.time()

    seeds = []

    rr, cc = circle(start[1], start[0], 1.5, img.shape[0:2])
    seed_colors_start = img[rr, cc]
    centroid_start = np.mean(seed_colors_start, axis=0)
    seeds.append(centroid_start)

    rr, cc = circle(goal[1], goal[0], 1.5, img.shape[0:2])
    seed_colors_goal = img[rr, cc]
    centroid_goal = np.mean(seed_colors_goal, axis=0)
    seeds.append(centroid_goal)

    if seed_colors_extra is not None and len(seed_colors_extra) > 0:
        centroid_extra = np.mean(seed_colors_extra, axis=0)
        seeds.append(centroid_extra)

    img_colors = np.vstack(img)

    if seed_colors_extra is not None and len(seed_colors_extra) >= 20:
        cov_matrix = np.cov(seed_colors_extra, rowvar=False)
        if np.linalg.matrix_rank(cov_matrix) < 3:
            cov_matrix = np.cov(img_colors, rowvar=False)
    else:
        cov_matrix = np.cov(img_colors, rowvar=False)

    inv_cov_matrix = np.linalg.inv(cov_matrix)
    cost_map = get_cost_map(img, seeds, inv_cov_matrix)
    # print('Cost map: %s seconds' % round((time.time() - start_time), 2))

    # ---------------
    #  Shortest path
    # ---------------
    # start_time = time.time()
    extraction = get_shortest_path(cost_map, start, goal, img, inv_cov_matrix)
    # print('Search: %s seconds' % round((time.time() - start_time), 2))

    # -----------------
    #  Post-processing
    # -----------------
    # start_time = time.time()
    straight_line = straight_line_check(extraction, cost_map, 1.25)

    if straight_line:
        extraction = straight_line
        smoothed_extraction = straight_line
        points = [straight_line[0], straight_line[-1]]
    else:
        if post_processing:
            smoothed_extraction, points = post_process_extraction(extraction, cost_map)
        else:
            smoothed_extraction, points = simplify_path(extraction, e=2)

    # print('Smoothing & post-processing: %s seconds' % round((time.time() - start_time), 2))
    print('Total extraction time: %s seconds' % round((time.time() - start_time_total), 2))
    return smoothed_extraction, points, extraction, cost_map


def post_process_extraction(extraction, cost_map):
    # start_time = time.time()

    # smoothed_extraction = smooth_using_search(extraction, cost_map)
    # smoothed_extraction = smooth_using_b_spline(extraction, sc=1000)
    smoothed_extraction = smooth_using_moving_centroid(extraction, m=20)
    smoothed_extraction, tp_indices = correct_sharp_turns(smoothed_extraction, cost_map.shape)
    smoothed_extraction = smooth_iteratively(smoothed_extraction, cost_map, 1.05, tp_indices=tp_indices)
    smoothed_extraction, points = simplify_path(smoothed_extraction, e=2, tp_indices=tp_indices)

    # print('Smoothing & post-processing: %s seconds' % round((time.time() - start_time), 2))
    return smoothed_extraction, points
