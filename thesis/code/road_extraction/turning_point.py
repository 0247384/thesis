import matplotlib.pyplot as plt
from rdp import rdp
from numpy import pi
from utils.line import fill, fit_line, Line
from utils.distance import euclidean_distance, closest_point

road_width = 9


def get_segment(path, i_start, max_length, margin=0, after=True):
    if len(path) > 0:
        mpml = margin + max_length
        segment = [path[i_start]]
        length = 0
        i_end_margin = 0
        margin_length = 0

        if after:
            max_i = len(path) - 1
            i = i_start

            while length < mpml and i < max_i:
                i += 1
                segment.append(path[i])
                dist = euclidean_distance(path[i - 1], path[i])
                length += dist
                if length < margin:
                    i_end_margin += 1
                    margin_length = length

            if length > mpml:
                segment = segment[:-1]
                length -= dist
                i -= 1

            return segment[i_end_margin:], length - margin_length, i_start + i_end_margin, i
        else:
            i = i_start

            while length < mpml and i > 0:
                i -= 1
                segment.insert(0, path[i])
                dist = euclidean_distance(path[i], path[i + 1])
                length += dist
                if length < margin:
                    i_end_margin -= 1
                    margin_length = length

            if length > mpml:
                segment = segment[1:]
                length -= dist
                i += 1

            if i_end_margin < 0:
                segment = segment[:i_end_margin]

            return segment, length - margin_length, i, i_start + i_end_margin
    else:
        return [], 0, i_start, i_start


def correct_sharp_turns(path, img_shape):
    theta = (3 / 4) * pi
    min_dist_betw_tps = 5 * road_width
    margin_fit = 3 * road_width
    min_len_fit = 2 * road_width
    max_len_fit = 6 * road_width
    max_l = len(path) - 1
    max_x, max_y = img_shape[1] - 1, img_shape[0] - 1

    # get potential turning points using RDP with a low epsilon
    turning_points = rdp(path, epsilon=1)
    turning_points = turning_points[1:-1]

    tp_indices = []
    for tp in turning_points:
        tuple_tp = tuple(tp)
        tp_indices.append(path.index(tuple_tp))

    # remove turning points to close to the start or the end of the path
    tp_indices_new = []
    for i_tp in tp_indices:
        if 10 <= i_tp < len(path) - 10:
            tp_indices_new.append(i_tp)
    tp_indices = tp_indices_new

    turning_points_new = []
    angles = []
    segment_data_tuples = []
    tp_indices_new = []

    # fit lines at turning points and store all relevant info
    for i in tp_indices:
        segment1, len_s1, s1_i_min, s1_i_max = get_segment(path, i, max_len_fit, margin_fit, after=False)
        segment2, len_s2, s2_i_min, s2_i_max = get_segment(path, i, max_len_fit, margin_fit, after=True)

        if len_s1 < min_len_fit:
            line1 = Line(path[0], path[i])
            s1_i_min = 0
        else:
            line1 = fit_line(segment1)

        if len_s2 < min_len_fit:
            line2 = Line(path[i], path[max_l])
            s2_i_max = max_l
        else:
            line2 = fit_line(segment2)

        segment1_data = segment1, len_s1, s1_i_min, s1_i_max
        segment2_data = segment2, len_s2, s2_i_min, s2_i_max

        if line1 and line2:
            angle = line1.angle(line2)
            if angle <= theta:
                ip = line1.intersection(line2)
                if ip:
                    ip = int(round(ip[0])), int(round(ip[1]))
                    cp = closest_point(ip, path[s1_i_max:s2_i_min])
                    line = Line(cp, ip)
                    tp = line.point_on_max_dist_from_p1(road_width)
                    tp = min(tp[0], max_x), min(tp[1], max_y)
                    turning_points_new.append(tp)
                    angles.append(angle)
                    segment_data_tuples.append((segment1_data, segment2_data))
                    tp_indices_new.append(i)

    turning_points = turning_points_new
    tp_indices = tp_indices_new

    # sort turning points by angle and keep only the turning points with the smallest angle in their neighborhood
    tp_indices_sorted_by_angle = sorted(range(len(angles)), key=lambda k: angles[k])
    to_keep = [True] * len(turning_points)
    to_remove = set()

    for i in tp_indices_sorted_by_angle:
        if i in to_remove:
            to_keep[i] = False
        else:
            tp_i = turning_points[i]
            for j, tp_j in enumerate(turning_points):
                if i != j and euclidean_distance(tp_i, tp_j) < min_dist_betw_tps:
                    to_remove.add(j)

    turning_points_new = []
    angles_new = []
    segment_data_tuples_new = []
    tp_indices_new = []

    for i, keep in enumerate(to_keep):
        if keep:
            turning_points_new.append(turning_points[i])
            angles_new.append(angles[i])
            segment_data_tuples_new.append(segment_data_tuples[i])
            tp_indices_new.append(tp_indices[i])

    turning_points = turning_points_new
    angles = angles_new
    segment_data_tuples = segment_data_tuples_new
    tp_indices = tp_indices_new

    # compute the distances between remaining adjacent turning points
    distances = []
    for i in range(len(turning_points) - 1):
        dist = euclidean_distance(turning_points[i], turning_points[i + 1])
        distances.append(dist)

    # recompute turning points with lower margins for adjacent turns that are still close to each other
    previous_close = False

    for i in range(len(turning_points)):
        if i < len(turning_points) - 1:
            dist = distances[i]
            if dist < 9 * road_width:
                next_close = True
            else:
                next_close = False
        else:
            next_close = False

        if previous_close or next_close:
            j = tp_indices[i]  # index tp in path

            if previous_close:
                dist_prev = distances[i - 1]
                s1, len_s1, s1_i_min, s1_i_max = get_segment(path, j, dist_prev / 2, dist_prev / 4, after=False)
                segment_data_tuples[i] = (s1, len_s1, s1_i_min, s1_i_max), segment_data_tuples[i][1]
            else:
                s1, len_s1, _, s1_i_max = segment_data_tuples[i][0]

            if next_close:
                dist_next = distances[i]
                s2, len_s2, s2_i_min, s2_i_max = get_segment(path, j, dist_next / 2, dist_next / 4, after=True)
                segment_data_tuples[i] = segment_data_tuples[i][0], (s2, len_s2, s2_i_min, s2_i_max)
                previous_close = True
            else:
                s2, len_s2, s2_i_min, _ = segment_data_tuples[i][1]
                previous_close = False

            line1 = fit_line(s1)
            line2 = fit_line(s2)

            view = False

            if view:
                plt.figure()

                px = [x for (x, y) in path]
                py = [y for (x, y) in path]
                plt.plot(px, py, c='g', linewidth=1)

                s1x = [x for (x, y) in s1]
                s1y = [y for (x, y) in s1]
                plt.plot(s1x, s1y, c='blue', linewidth=4)

                s2x = [x for (x, y) in s2]
                s2y = [y for (x, y) in s2]
                plt.plot(s2x, s2y, c='red', linewidth=4)

                pixels_line1 = line1.pixels()
                l1x = [x for (x, y) in pixels_line1]
                l1y = [y for (x, y) in pixels_line1]
                plt.plot(l1x, l1y, c='cyan', linewidth=2)

                pixels_line2 = line2.pixels()
                l2x = [x for (x, y) in pixels_line2]
                l2y = [y for (x, y) in pixels_line2]
                plt.plot(l2x, l2y, c='orange', linewidth=2)

                plt.scatter([path[j][0]], [path[j][1]], s=50, c='green')
                plt.scatter([turning_points[i][0]], [turning_points[i][1]], s=50, c='red')

            if line1 and line2:
                ip = line1.intersection(line2)
                if ip:
                    ip = int(round(ip[0])), int(round(ip[1]))
                    cp = closest_point(ip, path[s1_i_max:s2_i_min])
                    line = Line(cp, ip)
                    tp = line.point_on_max_dist_from_p1(1.42 * road_width)
                    tp = min(tp[0], max_x), min(tp[1], max_y)
                    turning_points[i] = tp

                    if view:
                        plt.scatter([tp[0]], [tp[1]], s=50, c='blue')
                        plt.show()

    # insert sharp turns in the path, connect all adjacent turning points that are still close to each other
    path_new = []
    tp_indices = []
    previous_close = False
    n = 0

    for i in range(len(turning_points)):
        tp = turning_points[i]
        _, _, _, s1_i_max = segment_data_tuples[i][0]
        _, _, s2_i_min, _ = segment_data_tuples[i][1]

        if i < len(turning_points) - 1:
            dist = distances[i]
            if dist < 9 * road_width:
                next_close = True
            else:
                next_close = False
        else:
            next_close = False

        if previous_close:
            if next_close:
                turn = fill([], turning_points[i - 1], tp)[1:]
                previous_close = True
            else:
                turn = fill([tp], turning_points[i - 1], path[s2_i_min])[1:]
                previous_close = False
        else:
            path_new.extend(path[n:s1_i_max])
            if next_close:
                turn = fill([], path[s1_i_max], tp)
                previous_close = True
            else:
                turn = fill([tp], path[s1_i_max], path[s2_i_min])
                previous_close = False

        path_new.extend(turn)
        tp_indices.append(path_new.index(tp))
        n = s2_i_min + 1

    path_new.extend(path[n:])

    return path_new, tp_indices
