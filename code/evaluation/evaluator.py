from math import sqrt
from skimage.draw import circle
from utils.distance import closest_point


class Evaluation:
    def __init__(self, com, cor, cor_all, qua, qua_all, red, mean_dist, rmse):
        self.completeness = com
        self.correctness = cor
        self.correctness_all = cor_all
        self.quality = qua
        self.quality_all = qua_all
        self.redundancy = red
        self.mean_distance = mean_dist
        self.rmse = rmse


def evaluate(len_ref, len_matched_ref, len_ext, len_matched_ext, len_matched_ext_all, sum_d, sum_sq_d, n_d):
    com = len_matched_ref / len_ref
    cor = len_matched_ext / len_ext
    cor_all = len_matched_ext_all / len_ext
    len_unmatched_ref = len_ref - len_matched_ref
    qua = len_matched_ext / (len_ext + len_unmatched_ref)
    qua_all = len_matched_ext_all / (len_ext + len_unmatched_ref)
    if len_matched_ext == 0:
        red = 0
    else:
        red = (len_matched_ext - len_matched_ref) / len_matched_ext
    mean_dist = sum_d / n_d
    rmse = sqrt(sum_sq_d / n_d)
    return Evaluation(com, cor, cor_all, qua, qua_all, red, mean_dist, rmse)


def get_buffer(path, buffer_width):
    buffered_path = set(path)
    for x, y in path:
        rr, cc = circle(y, x, buffer_width)
        for r, c in zip(rr, cc):
            buffered_path.add((c, r))
    return buffered_path


def distance_to_neighbor(pixel, neighbor):
    dx = pixel[0] - neighbor[0]
    dy = pixel[1] - neighbor[1]
    if (not -1 <= dx <= 1) or (not -1 <= dy <= 1):
        return False
    else:
        return sqrt(dx * dx + dy * dy)


def get_match(path, buffer):
    len_total = 0
    len_matched = 0
    matched_path = []
    previous_pixel = None
    previous_pixel_correct = False
    for pixel in path:
        if pixel in buffer:
            pixel_correct = True
            matched_path.append(pixel)
        else:
            pixel_correct = False
        if previous_pixel is not None:
            length = distance_to_neighbor(previous_pixel, pixel)
            if length:
                len_total += length
                if previous_pixel_correct and pixel_correct:
                    len_matched += length
                elif previous_pixel_correct or pixel_correct:
                    len_matched += length / 2
        previous_pixel = pixel
        previous_pixel_correct = pixel_correct
    return len_matched, len_total, matched_path


class Evaluator:
    def __init__(self):
        self.total_len_ext = 0
        self.total_len_matched_ext = 0
        self.total_len_matched_ext_all = 0
        self.total_len_ref = 0
        self.total_len_matched_ref = 0
        self.total_sum_d = 0
        self.total_sum_sq_d = 0
        self.total_n_d = 0

    def evaluate_all(self):
        return evaluate(self.total_len_ref, self.total_len_matched_ref, self.total_len_ext, self.total_len_matched_ext,
                        self.total_len_matched_ext_all, self.total_sum_d, self.total_sum_sq_d, self.total_n_d)

    def evaluate(self, ext, ref, refs, buffer_width, mpp):
        buffered_ext = get_buffer(ext, buffer_width)
        buffered_ref = get_buffer(ref, buffer_width)
        buffered_refs = get_buffer(refs, buffer_width)

        len_matched_ext, len_ext, matched_ext = get_match(ext, buffered_ref)
        len_matched_ext_all, _, _ = get_match(ext, buffered_refs)
        len_matched_ref, len_ref, matched_ref = get_match(ref, buffered_ext)

        sd = 0
        ssd = 0
        nd = 0
        for p in matched_ext:
            cp = closest_point(p, matched_ref)
            dx = (p[0] - cp[0]) * mpp
            dy = (p[1] - cp[1]) * mpp
            squared_distance = dx * dx + dy * dy
            ssd += squared_distance
            distance = sqrt(squared_distance)
            sd += distance
            nd += 1

        len_ext *= mpp
        len_matched_ext *= mpp
        len_matched_ext_all *= mpp
        len_ref *= mpp
        len_matched_ref *= mpp

        evaluation = evaluate(len_ref, len_matched_ref, len_ext, len_matched_ext, len_matched_ext_all, sd, ssd, nd)

        self.total_len_ext += len_ext
        self.total_len_matched_ext += len_matched_ext
        self.total_len_matched_ext_all += len_matched_ext_all
        self.total_len_ref += len_ref
        self.total_len_matched_ref += len_matched_ref
        self.total_sum_d += sd
        self.total_sum_sq_d += ssd
        self.total_n_d += nd

        return evaluation, matched_ext
