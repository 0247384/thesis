from utils.line import fill
from utils.distance import euclidean_distance, indices_closest_segments


class Extraction:
    def __init__(self):
        self.points = []
        self.correct_points = []

    def get_last(self):
        if len(self.points) > 0:
            return self.points[-1]
        else:
            return False

    def get_pixels(self):
        if len(self.points) > 1:
            return fill(self.points, self.points[0], self.points[-1])
        else:
            return self.points

    def get_segments(self):
        if len(self.points) < 2 or len(self.correct_points) < 2:
            return []
        else:
            segments = []
            pixels = self.get_pixels()
            segment = [pixels[0]]
            i_cp = 1
            for pixel in pixels[1:]:
                segment.append(pixel)
                if pixel == self.correct_points[i_cp]:
                    segments.append(segment)
                    segment = [pixel]
                    i_cp += 1
            return segments

    # returns correct point pair which contains the closest point,
    # if multiple: the one with the closest correct points
    def get_closest_correct_point_pair(self, point):
        segments = self.get_segments()
        if len(segments) == 0:
            return False
        elif len(segments) == 1:
            return segments[0][0], segments[0][-1]
        else:
            indices = indices_closest_segments(point, segments)
            if len(indices) == 1:
                segment = segments[indices[0]]
                return segment[0], segment[-1]
            else:
                pair = False
                min_dist = False
                for i in indices:
                    segment = segments[i]
                    cp1 = segment[0]
                    cp2 = segment[-1]
                    dist = euclidean_distance(cp1, point) + euclidean_distance(cp2, point)
                    if not min_dist or dist < min_dist:
                        pair = cp1, cp2
                        min_dist = dist
                return pair

    def extend(self, points, correct_point):
        if len(self.points) > 0:
            self.points = self.points[:-1]
        self.points.extend(points)
        self.correct_points.append(correct_point)

    def correct_segment(self, points, start, new_point, end):
        i_start = -1
        i_end = -1
        i_cp = 0
        for i in range(len(self.points)):
            if self.points[i] == self.correct_points[i_cp]:
                if self.points[i] == start:
                    i_start = i
                elif i_start > -1 and self.points[i] == end:
                    i_end = i
                    break
                else:
                    i_start = -1
                    i_end = -1
                i_cp += 1
        if i_start > -1 and i_end > -1:
            self.points = self.points[:i_start] + points + self.points[i_end + 1:]
            self.correct_points.insert(i_cp, new_point)

    def correct_point(self, point, new_point):
        if point == new_point:
            return False
        elif point in self.points:
            i_cp = 0
            for i in range(len(self.points)):
                if self.points[i] == self.correct_points[i_cp]:
                    if self.points[i] == point:
                        self.points[i] = new_point
                        self.correct_points[i_cp] = new_point
                    i_cp += 1
                else:
                    if self.points[i] == point:
                        self.points[i] = new_point
                        self.correct_points.insert(i_cp, new_point)
                        i_cp += 1
            return True
        else:
            return False

    def insert_point(self, point, index, correct=True):
        self.points.insert(index, point)
        if correct:
            i_cp = 0
            for i in range(index):
                if self.points[i] == self.correct_points[i_cp]:
                    i_cp += 1
            self.correct_points.insert(i_cp, point)

    def remove_point(self, point):
        if point in self.points:
            while point in self.points:
                self.points.remove(point)
            while point in self.correct_points:
                self.correct_points.remove(point)
            if len(self.points) == 1:
                self.correct_points = list(self.points)
            if len(self.points) > 1:
                first_point = self.points[0]
                last_point = self.points[-1]
                if len(self.correct_points) == 0:
                    self.correct_points = [first_point, last_point]
                if len(self.correct_points) > 0:
                    first_correct_point = self.correct_points[0]
                    if first_correct_point != first_point:
                        self.correct_points.insert(0, first_point)
                    last_correct_point = self.correct_points[-1]
                    if last_correct_point != last_point:
                        self.correct_points.append(last_point)
            return True
        else:
            return False

    def reset(self):
        self.points = []
        self.correct_points = []
