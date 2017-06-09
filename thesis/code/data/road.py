from math import pi, cos
from utils.line import line_to_pixels


def center(nodes):
    max_lat = -90
    min_lat = 90
    max_lon = -180
    min_lon = 180

    for node in nodes:
        lat = node.lat
        lon = node.lon

        if lat > max_lat:
            max_lat = lat
        if lat < min_lat:
            min_lat = lat
        if lon > max_lon:
            max_lon = lon
        if lon < min_lon:
            min_lon = lon

    lat = (max_lat + min_lat) / 2.0
    lon = (max_lon + min_lon) / 2.0

    return Node(lat, lon)


def degrees_per_pixel(lat, zoom_level):
    lat_rad = lat * pi / 180
    scale_factor = cos(lat_rad)

    deg_per_map_width = 360. / 2 ** zoom_level
    deg_per_pixel_width = deg_per_map_width / 512
    deg_per_map_height = scale_factor * 360. / 2 ** zoom_level
    deg_per_pixel_height = deg_per_map_height / 512

    return deg_per_pixel_width, deg_per_pixel_height


class Road:
    def __init__(self, name, segments):
        self.name = name
        self.segments = segments

    def center(self):
        all_nodes = set()

        for segment in self.segments:
            all_nodes.update(segment.nodes)

        return center(all_nodes)

    def pixels(self, size, zoom_level, center=None):
        pixels = []
        if center is None:
            center = self.center()
        deg_pixel_w, deg_pixel_h = degrees_per_pixel(center.lat, zoom_level)

        for segment in self.segments:
            pixels.extend(segment.pixels(size, center=center, deg_pixel_w=deg_pixel_w, deg_pixel_h=deg_pixel_h))

        return pixels


class Segment:
    def __init__(self, name, nodes):
        self.name = name
        self.nodes = nodes

    def center(self):
        return center(self.nodes)

    # either zoom_level OR (center AND deg_per_pixel_width AND deg_per_pixel_height) must be given
    def pixels(self, size, zoom_level=None, center=None, deg_pixel_w=None, deg_pixel_h=None, i_start=None, i_end=None):
        pixels = []
        node_pixels = []

        if zoom_level is not None:
            if center is None:
                center = self.center()
            deg_pixel_w, deg_pixel_h = degrees_per_pixel(center.lat, zoom_level)

        if i_start is None:
            i_start = 0
        if i_end is None:
            i_end = len(self.nodes) - 1

        for i in range(i_start, i_end + 1):
            pixel = self.nodes[i].pixel(size, center=center, deg_pixel_w=deg_pixel_w, deg_pixel_h=deg_pixel_h)
            node_pixels.append(pixel)

        for i in range(0, len(node_pixels) - 1):
            if len(pixels) > 0:
                del pixels[-1]
            line_pixels = line_to_pixels(node_pixels[i], node_pixels[i + 1])
            pixels.extend(line_pixels)

        return pixels


class Node:
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon

    # either zoom_level OR (deg_per_pixel_width AND deg_per_pixel_height) must be given
    def pixel(self, size, center, zoom_level=None, deg_pixel_w=None, deg_pixel_h=None):
        if zoom_level is not None:
            deg_pixel_w, deg_pixel_h = degrees_per_pixel(center.lat, zoom_level)

        d_lat = self.lat - center.lat
        d_lon = self.lon - center.lon

        vx = d_lon / deg_pixel_w
        vy = d_lat / deg_pixel_h

        x = vx + (size - 1) / 2
        y = -vy + (size - 1) / 2

        return round(x), round(y)
