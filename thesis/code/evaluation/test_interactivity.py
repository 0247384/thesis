import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.io import imread
from skimage.draw import circle
from mapping.mapping_tool import MappingTool
from data.road_collector import get_roads_from_xml_file
from evaluation.evaluator import Evaluator
from utils.distance import meter_per_pixel

buffer_width = 4.5
realtime = False  # True, False or None
mapping_style = 'semi-automatic'  # 'semi-automatic, 'manual' or None
save_path = '' # TODO
path_images = save_path  # + 'Images/'
path_roads = '' # TODO
mpl.rcParams['toolbar'] = 'None'


def close():
    plt.close()


def start_mapping(img, img_ref, mapping_style):
    fig = plt.figure()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0.01)

    # maximize window, only works for TkAgg backend
    fm = plt.get_current_fig_manager()
    fm.window.state('zoomed')

    ax = plt.gca()
    ax.axis('off')
    im = ax.imshow(img, interpolation='kaiser')

    mapping_tool = MappingTool(fig, ax, im, img, img_ref, mapping_style, rt_proposals=realtime, close=close)
    plt.show()

    return mapping_tool


def save_results(name, statistics, evaluation, extractions=None, mapping_style=None, extra_text=None):
    if mapping_style is not None:
        if mapping_style == 'semi-automatic' and realtime:
            file = open(save_path + name + '_' + mapping_style + '_real-time.txt', 'w')
        else:
            file = open(save_path + name + '_' + mapping_style + '.txt', 'w')
    else:
        file = open(save_path + name + '.txt', 'w')

    file.write('------------\n')
    file.write(name + '\n')
    file.write('------------\n')

    if mapping_style is not None:
        file.write('Mapping style: ' + mapping_style + '\n')
        file.write('Real-time: ' + str(realtime) + '\n')
        file.write('------------\n')

    if extra_text is not None:
        file.write(extra_text)
        file.write('------------\n')

    for k, v in statistics.items():
        file.write(k + ': ' + str(v) + '\n')
    file.write('------------\n')

    file.write('Completeness: {} %\n'.format(round(100 * evaluation.completeness, 2)))
    file.write('Correctness: {} %\n'.format(round(100 * evaluation.correctness, 2)))
    file.write('Quality: {} %\n'.format(round(100 * evaluation.quality, 2)))
    file.write('Redundancy: {} %\n'.format(round(100 * evaluation.redundancy, 2)))
    file.write('Mean distance: {} m\n'.format(round(evaluation.mean_distance, 2)))
    file.write('RMSE: {} m\n'.format(round(evaluation.rmse, 2)))
    file.write('------------\n')

    if extractions is not None:
        for i, extraction in enumerate(extractions):
            points = str(extraction.points)
            correct_points = str(extraction.correct_points)
            file.write('Segment ' + str(i + 1) + '\n')
            file.write('Points: ' + points + '\n')
            file.write('Given points: ' + correct_points + '\n')
            file.write('------------\n')

    file.close()


stop = False
roads = get_roads_from_xml_file(path_roads)
evaluator = Evaluator()
total_statistics = {}
total_len_extracted_previous = 0
total_len_reference_previous = 0
evaluation_count = 0

for image_name in os.listdir(path_images):
    if not stop and image_name[-3:] == 'png':
        image_name_list = image_name.split('_')

        if len(image_name_list) < 3:
            continue

        road_name = image_name_list[0]
        segment_number = int(image_name_list[1])
        zoom_level = int(image_name_list[-2][1:])
        last = image_name_list[-1]

        if last[0:1] == 'c':
            crop_size = int(image_name_list[-1][1:-4])
        else:
            crop_size = 0

        current_road = None
        for road in roads:
            if road.name == road_name:
                current_road = road
                center = road.segments[segment_number - 1].center()
                break

        if current_road is None or center is None:
            print('Unknown road')
            continue

        img = img_as_float(imread(path_images + image_name))
        if crop_size > 0:
            img = img[crop_size:-crop_size, crop_size:-crop_size]
        # shows the reference, so a user can check whether he's finished
        img_ref = np.matrix.copy(img)
        size = len(img)

        points_ref = []
        for road in roads:
            road_pixels = road.pixels(size, zoom_level, center=center)
            road_pixels_in_image = []
            for (x, y) in road_pixels:
                if buffer_width < x < (size - buffer_width) and buffer_width < y < (size - buffer_width):
                    road_pixels_in_image.append((x, y))
            if len(road_pixels_in_image) > 40:
                points_ref.extend(road_pixels_in_image)

        buffered_points_ref = set()
        for x, y in points_ref:
            rr, cc = circle(y, x, 4.5, (size, size))
            for r, c in zip(rr, cc):
                buffered_points_ref.add((c, r))

        for x, y in buffered_points_ref:
            img_ref[y][x] = (0, 0.7, 0)

        mapping_tool = start_mapping(img, img_ref, mapping_style)
        statistics = mapping_tool.get_statistics()
        extractions = mapping_tool.extractions

        points = []
        for extraction in extractions:
            points.extend(extraction.get_pixels())

        if len(points) > 0:
            evaluation_count += 1
            mpp = meter_per_pixel(center.lat, zoom_level)
            evaluation, _ = evaluator.evaluate(points, points_ref, points_ref, buffer_width, mpp)

            total_len_extracted = evaluator.total_len_ext - total_len_extracted_previous
            total_len_reference = evaluator.total_len_ref - total_len_reference_previous
            extra = 'Total distance extracted: {} km\n'.format(round(total_len_extracted / 1000, 2))
            extra += 'Total distance reference: {} km\n'.format(round(total_len_reference / 1000, 2))
            total_len_extracted_previous = evaluator.total_len_ext
            total_len_reference_previous = evaluator.total_len_ref

            save_results(road_name, statistics, evaluation, extractions, mapping_style, extra)

            for k, v in statistics.items():
                if k in total_statistics.keys():
                    total_statistics[k] += v
                else:
                    total_statistics[k] = v

        stop = mapping_tool.stop

if evaluation_count > 1:
    evaluation = evaluator.evaluate_all()

    total_len_extracted = evaluator.total_len_ext
    total_len_reference = evaluator.total_len_ref
    extra = 'Total distance extracted: {} km\n'.format(round(total_len_extracted / 1000, 2))
    extra += 'Total distance reference: {} km\n'.format(round(total_len_reference / 1000, 2))

    save_results('Total', total_statistics, evaluation, extra_text=extra)
