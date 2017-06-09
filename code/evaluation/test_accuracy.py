import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from PIL import Image as PILImage
from skimage import img_as_float
from skimage.io import imread
from data.road_collector import get_roads_from_xml_file
from evaluation.evaluator import Evaluator
from road_extraction.road_extractor import extract_road
from utils.image import Image
from utils.distance import euclidean_distance, meter_per_pixel
from utils.color import map_to_color
from utils.line import fill

path_roads = '' # TODO
path_images = '' # TODO
zoom_level = 16 # for Bing API: enter zoom level - 1
buffer_width = 4.5  # = road width in pixels / 2, depends on the zoom level
size = 1024
extension = '.png'

min_length = 250  # minimum length a test segment should have in meters
surrounding_roads_as_seeds = True
straight_line_baseline = False
save_init = False
save_cost_map = False
save_reference = False
save_extracted = False
show_results = False
show_curves = False
print_intermediate_averages = True
intermediate_avg_interval = 20


def print_abort(message):
    print('Road extraction aborted: ' + message)


roads = get_roads_from_xml_file(path_roads)
evaluator = Evaluator()
segments_tested = 0
segments_total = 0
start_time_total = time.time()
total_extraction_time = 0

for road in roads:
    i = 0
    road_name = road.name.replace('_', '-').replace('/', '-')

    for segment in road.segments:
        i += 1
        segments_total += 1

        print('------------')
        print(road_name + ' ' + str(i))
        print('------------')

        image_path = path_images + road_name + '_' + str(i) + '_z' + str(zoom_level) + '_s' + str(size) + extension
        if not os.path.exists(image_path):
            print_abort('bad test sample (road too short, contains a loop, outdated data, ...)')
            continue

        img = img_as_float(imread(image_path))
        center = segment.center()
        nodes = segment.nodes
        index_start = 0
        index_goal = len(nodes) - 1
        start = nodes[index_start].pixel(size, center, zoom_level=zoom_level)
        goal = nodes[index_goal].pixel(size, center, zoom_level=zoom_level)
        abort = False

        # print('Start: {}'.format(start))
        while start[0] < 0 or start[1] < 0 or start[0] >= len(img[0]) or start[1] >= len(img):
            index_start += 1
            if index_start >= len(nodes) - 1:
                abort = True
                print_abort('no start node within the image')
                break
            start = nodes[index_start].pixel(size, center, zoom_level=zoom_level)
            # print('Start: {}'.format(start))

        if abort:
            # os.remove(image_path)
            # print('Test sample removed')
            continue

        # print('Goal: {}'.format(goal))
        while goal[0] < 0 or goal[1] < 0 or goal[0] >= len(img[0]) or goal[1] >= len(img):
            index_goal -= 1
            if index_goal <= 0:
                abort = True
                print_abort('no end node within the image')
                break
            goal = nodes[index_goal].pixel(size, center, zoom_level=zoom_level)
            # print('Goal: {}'.format(goal))

        if abort:
            # os.remove(image_path)
            # print('Test sample removed')
            continue
        elif start == goal:
            print_abort('start position = end position')
            # os.remove(image_path)
            # print('Test sample removed')
            continue

        mpp = meter_per_pixel(center.lat, zoom_level)
        reference = segment.pixels(size, zoom_level=zoom_level, i_start=index_start, i_end=index_goal)

        pixel_length_reference = 0
        previous_pixel = None
        for pixel in reference:
            if previous_pixel is not None:
                pixel_length_reference += euclidean_distance(previous_pixel, pixel)
            previous_pixel = pixel

        if pixel_length_reference * mpp < min_length:
            print_abort('road segment too short')
            # os.remove(image_path)
            # print('Test sample removed')
            continue

        node_tuples = []
        for n in range(index_start, index_goal + 1):
            node_tuples.append((nodes[n].lat, nodes[n].lon))

        node_tuples_set = set(node_tuples)
        if len(node_tuples_set) < len(node_tuples):
            print_abort('road segment contains a loop')
            # os.remove(image_path)
            # print('Test sample removed')
            continue

        image = Image(img)
        img_ref = np.matrix.copy(img)

        if save_init:
            image.mark_pixel(start, radius=5.5)
            image.mark_pixel(goal, radius=5.5)
            result = PILImage.fromarray((image.get() * 255).astype(np.uint8))
            result.save(image_path[:-4] + '_init' + extension)
            continue

        if straight_line_baseline:
            start_time = time.time()
            extraction = fill([], start, goal)
            smoothed_extraction = extraction
            cost_map = np.zeros((len(img), len(img[0])))
            total_extraction_time += time.time() - start_time
        else:
            seed_colors = None
            if surrounding_roads_as_seeds:
                seed_colors = []
                for road_ref in roads:
                    if road_ref != road:
                        pixels = road_ref.pixels(size, zoom_level, center)
                        for x, y in pixels:
                            if 0 <= x < len(img[0]) and 0 <= y < len(img):
                                seed_colors.append(img[y][x])

            start_time = time.time()
            smoothed_extraction, points, extraction, cost_map = extract_road(img, start, goal, seed_colors)
            total_extraction_time += time.time() - start_time

            if show_curves:
                ex = [p[0] for p in extraction]
                ey = [p[1] for p in extraction]
                sx = [p[0] for p in smoothed_extraction]
                sy = [p[1] for p in smoothed_extraction]
                rx = [p[0] for p in reference]
                ry = [p[1] for p in reference]

                fig, ax = plt.subplots()
                ax.plot(ex, ey, 'r-')
                ax.plot(sx, sy, 'b-')
                ax.plot(rx, ry, 'g-')
                plt.show()

        all_references = set()
        for road_ref in roads:
            input = road_ref.pixels(size, zoom_level, center=center)
            for x, y in input:
                if 0 <= x < len(img[0]) and 0 <= y < len(img):
                    all_references.add((x, y))

        evaluation, matched_ext = evaluator.evaluate(smoothed_extraction, reference, all_references, buffer_width, mpp)

        print('------------')
        print('Completeness: {} %'.format(round(100 * evaluation.completeness, 2)))
        print('Correctness: {} %'.format(round(100 * evaluation.correctness, 2)))
        print('Correctness*: {} %'.format(round(100 * evaluation.correctness_all, 2)))
        print('Quality: {} %'.format(round(100 * evaluation.quality, 2)))
        print('Quality*: {} %'.format(round(100 * evaluation.quality_all, 2)))
        print('Redundancy: {} %'.format(round(100 * evaluation.redundancy, 2)))
        print('Mean distance: {} m'.format(round(evaluation.mean_distance, 2)))
        print('RMSE: {} m'.format(round(evaluation.rmse, 2)))
        print('------------')

        segments_tested += 1
        if print_intermediate_averages and segments_tested % intermediate_avg_interval == 0:
            eval_total = evaluator.evaluate_all()
            print('------------')
            print('Segments tested: {}'.format(segments_tested))
            print('Completeness: {} %'.format(round(100 * eval_total.completeness, 2)))
            print('Correctness: {} %'.format(round(100 * eval_total.correctness, 2)))
            print('Correctness*: {} %'.format(round(100 * eval_total.correctness_all, 2)))
            print('Quality: {} %'.format(round(100 * eval_total.quality, 2)))
            print('Quality*: {} %'.format(round(100 * eval_total.quality_all, 2)))
            print('Redundancy: {} %'.format(round(100 * eval_total.redundancy, 2)))
            print('Mean distance: {} m'.format(round(eval_total.mean_distance, 2)))
            print('RMSE: {} m'.format(round(eval_total.rmse, 2)))
            print('------------')
            print('Total: {} seconds'.format(round((time.time() - start_time_total), 2)))
            print('------------')

        if save_reference or show_results:
            image.set(img_ref)
            for pixel in reference:
                image.mark_pixel(pixel, radius=1.5)

        if save_extracted or show_results:
            image.set(img)
            # for pixel in extraction:
            #     image.mark_pixel(pixel, color=(1, 1, 0), radius=1.5)
            # for pixel in smoothed_extraction:
            #     cost = cost_map[pixel[1]][pixel[0]]
            #     color = map_to_color(cost, 0, 1)
            #     image.mark_pixel(pixel, color, radius=1.5)
            for pixel in smoothed_extraction:
                image.mark_pixel(pixel, color=(1, 1, 0), radius=1.5)
            #for pixel in matched_ext:
            #    image.mark_pixel(pixel, color=(0, 1, 0), radius=1.5)

        if save_cost_map:
            result = PILImage.fromarray((cost_map * 255).astype(np.uint8))
            result.save(image_path[:-4] + '_cost_map' + extension)

        if save_reference:
            result = PILImage.fromarray((img_ref * 255).astype(np.uint8))
            result.save(image_path[:-4] + '_reference' + extension)

        if save_extracted:
            result = PILImage.fromarray((img * 255).astype(np.uint8))
            result.save(image_path[:-4] + '_extracted' + extension)

        if show_results:
            fig, ax = plt.subplots(1, 2)
            fig.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.95)
            ax[0].imshow(img_ref, interpolation='kaiser')
            ax[0].set_title('Reference')
            # ax[0].imshow(cost_map, cmap='jet', norm=colors.LogNorm())
            # ax[0].set_title('Cost map')
            ax[1].imshow(img, interpolation='kaiser')
            ax[1].set_title('Extracted')
            for a in ax:
                a.set_xticks(())
                a.set_yticks(())
            fm = plt.get_current_fig_manager()
            # only works for TkAgg backend
            fm.window.state('zoomed')
            plt.show()

if segments_tested > 0:
    eval_total = evaluator.evaluate_all()
    total_len_extracted = evaluator.total_len_ext
    total_len_reference = evaluator.total_len_ref

    print('------------')
    print('Segments tested: {}'.format(segments_tested))
    print('Segments discarded: {}'.format(segments_total - segments_tested))
    print('Total distance extracted: {} km'.format(round(total_len_extracted / 1000, 2)))
    print('Total distance reference: {} km'.format(round(total_len_reference / 1000, 2)))
    print('Buffer width: {} pixels'.format(buffer_width))
    print('------------')
    print('Completeness: {} %'.format(round(100 * eval_total.completeness, 2)))
    print('Correctness: {} %'.format(round(100 * eval_total.correctness, 2)))
    print('Correctness*: {} %'.format(round(100 * eval_total.correctness_all, 2)))
    print('Quality: {} %'.format(round(100 * eval_total.quality, 2)))
    print('Quality*: {} %'.format(round(100 * eval_total.quality_all, 2)))
    print('Redundancy: {} %'.format(round(100 * eval_total.redundancy, 2)))
    print('Mean distance: {} m'.format(round(eval_total.mean_distance, 2)))
    print('RMSE: {} m'.format(round(eval_total.rmse, 2)))
    print('------------')
    print('Average time per extraction: {} seconds'.format(round((total_extraction_time / segments_tested), 2)))
    print('Total extraction time: {} seconds'.format(round(total_extraction_time, 2)))
    print('Total time: {} seconds'.format(round((time.time() - start_time_total), 2)))
    print('------------')
    print('* using all roads in the image as reference instead of only the desired road')
