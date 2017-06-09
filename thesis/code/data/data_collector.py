import os
import numpy as np
import PIL.Image as im
from skimage import img_as_float
import data.image_collector as image_collector
import data.road_collector as road_collector
from utils.image import Image
from utils.distance import euclidean_distance, meter_per_pixel

path_road_data = '' # TODO
path_images = '' # TODO
zoom_level = 16 # for Bing API: enter zoom level - 1
size = 1024
image_source = 'bing'  # 'mapbox', 'bing' or 'google'
min_length = 250  # minimum length a test segment should have in meters
save_complete = False
save_reference = False

if __name__ == '__main__':
    if image_source == 'mapbox':
        extension = '.jpg'
    elif image_source == 'bing':
        extension = '.png'
    elif image_source == 'google':
        extension = '.png'
    else:
        raise ValueError('Image source not supported!')

    # roads = osm.get_roads_from_osm_api(bbox)
    roads = road_collector.get_roads_from_xml_file(path_road_data)

    for road in roads:
        name = road.name.replace('_', '-').replace('/', '-')
        segments = road.segments

        if save_complete and len(segments) > 1:
            image_path = path_images + name + '_z' + str(zoom_level) + '_s' + str(size) + extension

            if not os.path.exists(image_path):
                img = image_collector.get_image(road.center(), zoom_level, size, image_source)

                if image_source == 'google':  # remove alpha channel
                    img_float = img_as_float(img)
                    img_new = np.zeros((len(img_float), len(img_float[0]), 3))
                    for r in range(0, len(img_float)):
                        for c in range(0, len(img_float[0])):
                            img_new[r][c] = img_float[r][c][:3]
                    img = im.fromarray((img_new * 255).astype(np.uint8))

                img.save(image_path)

                if save_reference:
                    image_path = image_path[:-4] + '_reference' + extension

                    if not os.path.exists(image_path):
                        img_float = img_as_float(img)

                        if image_source == 'google':  # remove alpha channel
                            img_new = np.zeros((len(img_float), len(img_float[0]), 3))
                            for r in range(0, len(img_float)):
                                for c in range(0, len(img_float[0])):
                                    img_new[r][c] = img_float[r][c][:3]
                            img_float = img_new

                        image = Image(img_float)

                        for pixel in road.pixels(size, zoom_level):
                            image.mark_pixel(pixel, radius=2)

                        img = im.fromarray((image.get() * 255).astype(np.uint8))
                        img.save(image_path)

        i = 0
        for segment in segments:
            i += 1
            image_path = path_images + name + '_' + str(i) + '_z' + str(zoom_level) + '_s' + str(size) + extension

            if not os.path.exists(image_path):
                mpp = meter_per_pixel(segment.center().lat, zoom_level)
                reference = segment.pixels(size, zoom_level=zoom_level)

                pixel_length_reference = 0
                previous_pixel = None
                for pixel in reference:
                    if previous_pixel is not None:
                        pixel_length_reference += euclidean_distance(previous_pixel, pixel)
                    previous_pixel = pixel

                if pixel_length_reference * mpp >= min_length:
                    img = image_collector.get_image(segment.center(), zoom_level, size, image_source)

                    if image_source == 'google':  # remove alpha channel
                        img_float = img_as_float(img)
                        img_new = np.zeros((len(img_float), len(img_float[0]), 3))
                        for r in range(0, len(img_float)):
                            for c in range(0, len(img_float[0])):
                                img_new[r][c] = img_float[r][c][:3]
                        img = im.fromarray((img_new * 255).astype(np.uint8))

                    img.save(image_path)

                    if save_reference:
                        image_path = image_path[:-4] + '_reference' + extension

                        if not os.path.exists(image_path):
                            img_float = img_as_float(img)

                            if image_source == 'google':  # remove alpha channel
                                img_new = np.zeros((len(img_float), len(img_float[0]), 3))
                                for r in range(0, len(img_float)):
                                    for c in range(0, len(img_float[0])):
                                        img_new[r][c] = img_float[r][c][:3]
                                img_float = img_new

                            image = Image(img_float)

                            for pixel in segment.pixels(size, zoom_level=zoom_level):
                                image.mark_pixel(pixel, radius=2)

                            img = im.fromarray((image.get() * 255).astype(np.uint8))
                            img.save(image_path)
