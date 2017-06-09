import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from skimage.io import imread
from skimage import img_as_float
from skimage.draw import circle
from road_extraction.road_extractor import extract_road
from utils.image import Image
from utils.color import map_to_color

path = '' # TODO
img = img_as_float(imread(path))
img_marked = np.matrix.copy(img)

input = []


def onclick(event):
    if event.xdata is None or event.ydata is None:
        return

    ix, iy = int(round(event.xdata)), int(round(event.ydata))

    if ix < 0:
        ix = 0
    if iy < 0:
        iy = 0

    input.append((ix, iy))

    rr, cc = circle(iy, ix, 3.5, img_marked.shape[0:2])
    img_marked[rr, cc] = (1, 1, 0)

    implot.set_data(img_marked)
    fig.canvas.draw()


plt.figure()
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=0.99, bottom=0.01)
ax = plt.gca()
implot = ax.imshow(img_marked, interpolation='kaiser')
fig = plt.gcf()
cid = fig.canvas.mpl_connect('button_press_event', onclick)
fm = plt.get_current_fig_manager()
# only works for TkAgg backend
fm.window.state('zoomed')
plt.show()

start = input[0][0], input[0][1]
goal = input[1][0], input[1][1]
smoothed_extraction, points, extraction, cost_map = extract_road(img, start, goal)

image = Image(img)

costs_extraction = []
for pixel in extraction:
    costs_extraction.append((pixel, cost_map[pixel[1]][pixel[0]]))
    # cost = cost_map[pixel[1]][pixel[0]]
    # color = map_to_color(cost, 0, 1)
    # image.mark_pixel(pixel, color, radius=1.5)
    # image.mark_pixel(pixel, (1, 0, 0), radius=1.5)

xe = [item[0][0] for item in costs_extraction]
ye = [item[0][1] for item in costs_extraction]
ce = [item[1] for item in costs_extraction]

costs_smoothed_extraction = []
for pixel in smoothed_extraction:
    costs_smoothed_extraction.append((pixel, cost_map[pixel[1]][pixel[0]]))
    # cost = cost_map[pixel[1]][pixel[0]]
    # color = map_to_color(cost, 0, 1)
    # image.mark_pixel(pixel, color, radius=1.5)
    image.mark_pixel(pixel, (1, 1, 0), radius=1.5)

xse = [item[0][0] for item in costs_smoothed_extraction]
yse = [item[0][1] for item in costs_smoothed_extraction]
cse = [item[1] for item in costs_smoothed_extraction]

fig, ax = plt.subplots(1, 1)
fig.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.95)
# ax[0].set_title('Cost map')
# ax[0].imshow(cost_map, cmap='jet', norm=colors.LogNorm())
# ax[1].set_title('Extracted road')
ax.imshow(image.get(), interpolation='kaiser')
# ax[2].set_title('Cost distribution')
# ax[2] = fig.add_subplot(133, projection='3d')
# ax[2].set_xlim(0, len(img[0]))
# ax[2].set_ylim(len(img), 0)
# ax[2].set_zlim(0, 1)
# ax[2].set_xlabel('X')
# ax[2].set_ylabel('Y')
# ax[2].set_zlabel('Cost')
# ax[2].plot(xe, ye, ce, 'r-')
# ax[2].plot(xse, yse, cse, 'b-')
# for a in ax:
#    a.set_xticks(())
#    a.set_yticks(())
fm = plt.get_current_fig_manager()
# only works for TkAgg backend
fm.window.state('zoomed')
plt.show()
