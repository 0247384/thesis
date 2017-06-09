import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.io import imread
from mapping.mapping_tool import MappingTool


path = '' # TODO


def close():
    plt.close()


def start_mapping():
    fig = plt.figure()
    ax = plt.gca()
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=0.99, bottom=0.01)

    # maximize window, only works for TkAgg backend
    fm = plt.get_current_fig_manager()
    fm.window.state('zoomed')

    img = img_as_float(imread(path))
    im = ax.imshow(img, interpolation='kaiser')

    mapping_tool = MappingTool(fig, ax, im, img, close=close)
    plt.show()

    return mapping_tool


mapping_tool = start_mapping()
