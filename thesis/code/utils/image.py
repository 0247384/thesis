from skimage.draw import circle


def get_neighbors(x, y, max_x, max_y):
    neighbors = []

    if x > 0:
        neighbors.append((x - 1, y))
        if y > 0:
            neighbors.append((x - 1, y - 1))
        if y < max_y:
            neighbors.append((x - 1, y + 1))

    if y > 0:
        neighbors.append((x, y - 1))
        if x < max_x:
            neighbors.append((x + 1, y - 1))

    if x < max_x:
        neighbors.append((x + 1, y))
        if y < max_y:
            neighbors.append((x + 1, y + 1))

    if y < max_y:
        neighbors.append((x, y + 1))

    return neighbors


def mark_pixel(img, pixel, color=(1, 1, 0), radius=1):
    rr, cc = circle(pixel[1], pixel[0], radius, img.shape[0:2])
    img[rr, cc] = color


class Image:
    def __init__(self, img):
        self.img = img

    def get(self):
        return self.img

    def set(self, img):
        self.img = img

    def get_neighbors(self, pixel):
        x, y = pixel
        max_x = self.img.shape[1] - 1
        max_y = self.img.shape[0] - 1
        return get_neighbors(x, y, max_x, max_y)

    def mark_pixel(self, pixel, color=(1, 1, 0), radius=1):
        mark_pixel(self.img, pixel, color, radius)
