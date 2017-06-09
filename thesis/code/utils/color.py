# Source: Colour Ramping for Data Visualisation, paulbourke.net/texture_colour/colourspace/
def map_to_color(v, vmin, vmax):
    r, g, b = 1., 1., 1.

    if v < vmin:
        v = vmin
    if v > vmax:
        v = vmax
    dv = vmax - vmin

    if v < (vmin + 0.25 * dv):
        r = 0.
        g = 4. * (v - vmin) / dv
    elif v < (vmin + 0.5 * dv):
        r = 0.
        b = 1. + 4. * (vmin + 0.25 * dv - v) / dv
    elif v < (vmin + 0.75 * dv):
        r = 4. * (v - vmin - 0.5 * dv) / dv
        b = 0.
    else:
        g = 1. + 4. * (vmin + 0.75 * dv - v) / dv
        b = 0.

    return r, g, b
