from mainui.draw import draw_plot_marked
from mainui.error import errormessage
import numpy as np
import pyqtgraph as pg

"""
removes highlight from previous selected points in the MDS graph
"""


def clear_plot_marked(graphMarked, view):
    for superpixel in graphMarked:
        view.getPlotItem().dataItems[0].points()[superpixel].setBrush(pg.mkBrush(100, 100, 150, 255))
    graphMarked.clear()


"""
Finds the point in the plot from the selected superpixel and  highlights it
"""


def on_click_superpixel(graphMarked, view, algs, image_path, superImage, foreground, background, x, y):
    try:
        clear_plot_marked(graphMarked, view)
        superpixel = algs.images_segmented[image_path][y][x]
        view.getPlotItem().dataItems[0].points()[superpixel].setBrush(pg.mkBrush(0, 255, 0, 255))
        graphMarked.append(superpixel)
        draw_plot_marked(image_path, superImage, algs, foreground, background,
                         graphMarked)
    except AttributeError:
        errormessage("No plot", "No plot has been  drawn")


"""
Finds the superpixel in the image from the selected point in mds and  highlights it
"""


def on_click_plot(graphMarked, view, mdsData, image_path, superImage, algs, foreground, background, _, points):
    clear_plot_marked(graphMarked, view)
    for point in points:
        graphMarked.append(np.where([point.pos()[0], point.pos()[1]] == mdsData)[0][0])
        point.setBrush(pg.mkBrush(0, 255, 0, 255))
    draw_plot_marked(image_path, superImage, algs, foreground, background,
                     graphMarked)
