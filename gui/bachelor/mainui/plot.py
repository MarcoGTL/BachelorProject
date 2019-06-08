
import sys

import numpy as np
import pyqtgraph as pg


def create_graph():
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    n = 1000
    print('Number of points: ' + str(n))
    data = np.random.normal(size=(2, n))
    print(data)
    # Create the main application instance
    app = pg.mkQApp()

    # Create the view
    view = pg.PlotWidget()
    view.resize(800, 600)
    view.setWindowTitle('Scatter plot using pyqtgraph with PyQT5')
    view.setAspectLocked(True)
    view.show()

    # Create the scatter plot and add it to the view
    scatter = pg.ScatterPlotItem(pen=pg.mkPen(width=1, color='r'), symbol='o', size=5)
    scatter.sigClicked.connect(on_click_graph)
    view.addItem(scatter)

    # Convert data array into a list of dictionaries with the x,y-coordinates
    pos = [{'pos': data[:, i]} for i in range(n)]
    print(pos)

    now = pg.ptime.time()
    scatter.setData(pos)
    print("Plot time: {} sec".format(pg.ptime.time() - now))

    # Gracefully exit the application
    sys.exit(app.exec_())


def on_click_graph(self, points):
    for point in points:
        print(point.pos()[0], point.pos()[1])


if __name__ == '__main__':
    create_graph()