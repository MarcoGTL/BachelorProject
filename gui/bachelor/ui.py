from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore
from mainui import designer
from algorithms import pipeline
from algorithms import MDS
import os
import sys
import numpy as np
import pyqtgraph as pg
from skimage.segmentation import find_boundaries

# add heat map, eraser, mds with some way of linking

colors = [[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255],
          [255, 0, 255], [192, 192, 192], [128, 128, 128], [128, 0, 0], [128, 128, 0], [0, 128, 0], [128, 0, 128],
          [0, 128, 128], [0, 0, 128]]


class MyFileBrowser(designer.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self):
        super(MyFileBrowser, self).__init__()
        self.setupUi(self)

        self.treeView.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.treeView.customContextMenuRequested.connect(self.context_menu)

        self.listWidget.setViewMode(QtWidgets.QListView.IconMode)
        self.listWidget.setIconSize(QtCore.QSize(64, 64))
        self.listWidget.currentItemChanged.connect(self.choose_image)
        self.listWidget.setMovement(0)

        self.clearMarkingsButton.clicked.connect(self.clear_markings)
        self.histogramButton.clicked.connect(self.set_histograms)
        self.superpixelButton.clicked.connect(self.calculate_superpixels)
        self.graphcutButton.clicked.connect(self.compute_cosegmentation)
        self.kmeansButton.clicked.connect(self.kmeans)

        self.bwRadioButton.clicked.connect(self.draw_results)
        self.bRadioButton.clicked.connect(self.draw_results)

        self.featureSelected.currentIndexChanged.connect(self.change_features)
        self.clusteringBox.currentIndexChanged.connect(self.clustering_options)
        self.kmeansFrame.setHidden(True)

        self.fgRadioButton.clicked.connect(self.currentPencil)
        self.bgRadioButton.clicked.connect(self.currentPencil)


        self.mdsData = []
        self.point = (-1, -1)
        self.image_paths = []
        self.algs = pipeline.Pipeline([])
        self.pencil = 2
        self.relative_image_path = ""
        self.foreground = dict()
        self.background = dict()
        self.currentFolder = "images"
        self.selectedFolder = "something went wrong"
        self.img_float = []
        self.image_path = ""
        self.graphMarked = []

        self.change_features()
        self.populate()

    def populate(self):
        path = os.getcwd() + '/' + self.currentFolder
        self.model = QtWidgets.QFileSystemModel()
        self.model.setRootPath(QtCore.QDir.rootPath())
        self.treeView.setModel(self.model)
        self.treeView.setRootIndex(self.model.index(path))
        self.treeView.setSortingEnabled(True)

    def context_menu(self):
        menu = QtWidgets.QMenu()
        open = menu.addAction("select folder")
        open.triggered.connect(self.select_folder)
        cursor = QtGui.QCursor()
        menu.exec_(cursor.pos())

    def select_folder(self):
        self.listWidget.clear()
        self.image.clear()
        self.foreground.clear()
        self.background.clear()
        self.algs = pipeline.Pipeline([])
        index = self.treeView.currentIndex()
        file_path = self.model.filePath(index)
        self.selectedFolder = os.path.basename(file_path)
        print(file_path)
        print(self.currentFolder)
        print(self.selectedFolder)
        for file in os.listdir(file_path):
            if file[-4:] == ".png" or file[-4:] == ".jpg":
                self.image_paths.append(self.currentFolder + '/' + self.selectedFolder + '/' + file)
                item = QtWidgets.QListWidgetItem()
                icon = QtGui.QIcon()
                pixmap = QtGui.QPixmap(file_path + '/' + file)  # QImage object
                icon.addPixmap(pixmap, QtGui.QIcon.Normal, QtGui.QIcon.Off)
                item.setIcon(icon)
                item.setText(file)
                self.listWidget.addItem(item)
            if len(self.image_paths) != 0:
                self.algs = pipeline.Pipeline(self.image_paths)
                print(self.image_paths)

    def choose_image(self):
        if self.listWidget.currentItem() is not None:
            index = self.treeView.currentIndex()
            self.image_path = self.currentFolder + '/' + self.selectedFolder + '/' + self.listWidget.currentItem().text()
            print(self.image_path)
            pixmap = QtGui.QPixmap(self.image_path)
            self.image.setPixmap(pixmap)
            self.image.mousePressEvent = self.movStart
            self.image.mouseMoveEvent = self.mov
            self.image.mouseReleaseEvent = self.releasemov
            self.superImage.mousePressEvent = self.movStart
            self.superImage.mouseMoveEvent = self.mov
            self.superImage.mouseReleaseEvent = self.releasemov
            self.superpixelButton.setEnabled(True)
            if self.image_path not in self.foreground:
                self.foreground[self.image_path] = []
                self.background[self.image_path] = []
            else:
                self.draw_markings()
            self.draw_bounds()

            if self.image_path in self.algs.images_cosegmented:
                if self.clusteringBox.currentIndex() == 0:
                    self.draw_results()
                if self.clusteringBox.currentIndex() == 1:
                    self.draw_clusters()
                if self.clusteringBox.currentIndex() == 2:
                    self.draw_clusters()


        else:
            self.disable_buttons()

    def change_super_pixel(self):
        self.draw_bounds()

    def movStart(self, event):
        self.image.update()
        x = event.pos().x()
        y = event.pos().y() - round((self.image.height() - self.image.pixmap().height()) / 2)
        if y < 0 or x < 0:
            return

        self.savePoint(x, y)
        self.point = (x, y)
        self.draw_markings()

    def savePoint(self, x, y):
        if (x, y) in self.foreground[self.image_path]:
            print("hello")
            self.foreground[self.image_path].remove((x, y))
        if (x, y) in self.background:
            self.background[self.image_path].remove((x, y))

        if self.pencil == 1:
            self.background[self.image_path].append((x, y))
        if self.pencil == 2:
            self.foreground[self.image_path].append((x, y))

    def mov(self, event):
        self.image.update()
        x = event.pos().x()
        y = event.pos().y() - round((self.image.height() - self.image.pixmap().height()) / 2)

        if y < 0 or x < 0:
            return
        if self.point == (-1, -1) or self.point == (x, y):
            self.point = (x, y)
            self.savePoint(x, y)
            self.draw_markings()
            return
        if abs(self.point[0] - x) < abs(self.point[1] - y):
            if self.point[1] > y:
                self.save_liney(x, self.point[0], y, self.point[1])
            else:
                self.save_liney(self.point[0], x, self.point[1], y)
        else:
            if self.point[0] > x:
                self.save_linex(x, self.point[0], y, self.point[1])
            else:
                self.save_linex(self.point[0], x, self.point[1], y)
        self.point = (x, y)
        self.draw_markings()

    def releasemov(self, event):
        self.point = (-1, -1)

    def save_linex(self, x1, x2, y1, y2):
        slope = (y1 - y2) / (x1 - x2)
        i = x1 + 1
        while i <= x2:
            self.savePoint(i, round(y1 + slope * (i - x1)))
            i = i + 1

    def save_liney(self, x1, x2, y1, y2):
        slope = (x1 - x2) / (y1 - y2)
        i = y1 + 1
        while i <= y2:
            self.savePoint(round(x1 + slope * (i - y1)), i)
            i = i + 1

    def draw_markings(self):
        pixmap = QtGui.QPixmap(self.image_path)
        self.image.setPixmap(pixmap)
        qp = QtGui.QPainter(self.image.pixmap())
        qp.setPen(QtGui.QPen(QtCore.Qt.blue, 1))
        for x in self.background[self.image_path]:
            qp.drawPoint(x[0], x[1])

        qp.setPen(QtGui.QPen(QtCore.Qt.red, 1))
        for x in self.foreground[self.image_path]:
            qp.drawPoint(x[0], x[1])

        if self.image_path in self.algs.images_segmented:
            qpbounds = QtGui.QPainter(self.superImage.pixmap())
            qpbounds.setPen(QtGui.QPen(QtCore.Qt.blue, 1))
            for x in self.background[self.image_path]:
                qpbounds.drawPoint(x[0], x[1])

            qpbounds.setPen(QtGui.QPen(QtCore.Qt.red, 1))
            for x in self.foreground[self.image_path]:
                qpbounds.drawPoint(x[0], x[1])

        self.update()
        self.image.update()
        self.superImage.update()

    def currentPencil(self):
        if self.fgRadioButton.isChecked():
            self.pencil = 2
        elif self.bgRadioButton.isChecked():
            self.pencil = 1
        elif self.eRadioButton.isChecked():
            self.pencil = 0

    def calculate_superpixels(self):
        self.algs.compute_superpixels(self.superpixelSpinBox.value(), self.compactnessSpinBox.value(),
                                           self.iterationsSpinBox.value(), self.sigmaSpinBox.value())
        self.algs.compute_neighbors()
        self.algs.compute_centers()
        self.draw_bounds()
        self.disable_buttons()
        self.enable_buttons(2)

    def set_histograms(self):
        self.algs.compute_feature_vectors(means_bgr=self.RGB.isChecked(), means_hsv=self.HSV.isChecked(), h_hist=self.Hue.isChecked(), h_hist_bins=self.HHist.value(), h_hist_entropy=self.HentropyCheckBox.isChecked(), s_hist=self.Saturation.isChecked(), s_hist_bins=self.SHist.value(), s_hist_entropy=self.SentropyCheckBox.isChecked(), hs_hist=self.HueSat.isChecked(), hs_hist_bins_h=self.HSHHist.value(), hs_hist_bins_s=self.HSSHist.value(), sift=self.Sift.isChecked(), sift_kp_size=self.siftKeyPoint.value(), hog=self.Hog, hog_winSize=(self.winSize.value(), self.winSize.value()), hog_blockSize=(self.blockSize.value(), self.blockSize.value()), hog_blockStride=(self.blockStride.value(), self.blockStride.value()), hog_cellSize=(self.cellSize.value(), self.cellSize.value()))
        self.disable_buttons()
        self.enable_buttons(4)

    def set_markings(self):
        foreground = []
        background = []
        for x in self.foreground[self.image_path]:
            if self.algs.images_segmented[self.image_path][x[1]][x[0]] not in foreground:
                foreground.append(self.algs.images_segmented[self.image_path][x[1]][x[0]])
        for y in self.background[self.image_path]:
            if self.algs.images_segmented[self.image_path][y[1]][y[0]] not in self.background:
                background.append(self.algs.images_segmented[self.image_path][y[1]][y[0]])
        self.algs.set_fg_segments(self.image_path, foreground)
        self.algs.set_bg_segments(self.image_path, background)


    def clear_markings(self):
        self.foreground[self.image_path].clear()
        self.background[self.image_path].clear()
        self.draw_markings()
        self.image.update()
        self.superImage.update()
        self.result_image.update()

    def compute_cosegmentation(self):
        self.algs.perform_graph_cut()
        self.graph_cut()
        self.bwRadioButton.setEnabled(True)
        self.bRadioButton.setEnabled(True)

    def kmeans(self):
        self.algs.perform_k_means_clustering(num_clusters=self.kclustervalue.value())
        self.draw_clusters()

    def graph_cut(self):
        self.draw_results()

    def draw_clusters(self):
        pixmap = QtGui.QPixmap(self.image_path)

        self.result_image.setPixmap(pixmap)
        results = self.algs.images_cosegmented[self.image_path]
        qp = QtGui.QPainter(self.result_image.pixmap())
        i = 0
        j = 0
        lengthy = len(results)
        lengthx = len(results[0])
        print(lengthy, lengthx)
        while i < lengthy:
            while j < lengthx:
                qp.setPen(QtGui.QColor(colors[results[i][j]][0], colors[results[i][j]][1], colors[results[i][j]][2]))
                qp.drawPoint(j, i)
                j = j + 1
            j = 0
            i = i + 1
        self.result_image.update()

    def draw_results(self):
        pixmap = QtGui.QPixmap(self.image_path)
        if self.bwRadioButton.isChecked():
            pixmap.fill(QtCore.Qt.black)
        self.result_image.setPixmap(pixmap)
        results = self.algs.images_cosegmented[self.image_path]
        qp = QtGui.QPainter(self.result_image.pixmap())
        if self.bwRadioButton.isChecked():
            qp.setPen(QtGui.QPen(QtCore.Qt.white, 1))
        elif self.bRadioButton.isChecked():
            qp.setPen(QtGui.QPen(QtCore.Qt.yellow, 3))
            results = find_boundaries(results)
        i = 0
        j = 0
        lengthy = len(results)
        lengthx = len(results[0])
        print(lengthy, lengthx)
        while i < lengthy:
            while j < lengthx:
                if results[i][j] == 1:
                    qp.drawPoint(j, i)
                j = j + 1
            j = 0
            i = i + 1
        self.result_image.update()

    def draw_bounds(self):
        pixmap = QtGui.QPixmap(self.image_path)
        self.superImage.setPixmap(pixmap)
        if self.algs.images_segmented[self.image_path] is None:
            return
        qp = QtGui.QPainter(self.superImage.pixmap())
        qp.setPen(QtGui.QPen(QtCore.Qt.red, 1))
        for x in self.foreground[self.image_path]:
            qp.drawPoint(x[0], x[1])
        qp.setPen(QtGui.QPen(QtCore.Qt.blue, 1))
        for x in self.background[self.image_path]:
            qp.drawPoint(x[0], x[1])
        qp.setPen(QtGui.QPen(QtCore.Qt.yellow, 3))
        boundaries = self.algs.get_superpixel_borders_mask(self.image_path)
        i = 0
        j = 0
        lengthy = len(boundaries)
        lengthx = len(boundaries[0])
        print(lengthy, lengthx)
        while i < lengthy:
            while j < lengthx:
                if boundaries[i][j]:
                    qp.drawPoint(j, i)
                j = j + 1
            j = 0
            i = i + 1
        self.superImage.update()
        print("hello")

    def clustering_options(self):
        if self.clusteringBox.currentIndex() == 0:
            self.kmeansFrame.setHidden(True)
            self.graphFrame.setVisible(True)
        elif self.clusteringBox.currentIndex() == 1:
            self.kmeansFrame.setVisible(True)
            self.graphFrame.setHidden(True)

    def create_graph(self):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.mdsData = MDS.mds_transform(self.algs.images_superpixels_feature_vector[self.image_path])
        print(self.mdsData)
        # Create the main application instance

        # Create the view
        self.view = pg.PlotWidget()
        self.view.resize(800, 600)
        self.view.setWindowTitle('Scatter plot using pyqtgraph with PyQT5')
        self.view.setAspectLocked(True)
        self.view.show()

        # Create the scatter plot and add it to the view
        scatter = pg.ScatterPlotItem(pen=pg.mkPen(width=1, color='r'), symbol='o', size=5)
        scatter.sigClicked.connect(self.on_click_graph)
        self.view.addItem(scatter)

        # Convert data array into a list of dictionaries with the x,y-coordinates
        pos = [{'pos': x} for x in self.mdsData]
        print(pos)
        scatter.setData(pos)

    def on_click_graph(self, graph, points):
        self.graphMarked.clear()
        for point in points:
            print([point.pos()[0], point.pos()[1]])
            self.graphMarked.append(np.where([point.pos()[0], point.pos()[1]] == self.mdsData)[0][0])
        print(self.graphMarked)
        self.draw_bounds()
        qp = QtGui.QPainter(self.superImage.pixmap())
        qp.setPen(QtGui.QPen(QtGui.QColor(0,255,0,25), 3))
        i = 0
        j = 0
        array = self.algs.images_segmented[self.image_path]
        print(array)
        lengthy = len(array)
        lengthx = len(array[0])
        print(lengthy, lengthx)
        while i < lengthy:
            while j < lengthx:
                if array[i][j] in self.graphMarked:
                    qp.drawPoint(j, i)
                j = j + 1
            j = 0
            i = i + 1
        self.superImage.update()

    def set_gmm(self):
        self.algs.compute_gmm(components_range=range(self.componentMin.value(), self.componentMax.value()), n_init= self.n_init.value())
        self.algs.compute_edge_uncertainties()
        self.algs.compute_node_uncertainties()

    def disable_buttons(self):
        self.superpixelButton.setDisabled(True)
        self.histogramButton.setDisabled(True)
        self.histogramRadioButton.setDisabled(True)
        self.graph_button.setDisabled(True)
        self.GMMButton.setDisabled(True)
        self.graphcutButton.setDisabled(True)
        self.kmeansButton.setDisabled(True)
        self.clearMarkingsButton.setDisabled(True)

    def enable_buttons(self, option):
        if option > 0:
            self.superpixelButton.setEnabled(True)
            self.clearMarkingsButton.setEnabled(True)
        if option > 1:
            self.histogramButton.setEnabled(True)
        if option > 3:
            self.histogramRadioButton.setEnabled(True)
            self.graph_button.setEnabled(True)
            self.GMMButton.setEnabled(True)
            self.kmeansButton.setEnabled(True)
        if option > 4:
            self.graphcutButton.setEnabled(True)

    def change_features(self):
        self.colorLabel1.setHidden(True)
        self.colorLabel2.setHidden(True)
        self.colorLabel3.setHidden(True)
        self.colorLabel4.setHidden(True)
        self.HHist.setHidden(True)
        self.SHist.setHidden(True)
        self.HSHHist.setHidden(True)
        self.HSSHist.setHidden(True)
        self.siftlabel.setHidden(True)
        self.siftKeyPoint.setHidden(True)
        self.entropyLabel_1.setHidden(True)
        self.entropyLabel_2.setHidden(True)
        self.HentropyCheckBox.setHidden(True)
        self.SentropyCheckBox.setHidden(True)
        self.windowHogLabel.setHidden(True)
        self.winSize.setHidden(True)
        self.blockHogLabel.setHidden(True)
        self.blockSize.setHidden(True)
        self.blockStrideHogLabel.setHidden(True)
        self.blockStride.setHidden(True)
        self.cellHogLabel.setHidden(True)
        self.cellSize.setHidden(True)
        self.binsHogLabel.setHidden(True)
        self.hogBins.setHidden(True)

        currentIndex = self.featureSelected.currentText()
        print(currentIndex)

        if currentIndex == "Hue":
            self.colorLabel1.setVisible(True)
            self.HHist.setVisible(True)
            self.entropyLabel_1.setVisible(True)
            self.HentropyCheckBox.setVisible(True)
        if currentIndex == "Saturation":
            self.colorLabel2.setVisible(True)
            self.SHist.setVisible(True)
            self.entropyLabel_2.setVisible(True)
            self.SentropyCheckBox.setVisible(True)
        if currentIndex == "Hue x Saturation":
            self.colorLabel3.setVisible(True)
            self.colorLabel4.setVisible(True)
            self.HSSHist.setVisible(True)
            self.HSHHist.setVisible(True)
        if currentIndex ==  "Sift":
            self.siftlabel.setVisible(True)
            self.siftKeyPoint.setVisible(True)
        if currentIndex == "Hog":
            self.windowHogLabel.setVisible(True)
            self.winSize.setVisible(True)
            self.blockHogLabel.setVisible(True)
            self.blockSize.setVisible(True)
            self.blockStrideHogLabel.setVisible(True)
            self.blockStride.setVisible(True)
            self.cellHogLabel.setVisible(True)
            self.cellSize.setVisible(True)
            self.binsHogLabel.setVisible(True)
            self.hogBins.setVisible(True)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    fb = MyFileBrowser()
    fb.show()
    app.exec_()
