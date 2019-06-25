from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore
from mainui import designer
from algorithms import pipeline
from algorithms import MDS
import os
import numpy as np
import pyqtgraph as pg
from skimage.segmentation import find_boundaries

# add heat map, eraser, ground truth comparison
from error import errormessage
from compare_pixel import compare_pixel
import xml.etree.ElementTree as ET


colors = [[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255],
          [255, 0, 255], [192, 192, 192], [128, 128, 128], [128, 0, 0], [128, 128, 0], [0, 128, 0], [128, 0, 128],
          [0, 128, 128], [0, 0, 128]]


class mainUI(designer.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self):
        super(mainUI, self).__init__()
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
        self.graphcutButton.clicked.connect(self.compute_graph_cut)
        self.kmeansButton.clicked.connect(self.kmeans)
        self.graph_button.clicked.connect(self.create_graph)
        self.GMMButton.clicked.connect(self.set_gmm)
        self.bwRadioButton.clicked.connect(self.draw_graph_cut)
        self.bRadioButton.clicked.connect(self.draw_graph_cut)
        self.colorRadioButton.clicked.connect(self.draw_kmeans)
        self.bwkRadioButton.clicked.connect(self.draw_kmeans)
        self.edgeRadioButton.clicked.connect(self.draw_uncertainties)
        self.graphRadioButton.clicked.connect(self.draw_uncertainties)
        self.nodeRadioButton.clicked.connect(self.draw_uncertainties)

        self.gt_originalRadioButton.clicked.connect(self.draw_gt)
        self.gt_overlapRadioButton.clicked.connect(self.draw_gt)

        self.k1.stateChanged.connect(self.update_kmeans)
        self.k2.stateChanged.connect(self.update_kmeans)
        self.k3.stateChanged.connect(self.update_kmeans)
        self.k4.stateChanged.connect(self.update_kmeans)
        self.k5.stateChanged.connect(self.update_kmeans)
        self.k6.stateChanged.connect(self.update_kmeans)
        self.k7.stateChanged.connect(self.update_kmeans)
        self.k8.stateChanged.connect(self.update_kmeans)
        self.k9.stateChanged.connect(self.update_kmeans)
        self.k10.stateChanged.connect(self.update_kmeans)
        self.k11.stateChanged.connect(self.update_kmeans)
        self.k12.stateChanged.connect(self.update_kmeans)
        self.k13.stateChanged.connect(self.update_kmeans)
        self.k14.stateChanged.connect(self.update_kmeans)
        self.k15.stateChanged.connect(self.update_kmeans)
        self.k16.stateChanged.connect(self.update_kmeans)


        self.featureSelected.currentIndexChanged.connect(self.change_features)
        self.clusteringBox.currentIndexChanged.connect(self.change_clustering)

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
        self.groundtruth = dict()
        self.currentFolder = "images"
        self.img_float = []
        self.image_path = ""
        self.graphMarked = []
        self.model = QtWidgets.QFileSystemModel()
        self.result = "None"
        self.relpath = ""
        self.change_clustering()
        self.change_features()
        self.populate()
        self.set_tooltips()

    def populate(self):
        path = os.getcwd() + '/' + self.currentFolder
        self.model.setRootPath(QtCore.QDir.rootPath())
        self.treeView.setModel(self.model)
        self.treeView.setRootIndex(self.model.index(path))
        self.treeView.setSortingEnabled(True)
        self.treeView.hideColumn(1)
        self.treeView.hideColumn(2)
        self.treeView.hideColumn(3)

    def context_menu(self):
        if self.model.isDir(self.treeView.currentIndex()):
            menu = QtWidgets.QMenu()
            open = menu.addAction("select folder")
            open.triggered.connect(self.select_folder)
            cursor = QtGui.QCursor()
            menu.exec_(cursor.pos())
        elif self.listWidget.currentItem() is not None:
            menu = QtWidgets.QMenu()
            open = menu.addAction("select image as ground truth")
            open.triggered.connect(self.select_gt)
            cursor = QtGui.QCursor()
            menu.exec_(cursor.pos())

    def select_gt(self):
        index = self.treeView.currentIndex()
        self.file_path = self.model.filePath(index)
        self.groundtruth[self.image_path] = self.file_path
        if self.result != "None":
            self.gt_overlapRadioButton.setEnabled(True)
        self.draw_gt()
        self.gt_originalRadioButton.setEnabled(True)

    def draw_gt(self):
        pixmap = QtGui.QPixmap(self.file_path)

        if self.image.pixmap().size() != pixmap.size():
            errormessage("Wrong gt selected", "Size of images do not match")
            return

        self.groundtruth[self.image_path] = self.file_path

        if self.gt_originalRadioButton.isChecked():
            self.compare_image.setPixmap(pixmap)
            self.compare_image.update()
        else:
            self.compare_image.setPixmap(pixmap)
            self.compare_image.pixmap().fill(QtCore.Qt.black)
            qp = QtGui.QPainter(self.compare_image.pixmap())
            qp.setPen(QtGui.QPen(QtCore.Qt.white, 1))
            gt = pixmap.toImage()

            results = self.algs.images_cosegmented[self.image_path]

            y = 0
            x = 0
            print(gt.width(),gt.height())
            wrong = 0
            if self.result == "kmeans":
                table = [self.k1.isChecked(), self.k2.isChecked(), self.k3.isChecked(), self.k4.isChecked(),
                         self.k5.isChecked(), self.k6.isChecked(), self.k7.isChecked(), self.k8.isChecked(),
                         self.k9.isChecked(), self.k10.isChecked(), self.k11.isChecked(), self.k12.isChecked(),
                         self.k13.isChecked(), self.k14.isChecked(), self.k15.isChecked(), self.k16.isChecked(), ]
                while y < gt.height():
                    while x < gt.width():
                        gtpixel = gt.pixelColor(x, y).getRgb()
                        compare = results[y][x]
                        if compare_pixel(gtpixel=gt.pixelColor(x, y).getRgb(), resultpixel=table[compare]):
                            qp.drawPoint(x, y)
                        else:
                            wrong = wrong + 1
                        x = x + 1
                    y = y + 1
                    x = 0
            elif self.result == "graphcut":
                while y < gt.height():
                    while x < gt.width():
                        gtpixel = gt.pixelColor(x, y).getRgb()
                        compare = results[y][x]
                        if (gtpixel[0] != 0):
                            print(gtpixel[0],gtpixel[1],gtpixel[2])
                            print(compare)
                        if compare_pixel(gtpixel, compare):
                            print("im here")
                            qp.drawPoint(x, y)
                        else:
                            wrong = wrong + 1
                        x = x + 1
                    y = y + 1
                    x = 0
            print(wrong)
            right =((self.compare_image.pixmap().width() * self.compare_image.pixmap().height()) - wrong) / (self.compare_image.pixmap().width() * self.compare_image.pixmap().height())*100
            print(right)
            self.gtpercentage.setText(str(round(right)) + "%")
            self.compare_image.update()

    def select_folder(self):
        self.listWidget.clear()
        self.image.clear()
        self.foreground.clear()
        self.background.clear()
        self.groundtruth.clear()
        self.image_paths.clear()
        self.algs = pipeline.Pipeline([])
        self.clusteringStatus.setText("Not calculated")
        self.extractionStatus.setText("Not calculated")
        self.superpixelStatus.setText("Not calculated")
        self.clusteringStatus.setStyleSheet("color: red")
        self.extractionStatus.setStyleSheet("color: red")
        self.superpixelStatus.setStyleSheet("color: red")

        index = self.treeView.currentIndex()
        file_path = self.model.filePath(index)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        common_prefix = os.path.commonprefix([dir_path, file_path])
        self.relpath = os.path.relpath(file_path, common_prefix)

        print(file_path)
        print(os.listdir(file_path))
        print(common_prefix)
        self.disable_buttons()

        for file in os.listdir(file_path):
            if file[-4:] == ".png" or file[-4:] == ".jpg":
                image_path = self.relpath + "/" + file
                self.image_paths.append(image_path)
                item = QtWidgets.QListWidgetItem()
                icon = QtGui.QIcon()
                pixmap = QtGui.QPixmap(image_path)  # QImage object
                icon.addPixmap(pixmap, QtGui.QIcon.Normal, QtGui.QIcon.Off)
                item.setIcon(icon)
                item.setText(file)
                self.listWidget.addItem(item)
        if len(self.image_paths) != 0:
            print(self.image_paths)
            self.algs = pipeline.Pipeline(self.image_paths)
            print(self.image_paths)

    def choose_image(self):
        if self.listWidget.currentItem() is not None:
            self.image_path = self.relpath + '/' + self.listWidget.currentItem().text()
            print(self.image_path)
            pixmap = QtGui.QPixmap(self.image_path)
            self.image.setPixmap(pixmap)
            self.image.mousePressEvent = self.move_start
            self.image.mouseMoveEvent = self.move_connect
            self.image.mouseReleaseEvent = self.release_move
            self.superImage.mousePressEvent = self.move_start
            self.superImage.mouseMoveEvent = self.move_connect
            self.superImage.mouseReleaseEvent = self.release_move
            self.enable_buttons(1)
            if self.image_path not in self.foreground:
                self.foreground[self.image_path] = []
                self.background[self.image_path] = []
            else:
                self.draw_markings()
            self.draw_bounds()
            self.graphMarked.clear()
            if self.algs.images_superpixels_uncertainties_node[self.image_path]:
                self.draw_uncertainties()

            if self.image_path in self.algs.images_cosegmented:
                if self.clusteringBox.currentIndex() == 0:
                    self.draw_graph_cut()
                elif self.clusteringBox.currentIndex() == 1:
                    self.draw_kmeans()
            if self.image_path in self.groundtruth:
                self.draw_gt()
            elif self.compare_image.pixmap() is not None:
                self.compare_image.clear()



        else:
            self.disable_buttons()

    def move_start(self, event):
        x = event.pos().x()
        y = event.pos().y() - round((self.image.height() - self.image.pixmap().height()) / 2)
        if y < 0 or x < 0 or y > self.image.pixmap().height() or x > self.image.pixmap().width():
            return
        if self.histogramRadioButton.isChecked():
            self.on_click_superpixel(x, y)
            print("hello")
        else:
            self.save_point(x, y)
            self.point = (x, y)
            self.draw_markings()

    def save_point(self, x, y):
        if (x, y) in self.foreground[self.image_path]:
            self.foreground[self.image_path].remove((x, y))
        elif (x, y) in self.background:
            self.background[self.image_path].remove((x, y))

        if self.pencil == 1:
            self.background[self.image_path].append((x, y))
        elif self.pencil == 2:
            self.foreground[self.image_path].append((x, y))

    def move_connect(self, event):
        if self.drawRadioButton.isChecked():
            self.image.update()
            x = event.pos().x()
            y = event.pos().y() - round((self.image.height() - self.image.pixmap().height()) / 2)

            if y < 0 or x < 0 or y > self.image.pixmap().height() or x > self.image.pixmap().width():
                return
            if self.point == (-1, -1) or self.point == (x, y):
                self.point = (x, y)
                self.save_point(x, y)
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

    def release_move(self, event):
        self.point = (-1, -1)

    def save_linex(self, x1, x2, y1, y2):
        slope = (y1 - y2) / (x1 - x2)
        i = x1 + 1
        while i <= x2:
            self.save_point(i, round(y1 + slope * (i - x1)))
            i = i + 1

    def save_liney(self, x1, x2, y1, y2):
        slope = (x1 - x2) / (y1 - y2)
        i = y1 + 1
        while i <= y2:
            self.save_point(round(x1 + slope * (i - y1)), i)
            i = i + 1

    def draw_markings(self):
        pixmap = QtGui.QPixmap(self.image_path)
        self.image.setPixmap(pixmap)
        qp = QtGui.QPainter(self.image.pixmap())
        qp.setPen(QtGui.QPen(QtCore.Qt.blue, 1))
        print(self.foreground)
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
        print(self.algs.images_superpixels)
        self.algs.compute_neighbors()
        self.algs.compute_centers()
        self.draw_bounds()
        self.draw_markings()
        self.disable_buttons()
        self.enable_buttons(2)
        self.superpixelStatus.setText("Calculated")
        self.superpixelStatus.setStyleSheet("color: lime")
        if self.extractionStatus.text() == "Calculated":
            self.extractionStatus.setText("Need recalculation")
            self.extractionStatus.setStyleSheet("color: red")
            if self.clusteringStatus.text() == "Calculated":
                self.clusteringStatus.setText("Recalculation needed")
                self.clusteringStatus.setStyleSheet("color: red")

    def set_histograms(self):
        self.algs.compute_feature_vectors(means_bgr=self.RGB.isChecked(), means_hsv=self.HSV.isChecked(),
                                          h_hist=self.Hue.isChecked(), h_hist_bins=self.HHist.value(),
                                          h_hist_entropy=self.HentropyCheckBox.isChecked(),
                                          s_hist=self.Saturation.isChecked(), s_hist_bins=self.SHist.value(),
                                          s_hist_entropy=self.SentropyCheckBox.isChecked(),
                                          hs_hist=self.HueSat.isChecked(), hs_hist_bins_h=self.HSHHist.value(),
                                          hs_hist_bins_s=self.HSSHist.value(), sift=self.Sift.isChecked(),
                                          sift_kp_size=self.siftKeyPoint.value(), hog=self.Hog.isChecked(),
                                          hog_winSize=(self.winSize.value(), self.winSize.value()),
                                          hog_blockSize=(self.blockSize.value(), self.blockSize.value()),
                                          hog_blockStride=(self.blockStride.value(), self.blockStride.value()),
                                          hog_cellSize=(self.cellSize.value(), self.cellSize.value()),
                                          hog_bins=self.hogBins.value())
        self.disable_buttons()
        self.enable_buttons(3)
        self.extractionStatus.setText("Calculated")
        self.extractionStatus.setStyleSheet("color: lime")
        if self.clusteringStatus.text() == "Calculated":
            self.clusteringStatus.setText("Need recalculation")
            self.clusteringStatus.setStyleSheet("color: red")

    def set_markings(self):
        foreground = []
        background = []
        for x in self.foreground[self.image_path]:
            if self.algs.images_segmented[self.image_path][x[1] - 1][x[0] - 1] not in foreground:
                foreground.append(self.algs.images_segmented[self.image_path][x[1]][x[0]])
        for y in self.background[self.image_path]:
            if self.algs.images_segmented[self.image_path][y[1] - 1][y[0] - 1] not in background:
                background.append(self.algs.images_segmented[self.image_path][y[1]][y[0]])
        self.algs.set_fg_segments(self.image_path, foreground)
        self.algs.set_bg_segments(self.image_path, background)

    def clear_markings(self):
        self.foreground[self.image_path].clear()
        self.background[self.image_path].clear()
        self.draw_markings()
        self.draw_bounds()
        self.image.update()
        self.superImage.update()
        self.result_image.update()

    def compute_graph_cut(self):
        self.algs.perform_graph_cut()
        self.result = "graphcut"
        self.draw_graph_cut()
        self.disable_buttons()
        self.enable_buttons(5)
        if self.gt_originalRadioButton.isEnabled():
            self.gt_overlapRadioButton.setEnabled(True)
        self.clusteringStatus.setText("Calculated")
        self.clusteringStatus.setStyleSheet("color: lime")

    def kmeans(self):
        self.set_markings()
        self.algs.perform_k_means_clustering(num_clusters=self.kclustervalue.value())
        self.result = "kmeans"
        self.draw_kmeans()
        self.enable_buttons(3)
        if self.gt_originalRadioButton.isEnabled():
            self.gt_overlapRadioButton.setEnabled(True)

    def draw_kmeans(self):
        if self.result == "kmeans":
            pixmap = QtGui.QPixmap(self.image_path)

            self.result_image.setPixmap(pixmap)
            results = self.algs.images_cosegmented[self.image_path]
            qp = QtGui.QPainter(self.result_image.pixmap())
            i = 0
            j = 0
            lengthy = len(results)
            lengthx = len(results[0])
            if self.colorRadioButton.isChecked():
                color = colors
            else:
                color = [(self.k1.isChecked() * 255, self.k1.isChecked() * 255, self.k1.isChecked() * 255),
                         (self.k2.isChecked() * 255, self.k2.isChecked() * 255, self.k2.isChecked() * 255),
                         (self.k3.isChecked() * 255, self.k3.isChecked() * 255, self.k3.isChecked() * 255),
                         (self.k4.isChecked() * 255, self.k4.isChecked() * 255, self.k4.isChecked() * 255),
                         (self.k5.isChecked() * 255, self.k5.isChecked() * 255, self.k5.isChecked() * 255),
                         (self.k6.isChecked() * 255, self.k6.isChecked() * 255, self.k6.isChecked() * 255),
                         (self.k7.isChecked() * 255, self.k7.isChecked() * 255, self.k7.isChecked() * 255),
                         (self.k8.isChecked() * 255, self.k8.isChecked() * 255, self.k8.isChecked() * 255),
                         (self.k9.isChecked() * 255, self.k9.isChecked() * 255, self.k9.isChecked() * 255),
                         (self.k10.isChecked() * 255, self.k10.isChecked() * 255, self.k10.isChecked() * 255),
                         (self.k11.isChecked() * 255, self.k11.isChecked() * 255, self.k11.isChecked() * 255),
                         (self.k12.isChecked() * 255, self.k12.isChecked() * 255, self.k12.isChecked() * 255),
                         (self.k13.isChecked() * 255, self.k13.isChecked() * 255, self.k13.isChecked() * 255),
                         (self.k14.isChecked() * 255, self.k14.isChecked() * 255, self.k14.isChecked() * 255),
                         (self.k15.isChecked() * 255, self.k15.isChecked() * 255, self.k15.isChecked() * 255),
                         (self.k16.isChecked() * 255, self.k16.isChecked() * 255, self.k16.isChecked() * 255) ]
        print(lengthy, lengthx)
        while i < lengthy:
            while j < lengthx:
                qp.setPen(QtGui.QColor(color[results[i][j]][0], color[results[i][j]][1], color[results[i][j]][2]))
                qp.drawPoint(j, i)
                j = j + 1
            j = 0
            i = i + 1
        self.result_image.update()

    def update_kmeans(self):
        if self.bwkRadioButton.isChecked():
            self.draw_kmeans()

    def draw_graph_cut(self):
        if self.result == "graphcut":
            pixmap = QtGui.QPixmap(self.image_path)
            if self.bwRadioButton.isChecked():
                pixmap.fill(QtCore.Qt.black)
            self.result_image.setPixmap(pixmap)
            results = self.algs.images_cosegmented[self.image_path]
            qp = QtGui.QPainter(self.result_image.pixmap())
            if self.bwRadioButton.isChecked():
                qp.setPen(QtGui.QPen(QtCore.Qt.white, 1))
            elif self.bRadioButton.isChecked():
                qp.setPen(QtGui.QPen(QtCore.Qt.yellow, 2))
                results = find_boundaries(results)
            y = 0
            x = 0
            lengthy = len(results)
            lengthx = len(results[0])
            print(lengthy, lengthx)
            while y < lengthy:
                while x < lengthx:
                    if results[y][x] == 1:
                        qp.drawPoint(x, y)
                    x = x + 1
                x = 0
                y = y + 1
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
        qp.setPen(QtGui.QPen(QtCore.Qt.yellow, 2))
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

        self.view = pg.PlotWidget()
        self.view.resize(800, 600)
        self.view.setWindowTitle('MDS graph')
        self.view.setAspectLocked(True)
        self.view.show()

        scatter = pg.ScatterPlotItem(pen=pg.mkPen(width=1, color='r'), symbol='o', size=7)
        pos = [{'pos': x} for x in self.mdsData]
        scatter.setData(pos)
        scatter.sigClicked.connect(self.on_click_graph)
        self.view.addItem(scatter)


    def on_click_graph(self, _, points):
        self.clear_graph_marked()
        for point in points:
            self.graphMarked.append(np.where([point.pos()[0], point.pos()[1]] == self.mdsData)[0][0])
            point.setBrush(pg.mkBrush(0, 255, 0, 255))
        self.draw_graph_marked()


    def on_click_superpixel(self, x, y):
        self.clear_graph_marked()
        superpixel = self.algs.images_segmented[self.image_path][y][x]
        self.view.getPlotItem().dataItems[0].points()[superpixel].setBrush(pg.mkBrush(0, 255, 0, 255))
        self.graphMarked.append(superpixel)
        self.draw_graph_marked()


    def clear_graph_marked(self):
        for superpixel in self.graphMarked:
            self.view.getPlotItem().dataItems[0].points()[superpixel].setBrush(pg.mkBrush(100, 100, 150, 255))
        self.graphMarked.clear()


    def draw_graph_marked(self):
        self.draw_bounds()
        print(self.graphMarked)
        print(self.mdsData)
        qp = QtGui.QPainter(self.superImage.pixmap())
        qp.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0, 25), 3))
        i = 0
        j = 0
        array = self.algs.images_segmented[self.image_path]
        lengthy = len(array)
        lengthx = len(array[0])
        while i < lengthy:
            while j < lengthx:
                if array[i][j] in self.graphMarked:
                    qp.drawPoint(j, i)
                j = j + 1
            j = 0
            i = i + 1
        self.superImage.update()


    def set_gmm(self):
        self.set_markings()
        totalForeground  = 0
        totalBackground = 0
        for i in self.algs.images_superpixels_foreground.values():
            print(i)
            print(len(i))
            totalForeground = totalForeground + len(i)
        for j in self.algs.images_superpixels_background.values():
            totalBackground = totalBackground + len(j)
        if totalForeground >= self.componentMax.value() and totalBackground >= self.componentMax.value():
            self.algs.compute_gmm(components_range=range(self.componentMin.value(), self.componentMax.value()),
                                  n_init=self.n_init.value())
            self.algs.compute_edge_uncertainties()
            self.algs.compute_node_uncertainties()
            self.draw_uncertainties()
            self.disable_buttons()
            self.enable_buttons(4)

        else:
            errormessage("Not enough superpixels marked",
                         "Both foreground and background need more superpixels marked than the maximum components")


    def draw_uncertainties(self):
        pixmap = QtGui.QPixmap(self.image_path)
        self.uncertainty_image.setPixmap(pixmap)
        if self.edgeRadioButton.isChecked():
            x = self.algs.images_superpixels_uncertainties_edge[self.image_path]
        elif self.nodeRadioButton.isChecked():
            x = self.algs.images_superpixels_uncertainties_node[self.image_path]
            print(x)
        elif self.graphRadioButton.isChecked():
            x = [1 - i for i in self.algs.images_superpixels_uncertainties_graph_cut[self.image_path]]
        self.uncertainty_image.setPixmap(pixmap)
        qp = QtGui.QPainter(self.uncertainty_image.pixmap())
        i = 0
        j = 0
        lengthy = pixmap.height()
        lengthx = pixmap.width()
        print(lengthy, lengthx)
        while i < lengthy:
            while j < lengthx:
                color = round(x[self.algs.images_segmented[self.image_path][i][j]] * 1020)
                if color <= 255:
                    qp.setPen(QtGui.QPen(QtGui.QColor(0, color, 255)))
                elif color <= 510:
                    qp.setPen(QtGui.QPen(QtGui.QColor(0, 255, 510 - color)))
                elif color <= 765:
                    qp.setPen(QtGui.QPen(QtGui.QColor(color - 510, 255, 0)))
                else:
                    qp.setPen(QtGui.QPen(QtGui.QColor(255, 1020 - color, 0)))

                qp.drawPoint(j, i)
                j = j + 1
            j = 0
            i = i + 1
        self.uncertainty_image.update()


    def disable_buttons(self):
        self.superpixelButton.setDisabled(True)
        self.histogramButton.setDisabled(True)
        self.histogramRadioButton.setDisabled(True)
        self.drawRadioButton.setDisabled(True)
        self.histogramRadioButton.setDisabled(True)
        self.graph_button.setDisabled(True)
        self.GMMButton.setDisabled(True)
        self.graphcutButton.setDisabled(True)
        self.kmeansButton.setDisabled(True)
        self.clearMarkingsButton.setDisabled(True)
        self.edgeRadioButton.setDisabled(True)
        self.nodeRadioButton.setDisabled(True)
        self.bwRadioButton.setDisabled(True)
        self.bRadioButton.setDisabled(True)
        self.graphRadioButton.setDisabled(True)


    def enable_buttons(self, option):
        if option > 0:
            self.superpixelButton.setEnabled(True)
            self.clearMarkingsButton.setEnabled(True)
        if option > 1:
            self.histogramButton.setEnabled(True)
            self.drawRadioButton.setEnabled(True)
            self.histogramRadioButton.setEnabled(True)
        if option > 2:
            self.histogramRadioButton.setEnabled(True)
            self.graph_button.setEnabled(True)
            self.GMMButton.setEnabled(True)
            self.kmeansButton.setEnabled(True)
            self.bwkRadioButton.setEnabled(True)
            self.colorRadioButton.setEnabled(True)
        if option > 3:
            self.graphcutButton.setEnabled(True)
            self.edgeRadioButton.setEnabled(True)
            self.nodeRadioButton.setEnabled(True)
        if option > 4:
            self.graphRadioButton.setEnabled(True)
            self.bwRadioButton.setEnabled(True)
            self.bRadioButton.setEnabled(True)


    def change_clustering(self):



        current_index = self.clusteringBox.currentText()

        if current_index == "Graph cut":
            self.graphCutModeLabel.setVisible(True)
            self.graphCutModeFrame.setVisible(True)
            self.clusterModeLabel.setHidden(True)
            self.clusterModeFrame.setHidden(True)

        if current_index == "kmeans clustering":
            self.clusterModeLabel.setVisible(True)
            self.clusterModeFrame.setVisible(True)
            self.graphCutModeLabel.setHidden(True)
            self.graphCutModeFrame.setHidden(True)


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

        current_index = self.featureSelected.currentText()

        if current_index == "Hue":
            self.colorLabel1.setVisible(True)
            self.HHist.setVisible(True)
            self.entropyLabel_1.setVisible(True)
            self.HentropyCheckBox.setVisible(True)
        if current_index == "Saturation":
            self.colorLabel2.setVisible(True)
            self.SHist.setVisible(True)
            self.entropyLabel_2.setVisible(True)
            self.SentropyCheckBox.setVisible(True)
        if current_index == "Hue x Saturation":
            self.colorLabel3.setVisible(True)
            self.colorLabel4.setVisible(True)
            self.HSSHist.setVisible(True)
            self.HSHHist.setVisible(True)
        if current_index == "Sift":
            self.siftlabel.setVisible(True)
            self.siftKeyPoint.setVisible(True)
        if current_index == "Hog":
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

    def write_xml_settings(self):
        root = ET.Element("root")
        settings = ET.SubElement("settings")
        ET.SubElement(settings, "super_pixel_quantity")

    def set_tooltips(self):
        self.superpixelCalculationLabel.setToolTip("Calculate the superpixels for all images in the folder using SLIC")
        self.superpixelQuantityLabel.setToolTip("Set the approximate amount of superpixels each image will have")
        self.iterationsLabel.setToolTip("Maximum number of iterations of k-means")
        self.compactnessLabel.setToolTip("Balances color proximity and space proximity. Higher results in more square superpixels")
        self.sigmaLabel.setToolTip("Width of Gaussian smoothing kernel for optional pre-processing. 0 for no smoothing")



        self.featureExtractionLabel.setToolTip(
            "Compute feature vectors for each superpixel consisting of the selected features")
        self.RGB.setToolTip("Means of RGB")
        self.HSV.setToolTip("Means of HSV")
        self.superpixelProgress.setToolTip("Right click on a folder and select on an image" 
                                           "Procceed by pressing calculate superpixels")




if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    fb = mainUI()
    fb.show()
    app.exec_()
