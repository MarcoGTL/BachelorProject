from PyQt5 import QtWidgets
from PyQt5 import QtGui

from draw import draw_uncertainties, draw_bounds, draw_gt, draw_kmeans, draw_markings, draw_graph_cut
from fileSystem import context_menu, populate
from initialize import initialize
from mainui import designer
from algorithms import pipeline
from algorithms import MDS
import os
import pyqtgraph as pg

from error import errormessage
import xml.etree.ElementTree as ET

from plot import on_click_superpixel, on_click_plot
from visibility import change_features, change_cosegmentation, enable_buttons, disable_buttons

"""
Author: Marco

A class instantiating the layout from QT Designer and adding functionality to everything. Contains some functions that
require a lot of the UI variables. Other functions are available in the mainui folder

Required Packages:
    pyqtgraph
    PyQt5
    xml (usually pre-installed)
    
    Attributes:
        mdsData = List                                 # Downscaled version of feature vector
        point = (int, int)                             # Last point drawn
        image_paths = List                             # Image paths of selected folder
        algs: Pipeline                                 # Pipeline of algorithms used for cosegmentation
        pencil = 2                                     # Current drawing pencil"
        relative_image_path = ""                       # Path of folder relative to program"
        foreground: dict                               # Foreground  points"
        background: dict                               # Background points"
        groundtruth: dict                              # Ground truth images with normal image path as key"
        image_path: String                             # Path of selected image
        plotMarked: List                               # Current set of points selected in mds Plot
        model: QFileSystemModel                        # File system for selecting images 
"""


class mainUI(designer.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self):
        super(mainUI, self).__init__()
        self.setupUi(self)  # QT Designer template instantiated

        # connects everything to it's appropriate function and adds tooltips
        initialize(self.treeView, self.context_menu, self.listWidget, self.choose_image, self.clearMarkingsButton,
                   self.clear_markings, self.histogramButton, self.set_feature_vector, self.superpixelButton,
                   self.calculate_superpixels, self.graphcutButton, self.compute_graph_cut, self.kmeansButton,
                   self.kmeans, self.graph_button, self.create_plot, self.GMMButton, self.set_gmm, self.bwRadioButton,
                   self.draw_graph_cut, self.bRadioButton, self.colorRadioButton, self.draw_kmeans, self.bwkRadioButton,
                   self.edgeRadioButton, self.draw_uncertainties, self.graphRadioButton, self.nodeRadioButton,
                   self.gt_originalRadioButton, self.draw_gt, self.gt_overlapRadioButton, self.k1, self.update_kmeans,
                   self.k2, self.k3, self.k4, self.k5, self.k6, self.k7, self.k8, self.k9, self.k10, self.k11, self.k12,
                   self.k13, self.k14, self.k15, self.k16, self.featureSelected, self.change_features,
                   self.clusteringBox, self.change_cosegmentation, self.clustering_options, self.kmeansFrame,
                   self.fgRadioButton, self.currentPencil, self.bgRadioButton, self.superpixelCalculationLabel,
                   self.superpixelQuantityLabel, self.iterationsLabel, self.compactnessLabel, self.sigmaLabel,
                   self.featureExtractionLabel, self.RGB, self.HSV, self.Hue, self.Saturation, self.HueSat, self.Sift,
                   self.Hog, self.currentLabel, self.colorLabel1, self.colorLabel2, self.colorLabel3, self.colorLabel4,
                   self.entropyLabel_1, self.entropyLabel_2, self.siftlabel, self.windowHogLabel,
                   self.blockStrideHogLabel, self.cellHogLabel, self.binsHogLabel, self.cosegmentationLabel,
                   self.componentLabel, self.restartLabel, self.pairwiseLabel, self.scaleLabel, self.clusterLabel,
                   self.clusterFGLabel, self.markingLabel, self.superpixelLabel, self.uncertaintyLabel,
                   self.graphCutModeLabel, self.clusterModeLabel, self.gtLabel, self.superpixelProgress,
                   self.featureProgress, self.cosegmentationProgress)

        # initialize global variables which some functions need"
        self.mdsData = []  # Downscaled version of feature vector"
        self.point = (-1, -1)  # Last point drawn"
        self.image_paths = []  # Image paths of selected folder"
        self.algs = pipeline.Pipeline([])  # Pipeline of algorithms"
        self.pencil = 2  # Current drawing pencil"
        self.relative_image_path = ""  # Path of folder relative to program"
        self.foreground = dict()  # Foreground  points"
        self.background = dict()  # Background points"
        self.groundtruth = dict()  # Ground truth images with normal image path as key"
        self.image_path = ""  # Path of selected image
        self.plotMarked = []  # Current set of points selected in mds Plot
        self.model = QtWidgets.QFileSystemModel()  #
        self.result = "None"
        self.relpath = ""
        self.change_cosegmentation()
        self.change_features()
        populate("images", self.model, self.treeView)

    """
    Generates context menu folder when  right clicking a folder in the file system view
    """

    def context_menu(self):
        context_menu(self.model, self.treeView, self.select_folder, self.listWidget, self.select_gt)

    """
    Selects a ground truth for the current image by right clicking a image.
    Checks if there are co-segmentation results to enable overlap between result and ground truth
    draws ground truth in the ground truth tab
    """

    def select_gt(self):
        index = self.treeView.currentIndex()
        self.file_path = self.model.filePath(index)
        self.groundtruth[self.image_path] = self.file_path
        if self.result != "None":
            self.gt_overlapRadioButton.setEnabled(True)
        self.draw_gt()
        self.gt_originalRadioButton.setEnabled(True)

    """
    draws ground truth differently depending on which co-segmentation has been used
    """

    def draw_gt(self):
        draw_gt(self.file_path, self.image, self.groundtruth, self.image_path, self.gt_originalRadioButton,
                self.compare_image, self.algs, self.result, self.k1, self.k2, self.k3, self.k4, self.k5, self.k6,
                self.k7, self.k8, self.k9, self.k10, self.k11, self.k12, self.k13, self.k14, self.k15, self.k16,
                self.gtpercentage)

    """
    Clears all settings from the previous folder if it exists and resets the progress box
    Finds all images in folder and adds them to image_paths
    """

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
            self.algs = pipeline.Pipeline(self.image_paths)

    """
    Selects an image from image_paths, draws images that already have been calculated and draws markings
     that have already been drawn.  Sets marking/selection ability for the superpixel tab and the draw tab
    """

    def choose_image(self):
        if self.listWidget.currentItem() is not None:
            self.image_path = self.relpath + '/' + self.listWidget.currentItem().text()
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
                draw_markings(self.image_path, self.image, self.background, self.foreground, self.algs, self.superImage)
            draw_bounds(self.image_path, self.superImage, self.algs, self.foreground, self.background)
            self.plotMarked.clear()
            if self.algs.images_superpixels_uncertainties_node[self.image_path]:
                self.draw_uncertainties()

            if self.image_path in self.algs.images_cosegmented:
                if self.result == "graphcut":
                    self.draw_graph_cut()
                elif self.result == "kmeans":
                    self.draw_kmeans()
            if self.image_path in self.groundtruth:
                self.draw_gt()
            elif self.compare_image.pixmap() is not None:
                self.compare_image.clear()



        else:
            self.disable_buttons()

    """
    Stores the point on the graph that has been marked in foreground or background. Sets last point marked
    If MDS is selected hightlights the superpixel and corresponding point instead.
    """

    def move_start(self, event):
        x = event.pos().x()
        y = event.pos().y() - round((self.image.height() - self.image.pixmap().height()) / 2)
        if y < 0 or x < 0 or y > self.image.pixmap().height() or x > self.image.pixmap().width():
            return
        if self.histogramRadioButton.isChecked():
            on_click_superpixel(self.plotMarked, self.view, self.algs, self.image_path, self.superImage,
                                self.foreground, self.background, x, y)
        else:
            self.save_point(x, y)
            self.point = (x, y)
            draw_markings(self.image_path, self.image, self.background, self.foreground, self.algs, self.superImage)

    """
    Stores ooint and checks if the point has been marked previously and removes that entry
    """

    def save_point(self, x, y):
        if (x, y) in self.foreground[self.image_path]:
            self.foreground[self.image_path].remove((x, y))
        elif (x, y) in self.background:
            self.background[self.image_path].remove((x, y))

        if self.pencil == 1:
            self.background[self.image_path].append((x, y))
        elif self.pencil == 2:
            self.foreground[self.image_path].append((x, y))

    """
    connects last marked point with the current mouse position using midpoint line algorithm. MLA iterates vertically
    instead of horizontally if it is larger in length compared to width. Saves all points between and including.
    """

    def move_connect(self, event):
        if self.drawRadioButton.isChecked():
            x = event.pos().x()
            y = event.pos().y() - round((self.image.height() - self.image.pixmap().height()) / 2)

            if y < 0 or x < 0 or y > self.image.pixmap().height() or x > self.image.pixmap().width():
                return
            if self.point == (-1, -1) or self.point == (x, y):
                self.point = (x, y)
                self.save_point(x, y)
                draw_markings(self.image_path, self.image, self.background, self.foreground, self.algs, self.superImage)
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
            draw_markings(self.image_path, self.image, self.background, self.foreground, self.algs, self.superImage)

    """
    Midpoint line algorithm that iterates horizontally
    """

    def save_linex(self, x1, x2, y1, y2):
        slope = (y1 - y2) / (x1 - x2)
        i = x1 + 1
        while i <= x2:
            self.save_point(i, round(y1 + slope * (i - x1)))
            i = i + 1

    """
    Midpoint line algorithm that iterates vertically
    """

    def save_liney(self, x1, x2, y1, y2):
        slope = (x1 - x2) / (y1 - y2)
        i = y1 + 1
        while i <= y2:
            self.save_point(round(x1 + slope * (i - y1)), i)
            i = i + 1

    """
    Sets last point back to default when the mouse is released
    """

    def release_move(self, event):
        self.point = (-1, -1)

    """
    Sets if the pen is marking foreground(2) or background(1)
    """

    def currentPencil(self):
        if self.fgRadioButton.isChecked():
            self.pencil = 2
        elif self.bgRadioButton.isChecked():
            self.pencil = 1

    """
    Calculates superpixels as well as neighbors and centers. Then draws the superpixels in the superpixel tab by their
    bounds
    """

    def calculate_superpixels(self):
        self.algs.compute_superpixels(self.superpixelSpinBox.value(), self.compactnessSpinBox.value(),
                                      self.iterationsSpinBox.value(), self.sigmaSpinBox.value())
        self.algs.compute_neighbors()
        self.algs.compute_centers()
        draw_bounds(self.image_path, self.superImage, self.algs, self.foreground, self.background)
        draw_markings(self.image_path, self.image, self.background, self.foreground, self.algs, self.superImage)
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

    """
    Calculates the feature vector
    """

    def set_feature_vector(self):
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

    """
    Checks all points marked which superpixel they represent and add to the corresponding foreground/background list
    """

    def set_markings(self):
        foreground = []
        background = []
        for x in self.foreground[self.image_path]:
            if self.algs.images_segmented[self.image_path][x[1]][x[0]] not in foreground:
                foreground.append(self.algs.images_segmented[self.image_path][x[1]][x[0]])
        for y in self.background[self.image_path]:
            if self.algs.images_segmented[self.image_path][y[1]][y[0]] not in background:
                background.append(self.algs.images_segmented[self.image_path][y[1]][y[0]])
        self.algs.set_fg_segments(self.image_path, foreground)
        self.algs.set_bg_segments(self.image_path, background)

    """
    Clears all markings and updates display
    """

    def clear_markings(self):
        self.foreground[self.image_path].clear()
        self.background[self.image_path].clear()
        draw_markings(self.image_path, self.image, self.background, self.foreground, self.algs, self.superImage)
        draw_bounds(self.image_path, self.superImage, self.algs, self.foreground, self.background)
        self.image.update()
        self.superImage.update()
        self.result_image.update()

    """
    Compute the graph cut and draw the results.
    """

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

    """
    Compute the kmeans and draw the results.
    """

    def kmeans(self):
        self.algs.perform_k_means_clustering(num_clusters=self.kclustervalue.value())
        self.result = "kmeans"
        self.draw_kmeans()
        self.enable_buttons(3)
        if self.gt_originalRadioButton.isEnabled():
            self.gt_overlapRadioButton.setEnabled(True)

    """
    Update the kmeans display when a different mode is selected
    """

    def update_kmeans(self):
        if self.bwkRadioButton.isChecked():
            self.draw_kmeans()

    """
    Sets what mode options are available depending on the type of co-segmentation selected
    """

    def clustering_options(self):
        if self.clusteringBox.currentIndex() == 0:
            self.kmeansFrame.setHidden(True)
            self.graphFrame.setVisible(True)
        elif self.clusteringBox.currentIndex() == 1:
            self.kmeansFrame.setVisible(True)
            self.graphFrame.setHidden(True)

    """
    Create scatter plot corresponding to a downscaled version of the feature vector using MDS
    """

    def create_plot(self):
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
        scatter.sigClicked.connect(self.on_click_plot)
        self.view.addItem(scatter)

    """
    When clicking on a scatterplot point it will highlight as well as the corresponding superpixel in the superpixel
    image
    """

    def on_click_plot(self, _, points):
        on_click_plot(self.plotMarked, self.view, self.mdsData, self.image_path, self.superImage, self.algs,
                      self.foreground, self.background, _, points)

    """
    Checks if enough superpixels are marked then calculates the gaussian mixture model for graph cut as well as
    the uncertainty for nodes and edge
    """

    def set_gmm(self):
        self.set_markings()
        totalForeground = 0
        totalBackground = 0
        for i in self.algs.images_superpixels_foreground.values():
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

    """
    Unfinished code for exporting and importing XML for settings. To be worked on in future work
    """

    def write_xml_settings(self):
        root = ET.Element("root")
        settings = ET.SubElement("settings")
        ET.SubElement(settings, "super_pixel_quantity")

        tree = ET.ElementTree(root)
        tree.write(settings.xml)

    """
    When changing features will disable  the unused and enable the used parameters for the feature vector
    """

    def change_features(self):
        change_features(self.colorLabel1, self.colorLabel2, self.colorLabel3, self.colorLabel4, self.HHist, self.SHist,
                        self.HSHHist, self.HSSHist, self.siftlabel, self.siftKeyPoint, self.entropyLabel_1,
                        self.entropyLabel_2, self.HentropyCheckBox, self.SentropyCheckBox, self.windowHogLabel,
                        self.winSize, self.blockHogLabel, self.blockSize, self.blockStrideHogLabel, self.blockStride,
                        self.cellHogLabel, self.cellSize, self.binsHogLabel, self.hogBins,
                        self.featureSelected.currentText())

    """
    When changing features will disable  the unused and enable the used modes for co-segmentation display
    """

    def change_cosegmentation(self):
        change_cosegmentation(self.graphCutModeLabel, self.graphCutModeFrame, self.clusterModeLabel,
                              self.clusterModeFrame,
                              self.clusteringBox.currentText())

    """
    Enable all available buttons for the current step of co-segmentation
    """

    def enable_buttons(self, option):
        enable_buttons(self.superpixelButton, self.clearMarkingsButton, self.histogramButton, self.drawRadioButton,
                       self.histogramRadioButton, self.graph_button, self.GMMButton, self.kmeansButton,
                       self.bwkRadioButton, self.colorRadioButton, self.graphcutButton, self.edgeRadioButton,
                       self.nodeRadioButton, self.graphRadioButton, self.bwRadioButton, self.bRadioButton, option)

    """
    Simply disables all buttons. To be used in combination with enabble_buttons though not used in all scenarios
    """

    def disable_buttons(self):
        disable_buttons(self.superpixelButton, self.histogramButton, self.histogramRadioButton, self.drawRadioButton,
                        self.graph_button, self.GMMButton, self.graphcutButton, self.kmeansButton,
                        self.clearMarkingsButton, self.edgeRadioButton, self.nodeRadioButton, self.bwRadioButton,
                        self.bRadioButton, self.graphRadioButton)

    """
    draws uncertainties on image in uncertainty tab
    """

    def draw_uncertainties(self):
        draw_uncertainties(self.image_path, self.uncertainty_image, self.edgeRadioButton.isChecked(), self.algs,
                           self.nodeRadioButton.isChecked(), self.graphRadioButton.isChecked())

    """
    draws kmeans on image in results tab
    """

    def draw_kmeans(self):
        draw_kmeans(self.result, self.image_path, self.result_image, self.algs, self.ColorRadioButton.isChecked(),
                    self.k1, self.k2,
                    self.k3, self.k4, self.k5, self.k6, self.k7, self.k8, self.k9, self.k10, self.k11, self.k12,
                    self.k13, self.k14, self.k15, self.k16)

    def draw_graph_cut(self):
        draw_graph_cut(self.result, self.image_path, self.bwRadioButton.isChecked(), self.result_image,
                                   self.algs,
                                   self.bRadioButton.isChecked())

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    fb = mainUI()
    fb.show()
    app.exec_()
