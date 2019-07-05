from PyQt5 import QtCore, QtWidgets

from mainui.tooltips import set_tooltips

"""
Connects all buttons and sets tooltip
"""


def initialize(treeView, context_menu, listWidget, choose_image, clearMarkingsButton, clear_markings, histogramButton,
               set_histograms, superpixelButton, calculate_superpixels, graphcutButton, compute_graph_cut, kmeansButton,
               kmeans, graph_button, create_graph, GMMButton, set_gmm, bwRadioButton, draw_graph_cut, bRadioButton,
               colorRadioButton, draw_kmeans, bwkRadioButton, edgeRadioButton, draw_uncertainties, graphRadioButton,
               nodeRadioButton, gt_originalRadioButton, draw_gt, gt_overlapRadioButton, k1, update_kmeans, k2, k3, k4,
               k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, featureSelected, change_features, clusteringBox,
               change_clustering, clustering_options, kmeansFrame, fgRadioButton, currentPencil, bgRadioButton,
               superpixelCalculationLabel, superpixelQuantityLabel, iterationsLabel, compactnessLabel, sigmaLabel,
               featureExtractionLabel, RGB, HSV, Hue, Saturation, HueSat, Sift, Hog, currentLabel, colorLabel1,
               colorLabel2, colorLabel3, colorLabel4, entropyLabel_1, entropyLabel_2, siftlabel, windowHogLabel,
               blockStrideHogLabel, cellHogLabel, binsHogLabel, cosegmentationLabel, componentLabel, restartLabel,
               pairwiseLabel, scaleLabel, clusterLabel, clusterFGLabel, markingLabel, superpixelLabel, uncertaintyLabel,
               graphCutModeLabel, clusterModeLabel, gtLabel, superpixelProgress, featureProgress,
               cosegmentationProgress):
    treeView.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
    treeView.customContextMenuRequested.connect(context_menu)

    listWidget.setViewMode(QtWidgets.QListView.IconMode)
    listWidget.setIconSize(QtCore.QSize(64, 64))
    listWidget.currentItemChanged.connect(choose_image)
    listWidget.setMovement(0)

    clearMarkingsButton.clicked.connect(clear_markings)
    histogramButton.clicked.connect(set_histograms)
    superpixelButton.clicked.connect(calculate_superpixels)
    graphcutButton.clicked.connect(compute_graph_cut)
    kmeansButton.clicked.connect(kmeans)
    graph_button.clicked.connect(create_graph)
    GMMButton.clicked.connect(set_gmm)
    bwRadioButton.clicked.connect(draw_graph_cut)
    bRadioButton.clicked.connect(draw_graph_cut)
    colorRadioButton.clicked.connect(draw_kmeans)
    bwkRadioButton.clicked.connect(draw_kmeans)
    edgeRadioButton.clicked.connect(draw_uncertainties)
    graphRadioButton.clicked.connect(draw_uncertainties)
    nodeRadioButton.clicked.connect(draw_uncertainties)

    gt_originalRadioButton.clicked.connect(draw_gt)
    gt_overlapRadioButton.clicked.connect(draw_gt)

    k1.stateChanged.connect(update_kmeans)
    k2.stateChanged.connect(update_kmeans)
    k3.stateChanged.connect(update_kmeans)
    k4.stateChanged.connect(update_kmeans)
    k5.stateChanged.connect(update_kmeans)
    k6.stateChanged.connect(update_kmeans)
    k7.stateChanged.connect(update_kmeans)
    k8.stateChanged.connect(update_kmeans)
    k9.stateChanged.connect(update_kmeans)
    k10.stateChanged.connect(update_kmeans)
    k11.stateChanged.connect(update_kmeans)
    k12.stateChanged.connect(update_kmeans)
    k13.stateChanged.connect(update_kmeans)
    k14.stateChanged.connect(update_kmeans)
    k15.stateChanged.connect(update_kmeans)
    k16.stateChanged.connect(update_kmeans)

    featureSelected.currentIndexChanged.connect(change_features)
    clusteringBox.currentIndexChanged.connect(change_clustering)

    clusteringBox.currentIndexChanged.connect(clustering_options)
    kmeansFrame.setHidden(True)

    fgRadioButton.clicked.connect(currentPencil)
    bgRadioButton.clicked.connect(currentPencil)
    set_tooltips(superpixelCalculationLabel, superpixelQuantityLabel, iterationsLabel,
                 compactnessLabel, sigmaLabel, featureExtractionLabel, RGB, HSV, Hue,
                 Saturation, HueSat, Sift, Hog, currentLabel, colorLabel1,
                 colorLabel2, colorLabel3, colorLabel4, entropyLabel_1, entropyLabel_2,
                 siftlabel, windowHogLabel, blockStrideHogLabel, cellHogLabel,
                 binsHogLabel, graph_button, cosegmentationLabel, componentLabel,
                 restartLabel, pairwiseLabel, scaleLabel, clusterLabel, clusterFGLabel,
                 markingLabel, superpixelLabel, uncertaintyLabel, nodeRadioButton,
                 edgeRadioButton, graphRadioButton, graphCutModeLabel, bwRadioButton,
                 bRadioButton, clusterModeLabel, bwkRadioButton, colorRadioButton, gtLabel,
                 gt_overlapRadioButton, gt_originalRadioButton, superpixelProgress,
                 featureProgress, cosegmentationProgress)
