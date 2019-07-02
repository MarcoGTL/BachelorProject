def change_features(colorLabel1, colorLabel2, colorLabel3, colorLabel4, HHist, SHist, HSHHist, HSSHist, siftlabel,
                    siftKeyPoint, entropyLabel_1, entropyLabel_2, HentropyCheckBox, SentropyCheckBox, windowHogLabel,
                    winSize, blockHogLabel, blockSize, blockStrideHogLabel, blockStride, cellHogLabel, cellSize,
                    binsHogLabel, hogBins, current_index):
    colorLabel1.setHidden(True)
    colorLabel2.setHidden(True)
    colorLabel3.setHidden(True)
    colorLabel4.setHidden(True)
    HHist.setHidden(True)
    SHist.setHidden(True)
    HSHHist.setHidden(True)
    HSSHist.setHidden(True)
    siftlabel.setHidden(True)
    siftKeyPoint.setHidden(True)
    entropyLabel_1.setHidden(True)
    entropyLabel_2.setHidden(True)
    HentropyCheckBox.setHidden(True)
    SentropyCheckBox.setHidden(True)
    windowHogLabel.setHidden(True)
    winSize.setHidden(True)
    blockHogLabel.setHidden(True)
    blockSize.setHidden(True)
    blockStrideHogLabel.setHidden(True)
    blockStride.setHidden(True)
    cellHogLabel.setHidden(True)
    cellSize.setHidden(True)
    binsHogLabel.setHidden(True)
    hogBins.setHidden(True)

    if current_index == "Hue":
        colorLabel1.setVisible(True)
        HHist.setVisible(True)
        entropyLabel_1.setVisible(True)
        HentropyCheckBox.setVisible(True)
    if current_index == "Saturation":
        colorLabel2.setVisible(True)
        SHist.setVisible(True)
        entropyLabel_2.setVisible(True)
        SentropyCheckBox.setVisible(True)
    if current_index == "Hue x Saturation":
        colorLabel3.setVisible(True)
        colorLabel4.setVisible(True)
        HSSHist.setVisible(True)
        HSHHist.setVisible(True)
    if current_index == "Sift":
        siftlabel.setVisible(True)
        siftKeyPoint.setVisible(True)
    if current_index == "Hog":
        windowHogLabel.setVisible(True)
        winSize.setVisible(True)
        blockHogLabel.setVisible(True)
        blockSize.setVisible(True)
        blockStrideHogLabel.setVisible(True)
        blockStride.setVisible(True)
        cellHogLabel.setVisible(True)
        cellSize.setVisible(True)
        binsHogLabel.setVisible(True)
        hogBins.setVisible(True)


def change_clustering(graphCutModeLabel, graphCutModeFrame, clusterModeLabel, clusterModeFrame,current_index):
    if current_index == "Graph cut":
        graphCutModeLabel.setVisible(True)
        graphCutModeFrame.setVisible(True)
        clusterModeLabel.setHidden(True)
        clusterModeFrame.setHidden(True)

    if current_index == "kmeans clustering":
        clusterModeLabel.setVisible(True)
        clusterModeFrame.setVisible(True)
        graphCutModeLabel.setHidden(True)
        graphCutModeFrame.setHidden(True)


def enable_buttons(superpixelButton, clearMarkingsButton, histogramButton, drawRadioButton, histogramRadioButton,
                   graph_button, GMMButton, kmeansButton, bwkRadioButton, colorRadioButton, graphcutButton,
                   edgeRadioButton, nodeRadioButton, graphRadioButton, bwRadioButton, bRadioButton, option):
    if option > 0:
        superpixelButton.setEnabled(True)
        clearMarkingsButton.setEnabled(True)
    if option > 1:
        histogramButton.setEnabled(True)
        drawRadioButton.setEnabled(True)
        histogramRadioButton.setEnabled(True)
    if option > 2:
        histogramRadioButton.setEnabled(True)
        graph_button.setEnabled(True)
        GMMButton.setEnabled(True)
        kmeansButton.setEnabled(True)
        bwkRadioButton.setEnabled(True)
        colorRadioButton.setEnabled(True)
    if option > 3:
        graphcutButton.setEnabled(True)
        edgeRadioButton.setEnabled(True)
        nodeRadioButton.setEnabled(True)
    if option > 4:
        graphRadioButton.setEnabled(True)
        bwRadioButton.setEnabled(True)
        bRadioButton.setEnabled(True)


def disable_buttons(superpixelButton, histogramButton, histogramRadioButton, drawRadioButton, graph_button, GMMButton,
                    graphcutButton, kmeansButton, clearMarkingsButton, edgeRadioButton, nodeRadioButton, bwRadioButton,
                    bRadioButton, graphRadioButton):
    superpixelButton.setDisabled(True)
    histogramButton.setDisabled(True)
    histogramRadioButton.setDisabled(True)
    drawRadioButton.setDisabled(True)
    histogramRadioButton.setDisabled(True)
    graph_button.setDisabled(True)
    GMMButton.setDisabled(True)
    graphcutButton.setDisabled(True)
    kmeansButton.setDisabled(True)
    clearMarkingsButton.setDisabled(True)
    edgeRadioButton.setDisabled(True)
    nodeRadioButton.setDisabled(True)
    bwRadioButton.setDisabled(True)
    bRadioButton.setDisabled(True)
    graphRadioButton.setDisabled(True)