def set_tooltips(superpixelCalculationLabel, superpixelQuantityLabel, iterationsLabel, compactnessLabel, sigmaLabel,
                 featureExtractionLabel, RGB, HSV, Hue, Saturation, HueSat, Sift, Hog, currentLabel, colorLabel1,
                 colorLabel2, colorLabel3, colorLabel4, entropyLabel_1, entropyLabel_2, siftlabel, windowHogLabel,
                 blockStrideHogLabel, cellHogLabel, binsHogLabel, graph_button, cosegmentationLabel, componentLabel,
                 restartLabel, pairwiseLabel, scaleLabel, clusterLabel, clusterFGLabel, markingLabel, superpixelLabel,
                 uncertaintyLabel, nodeRadioButton, edgeRadioButton, graphRadioButton, graphCutModeLabel, bwRadioButton,
                 bRadioButton, clusterModeLabel, bwkRadioButton, colorRadioButton, gtLabel, gt_overlapRadioButton,
                 gt_originalRadioButton, superpixelProgress, featureProgress, cosegmentationProgress):
    superpixelCalculationLabel.setToolTip("Calculate the superpixels for all images in the folder using SLIC")

    superpixelQuantityLabel.setToolTip("Set the approximate amount of superpixels each image will have")
    iterationsLabel.setToolTip("Maximum number of iterations of k-means")
    compactnessLabel.setToolTip(
        "Balances color proximity and space proximity. Higher results in more square superpixels")
    sigmaLabel.setToolTip("Width of Gaussian smoothing kernel for optional pre-processing. 0 for no smoothing")

    featureExtractionLabel.setToolTip(
        "Compute feature vector for each superpixel consisting of the selected features")

    RGB.setToolTip("Means of RGB")
    HSV.setToolTip("Means of HSV")
    Hue.setToolTip("Histogram of hue")
    Saturation.setToolTip("Include a histogram of saturation")
    HueSat.setToolTip("Flattened 2D histogram of Hue and saturation")
    Sift.setToolTip("Include a SIFT descriptor")
    Hog.setToolTip("Include a HOG descriptor")
    currentLabel.setToolTip("Currently viewing parameters of set feature")

    colorLabel1.setToolTip("Number of bins for hue histogram")
    colorLabel2.setToolTip("Number of bins for saturation histogram")
    colorLabel3.setToolTip("Number of Hue bins for hue/saturation histogram")
    colorLabel4.setToolTip("Number of saturation bins for hue/saturation histogram")
    entropyLabel_1.setToolTip("Include entropy of hue histogram")
    entropyLabel_2.setToolTip("Include entropy of saturation histogram")
    siftlabel.setToolTip("Diameter of the keypoint neighborhood")
    windowHogLabel.setToolTip(
        "Size in pixels of the detection window in (width, height) of the HOG descriptor")
    blockStrideHogLabel.setToolTip("Block size of the HOG descriptor in pixels. Align to cell size")
    cellHogLabel.setToolTip("Cell size of the HOG descriptor")
    binsHogLabel.setToolTip("Number of bins per cell of the HOG descriptor")
    graph_button.setToolTip("Transforms feature vector to two dimensions and creates a scatterplot")

    cosegmentationLabel.setToolTip("Classify each superpixel in each image")
    componentLabel.setToolTip("Try to fit a model for these amounts of components")
    restartLabel.setToolTip("Number of restarts per number of components")
    pairwiseLabel.setToolTip("Used to scale the pairwise term in relation to the unary term")
    scaleLabel.setToolTip(
        "Used to adjust the strength of the response in the pairwise term value depending on distance")

    clusterLabel.setToolTip("Desired amount of clusters")
    clusterFGLabel.setToolTip("Set which clusters belong to foreground. Used for ground truth compare")

    markingLabel.setToolTip("Set user suggested fg bg superpixels. Used for graph-cut")

    superpixelLabel.setToolTip("Set whether clicking on images will interact with the MDS plot or draw")

    uncertaintyLabel.setToolTip("Set which uncertainty is displayed")
    nodeRadioButton.setToolTip("Based on entropy of node beliefs")
    edgeRadioButton.setToolTip(
        "Based on the entropy of the proportion of its K (=10) nearest neighbours of marked superpixels")
    graphRadioButton.setToolTip(
        "Based on the difference in energy between the optimal assignment and opposite")

    graphCutModeLabel.setToolTip("Set which display mode for graph-cut.")
    bwRadioButton.setToolTip("Makes background black and foreground white")
    bRadioButton.setToolTip("Creates a border that segments foreground and background")

    clusterModeLabel.setToolTip("Set which display mode for kmeans")
    bwkRadioButton.setToolTip("Makes background black and foreground white based on user set foreground")
    colorRadioButton.setToolTip("Each segment gets a differnt color")

    gtLabel.setToolTip("Set display mode for ground truth")
    gt_overlapRadioButton.setToolTip(
        "Overlaps result with ground truth. Black areas are where the comparison does not match")
    gt_originalRadioButton.setToolTip("THe original set ground truth image")

    superpixelProgress.setToolTip(
        "Right click on a folder and select on an image then press calculate superpixels")
    featureProgress.setToolTip("Select features to include then adjust parameters and calculate")
    cosegmentationProgress.setToolTip(
        "Set the parameters for co-segmentation and then perform the co-segmentation")