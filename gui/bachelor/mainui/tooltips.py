def set_tooltips(superpixelCalculationLabel, superpixelQuantityLabel, iterationsLabel, sigmaLabel,
                 featureExtractionLabel, RGB, HSV):
    superpixelCalculationLabel.setToolTip("Calculate the superpixels for all images in the folder using SLIC")
    superpixelQuantityLabel.setToolTip("Set the amount of superpixels each image will have")
    iterationsLabel.setToolTip("Maximum number of iterations of k-means")
    sigmaLabel.setToolTip("Width of Gaussian smoothing kernel for optional pre-processing. 0 for no smoothing")
    featureExtractionLabel.setToolTip(
        "Compute feature vectors for each superpixel consisting of the selected features")
    RGB.setToolTip("Means of RGB")
    HSV.setToolTip("Means of HSV")