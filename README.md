# Interactive image co-segmentation
## Required packages
* PyQt 5
* pyqtgraph
* opencv-contrib-python (Using version 3.4.2.17, as newer versions do not include SIFT)
* numpy
* PyMaxflow
* scikit-image
* scikit-learn
* scipy
* matplotlib

## How to run the program
The program with GUI can be run from the main method in ui.py.
The file pipeline.py contains a main method that can be used to perform the co-segmentation pipeline without the GUI.

## User Guide
To perform co-segmentation perform the following steps:
1. Right click on a folder in the file browser containing the images and press select folder.
1. Left click on an image thumbnail in the image browser to display it.
1. Compute superpixels.
1. Compute features.
   1. View the mds plot
1. Choose a co-segmentation algorithm.
1. Perform the co-segmentation.
1. View the results in the result tab.
   1. To compare with ground-truth images, navigate to the ground-truth image in the file browser and right click it.
