# Interactive image co-segmentation
A Python program that can be used to interactively perform co-segmentation through a graphical user interface. This program is part of a collaborative Bachelor's project between Marco Lu & Niek de Vries. More information about the program can be found in the Bachelor's theses "Interactive image co-segmentation" and "User-driven image co-segmentation" by Marco and Niek respectively.

## Required packages
This section lists the packages that are required to run the program. This could be done using `pip install`.
* PyQt 5
* pyqtgraph
* opencv-contrib-python (Using version 3.4.2.17, as newer versions do not include SIFT. Use `pip install opencv-contrib-python==3.4.2.17`)
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
