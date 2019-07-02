import os

from PyQt5 import QtWidgets, QtGui, QtCore


def context_menu(model, treeView, select_folder, listWidget, select_gt):
    if model.isDir(treeView.currentIndex()):
        menu = QtWidgets.QMenu()
        open = menu.addAction("select folder")
        open.triggered.connect(select_folder)
        cursor = QtGui.QCursor()
        menu.exec_(cursor.pos())
    elif listWidget.currentItem() is not None:
        menu = QtWidgets.QMenu()
        open = menu.addAction("select image as ground truth")
        open.triggered.connect(select_gt)
        cursor = QtGui.QCursor()
        menu.exec_(cursor.pos())


def populate(currentFolder, model, treeView):
    path = os.getcwd() + '/' + currentFolder
    model.setRootPath(QtCore.QDir.rootPath())
    treeView.setModel(model)
    treeView.setRootIndex(model.index(path))
    treeView.setSortingEnabled(True)
    treeView.hideColumn(1)
    treeView.hideColumn(2)
    treeView.hideColumn(3)