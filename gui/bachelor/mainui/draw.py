from PyQt5 import QtGui, QtCore
from skimage.segmentation import find_boundaries

from compare_pixel import compare_pixel
from error import errormessage


def draw_uncertainties(image_path, uncertainty_image, edge, algs, node, graph):
    pixmap = QtGui.QPixmap(image_path)
    uncertainty_image.setPixmap(pixmap)
    if edge:
        x = algs.images_superpixels_uncertainties_edge[image_path]
    elif node:
        x = algs.images_superpixels_uncertainties_node[image_path]
        print(x)
    elif graph:
        x = [1 - i for i in algs.images_superpixels_uncertainties_graph_cut[image_path]]
    uncertainty_image.setPixmap(pixmap)
    qp = QtGui.QPainter(uncertainty_image.pixmap())
    i = 0
    j = 0
    lengthy = pixmap.height()
    lengthx = pixmap.width()
    print(lengthy, lengthx)
    while i < lengthy:
        while j < lengthx:
            color = round(x[algs.images_segmented[image_path][i][j]] * 1020)
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

    uncertainty_image.update()


def draw_bounds(image_path, superImage, algs, foreground, background):
    pixmap = QtGui.QPixmap(image_path)
    superImage.setPixmap(pixmap)
    if algs.images_segmented[image_path] is None:
        return
    qp = QtGui.QPainter(superImage.pixmap())
    qp.setPen(QtGui.QPen(QtCore.Qt.red, 1))
    for x in foreground[image_path]:
        qp.drawPoint(x[0], x[1])
    qp.setPen(QtGui.QPen(QtCore.Qt.blue, 1))
    for x in background[image_path]:
        qp.drawPoint(x[0], x[1])
    qp.setPen(QtGui.QPen(QtCore.Qt.yellow, 2))
    boundaries = algs.get_superpixel_borders_mask(image_path)
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
    superImage.update()


def draw_gt(file_path, image, groundtruth, image_path, gt_originalRadioButton, compare_image, algs, result, k1, k2, k3,
            k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, gtpercentage):
    pixmap = QtGui.QPixmap(file_path)

    if image.pixmap().size() != pixmap.size():
        errormessage("Wrong gt selected", "Size of images do not match")
        return

    groundtruth[image_path] = file_path

    if gt_originalRadioButton.isChecked():
        compare_image.setPixmap(pixmap)
        compare_image.update()
    else:
        compare_image.setPixmap(pixmap)
        compare_image.pixmap().fill(QtCore.Qt.black)
        qp = QtGui.QPainter(compare_image.pixmap())
        qp.setPen(QtGui.QPen(QtCore.Qt.white, 1))
        gt = pixmap.toImage()

        results = algs.images_cosegmented[image_path]

        y = 0
        x = 0
        print(gt.width(), gt.height())
        wrong = 0
        if result == "kmeans":
            table = [k1.isChecked(), k2.isChecked(), k3.isChecked(), k4.isChecked(),
                     k5.isChecked(), k6.isChecked(), k7.isChecked(), k8.isChecked(),
                     k9.isChecked(), k10.isChecked(), k11.isChecked(), k12.isChecked(),
                     k13.isChecked(), k14.isChecked(), k15.isChecked(), k16.isChecked(), ]
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
        elif result == "graphcut":
            while y < gt.height():
                while x < gt.width():
                    gtpixel = gt.pixelColor(x, y).getRgb()
                    compare = results[y][x]
                    if (gtpixel[0] != 0):
                        print(gtpixel[0], gtpixel[1], gtpixel[2])
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
        right = ((compare_image.pixmap().width() * compare_image.pixmap().height()) - wrong) / (
                compare_image.pixmap().width() * compare_image.pixmap().height()) * 100
        print(right)
        gtpercentage.setText(str(round(right)) + "%")
        compare_image.update()


def draw_plot_marked(image_path, superImage, algs, foreground, background, graphMarked):
    draw_bounds(image_path, superImage, algs, foreground, background)
    print(graphMarked)
    qp = QtGui.QPainter(superImage.pixmap())
    qp.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0, 25), 3))
    i = 0
    j = 0
    array = algs.images_segmented[image_path]
    lengthy = len(array)
    lengthx = len(array[0])
    while i < lengthy:
        while j < lengthx:
            if array[i][j] in graphMarked:
                qp.drawPoint(j, i)
            j = j + 1
        j = 0
        i = i + 1
    superImage.update()


def draw_kmeans(result, image_path, result_image, algs, colorRadioButton, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11,
                k12, k13, k14, k15, k16):
    if result == "kmeans":
        pixmap = QtGui.QPixmap(image_path)

        result_image.setPixmap(pixmap)
        results = algs.images_cosegmented[image_path]
        qp = QtGui.QPainter(result_image.pixmap())
        i = 0
        j = 0
        lengthy = len(results)
        lengthx = len(results[0])
        if colorRadioButton.isChecked():
            color = [[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255],
          [255, 0, 255], [192, 192, 192], [128, 128, 128], [128, 0, 0], [128, 128, 0], [0, 128, 0], [128, 0, 128],
          [0, 128, 128], [0, 0, 128]]
        else:
            color = [(k1.isChecked() * 255, k1.isChecked() * 255, k1.isChecked() * 255),
                     (k2.isChecked() * 255, k2.isChecked() * 255, k2.isChecked() * 255),
                     (k3.isChecked() * 255, k3.isChecked() * 255, k3.isChecked() * 255),
                     (k4.isChecked() * 255, k4.isChecked() * 255, k4.isChecked() * 255),
                     (k5.isChecked() * 255, k5.isChecked() * 255, k5.isChecked() * 255),
                     (k6.isChecked() * 255, k6.isChecked() * 255, k6.isChecked() * 255),
                     (k7.isChecked() * 255, k7.isChecked() * 255, k7.isChecked() * 255),
                     (k8.isChecked() * 255, k8.isChecked() * 255, k8.isChecked() * 255),
                     (k9.isChecked() * 255, k9.isChecked() * 255, k9.isChecked() * 255),
                     (k10.isChecked() * 255, k10.isChecked() * 255, k10.isChecked() * 255),
                     (k11.isChecked() * 255, k11.isChecked() * 255, k11.isChecked() * 255),
                     (k12.isChecked() * 255, k12.isChecked() * 255, k12.isChecked() * 255),
                     (k13.isChecked() * 255, k13.isChecked() * 255, k13.isChecked() * 255),
                     (k14.isChecked() * 255, k14.isChecked() * 255, k14.isChecked() * 255),
                     (k15.isChecked() * 255, k15.isChecked() * 255, k15.isChecked() * 255),
                     (k16.isChecked() * 255, k16.isChecked() * 255, k16.isChecked() * 255)]
        while i < lengthy:
            while j < lengthx:
                qp.setPen(QtGui.QColor(color[results[i][j]][0], color[results[i][j]][1], color[results[i][j]][2]))
                qp.drawPoint(j, i)
                j = j + 1
            j = 0
            i = i + 1
        result_image.update()


def draw_markings(image_path, image, background, foreground, algs, superImage):
    pixmap = QtGui.QPixmap(image_path)
    image.setPixmap(pixmap)
    qp = QtGui.QPainter(image.pixmap())
    qp.setPen(QtGui.QPen(QtCore.Qt.blue, 1))
    for x in background[image_path]:
        qp.drawPoint(x[0], x[1])

    qp.setPen(QtGui.QPen(QtCore.Qt.red, 1))
    for x in foreground[image_path]:
        qp.drawPoint(x[0], x[1])

    if image_path in algs.images_segmented:
        qpbounds = QtGui.QPainter(superImage.pixmap())
        qpbounds.setPen(QtGui.QPen(QtCore.Qt.blue, 1))
        for x in background[image_path]:
            qpbounds.drawPoint(x[0], x[1])

        qpbounds.setPen(QtGui.QPen(QtCore.Qt.red, 1))
        for x in foreground[image_path]:
            qpbounds.drawPoint(x[0], x[1])
    image.update()
    superImage.update()


def draw_graph_cut(result, image_path, bwRadioButton, result_image, algs, bRadioButton):
    if result == "graphcut":
        pixmap = QtGui.QPixmap(image_path)
        if bwRadioButton.isChecked():
            pixmap.fill(QtCore.Qt.black)
        result_image.setPixmap(pixmap)
        results = algs.images_cosegmented[image_path]
        qp = QtGui.QPainter(result_image.pixmap())
        if bwRadioButton.isChecked():
            qp.setPen(QtGui.QPen(QtCore.Qt.white, 1))
        elif bRadioButton.isChecked():
            qp.setPen(QtGui.QPen(QtCore.Qt.yellow, 2))
            results = find_boundaries(results)
        y = 0
        x = 0
        lengthy = len(results)
        lengthx = len(results[0])
        while y < lengthy:
            while x < lengthx:
                if results[y][x] == 1:
                    qp.drawPoint(x, y)
                x = x + 1
            x = 0
            y = y + 1
        result_image.update()