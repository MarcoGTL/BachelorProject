from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore
from mainui import designer
from algorithms import slic
import os


class MyFileBrowser(designer.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self):
        super(MyFileBrowser, self).__init__()
        self.setupUi(self)
        self.treeView.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.treeView.customContextMenuRequested.connect(self.context_menu)
        self.listWidget.setViewMode(QtWidgets.QListView.IconMode)
        self.listWidget.setIconSize(QtCore.QSize(64,64))
        self.listWidget.currentItemChanged.connect(self.choose_image)
        self.point = (-1, -1) #current point
        self.pointlist = [] # contains all marked points
        self.superpixel = [] # contains for all pixels in picture the superpixel it belongs to
        self.calculateButton.clicked.connect(self.calculate)
        self.superpixelcheck.clicked.connect(self.change_super_pixel)
        self.superpixelspin.valueChanged.connect(self.change_super_pixel)
        self.superpixelcheck.setDisabled(True)
        self.calculateButton.setDisabled(True)
        self.img_float = []
        self.image_path = ""
        self.populate()


    def populate(self):
        path = os.getcwd() + "/images"
        self.model = QtWidgets.QFileSystemModel()
        self.model.setRootPath(QtCore.QDir.rootPath())
        self.treeView.setModel(self.model)
        self.treeView.setRootIndex(self.model.index(path))
        self.treeView.setSortingEnabled(True)

    def context_menu(self):
        menu = QtWidgets.QMenu()
        open = menu.addAction("select folder")
        open.triggered.connect(self.select_folder)
        cursor = QtGui.QCursor()
        menu.exec_(cursor.pos())

    def select_folder(self):
        self.listWidget.clear()
        self.image.clear()
        index = self.treeView.currentIndex()
        file_path = self.model.filePath(index)
        for file in os.listdir(file_path):
            item = QtWidgets.QListWidgetItem()
            icon = QtGui.QIcon()
            pixmap = QtGui.QPixmap(file_path+'/'+file)  # QImage object
            icon.addPixmap(pixmap, QtGui.QIcon.Normal, QtGui.QIcon.Off)
            item.setIcon(icon)
            item.setText(file)
            self.listWidget.addItem(item)

    def choose_image(self):
        if self.listWidget.currentItem() is not None:
            self.pointlist.clear()
            index = self.treeView.currentIndex()
            self.image_path = self.model.filePath(index) + '/' + self.listWidget.currentItem().text()
            self.img_float = slic.read_image_as_float64(self.image_path)
            print(self.image_path)
            self.change_super_pixel()
            pixmap = QtGui.QPixmap(self.image_path)
            self.image.setPixmap(pixmap)
            self.image.mouseMoveEvent = self.mov
            self.image.mousePressEvent = self.mov
            self.image.mouseReleaseEvent = self.stopmov
            self.calculateButton.setEnabled(True)
            self.superpixelcheck.setEnabled(True)
        else:
            self.calculateButton.setEnabled(False)
            self.superpixelcheck.setEnabled(False)

    def change_super_pixel(self):
        if self.listWidget.currentItem() is not None:
            self.superpixel = slic.get_segmented_pixels(self.img_float, self.superpixelspin.value())
            if self.superpixelcheck.isChecked():
                self.draw_bounds()

    def mov(self, event):
        x = event.pos().x()
        y = event.pos().y()
        self.point = (x, y)
        self.update()
        self.image.update()

    def stopmov(self, event):
        self.point = (-1, -1)
        print("point list: ", *self.pointlist)

    def paintEvent(self, event):
        if self.point[0] != -1:
            qp = QtGui.QPainter(self.image.pixmap())
            self.pointlist.append(self.point)
            qp.setPen(QtGui.QPen(QtCore.Qt.red, 3))
            qp.drawPoint(self.point[0], self.point[1])

    def calculate(self):
        self.gets_slic()
        i = 0
        foreground = []
        positionprinter = set()
        while i < self.superpixelspin.value(): # change 100 later with slider
            foreground.append(False)
            i = i + 1
        for x in self.pointlist:
            position = self.superpixel[x[1]][x[0]]
            foreground[position] = True
            positionprinter.add(position)
        f = open("positions.txt","w")
        for x in positionprinter:
            f.write(str(x) + " ")
        f.close()

    def draw_bounds(self):
        pixmap = QtGui.QPixmap(self.image_path)
        self.image.setPixmap(pixmap)
        qp = QtGui.QPainter(self.image.pixmap())
        qp.setPen(QtGui.QPen(QtCore.Qt.red, 3))
        for x in self.pointlist:
            qp.drawPoint(x[0], x[1])
        if self.superpixelcheck.isChecked():
            qp.setPen(QtGui.QPen(QtCore.Qt.yellow, 3))
            boundaries = slic.find_boundaries(self.superpixel)
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
            self.image.update()
            print("hello")




if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    fb = MyFileBrowser()
    fb.show()
    app.exec_()
