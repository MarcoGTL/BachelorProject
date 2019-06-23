from PyQt5 import QtWidgets


def errormessage(text,informativetext):
    msg = QtWidgets.QMessageBox()
    msg.setIcon(QtWidgets.QMessageBox.Critical)
    msg.setText(text)
    msg.setInformativeText(informativetext)
    msg.setWindowTitle("Error")
    msg.exec_()