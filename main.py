import sys
import os
import PySide2

import time
# # from PyQt5.QtWidgets import QApplication, QPushButton, QLabel
# # from PyQt5.QtCore import pyqtSlot
dirname = os.path.dirname(PySide2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
# import random
# from PySide2 import QtCore, QtWidgets, QtGui
# import vtk
# from vtk.qt4 import QVTKRenderWindowInteractor
#
#
# class MyWidget(QtWidgets.QWidget):
#     def __init__(self):
#         super().__init__()
#
#         self.hello = ["Hallo Welt", "Hei maailma", "Hola Mundo", "Привет мир"]
#
#         self.button = QtWidgets.QPushButton("Click me!")
#         self.text = QtWidgets.QLabel("Hello World")
#         self.text.setAlignment(QtCore.Qt.AlignCenter)
#
#         self.layout = QtWidgets.QVBoxLayout()
#         self.layout.addWidget(self.text)
#
#         self.layout.addWidget(self.button)
#         self.setLayout(self.layout)
#
#         self.button.clicked.connect(self.magic)
#
#         self.frame = QtWidgets.QFrame()
#         self.vtkWidget = QVTKRenderWindowInteractor()
#         self.layout.addWidget(self.vtkWidget)
#
#         self.ren = vtk.vtkRenderer()
#         self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
#         self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
#         self.dicomreader = vtk.vtkDICOMImageReader()
#
#
#     @QtCore.Slot()
#     def magic(self):
#         self.text.setText(random.choice(self.hello))
#         dirname = QtWidgets.QFileDialog.getExistingDirectory(self, "DICOM文件夹", '\home', QtWidgets.QFileDialog.ShowDirsOnly)
#         self.dicomreader.SetDirectoryName(dirname)
#         self.dicomreader.Update()
#         self.imageViewer = vtk.vtkImageViewer2()
#         self.imageViewer.SetInputConnection(self.dicomreader.GetOutputPort())
#         self.renderWindowInteractor = vtk.vtkRenderWindowInteractor(self.iren)
#
#
#
#
#
# if __name__ == "__main__":
#     app = QtWidgets.QApplication([])
#
#     widget = MyWidget()
#     widget.resize(800, 600)
#     widget.show()
#
#     sys.exit(app.exec_())

import vtk
from PySide2.QtWidgets import QMainWindow, QPushButton, QFrame, QVBoxLayout, QHBoxLayout, QFileDialog, QApplication, QMessageBox, QLineEdit, QLabel
from PySide2.QtCore import Slot, QDir, Qt
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk.util import numpy_support
from vesselfit import capture_mouse



class MainWindow(QMainWindow):

    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setWindowTitle("股浅动脉自动提取")
        self.setMinimumSize(800, 600)

        self.open_directory_button = QPushButton("打开文件夹")
        self.open_directory_button.clicked.connect(self.open_directory_button_clicked)
        self.frame = QFrame()

        self.main_layout = QVBoxLayout()
        self.path_layout = QHBoxLayout()
        self.path_edit = QLineEdit(self)
        # self.pathtext.setMaximumHeight(30)
        self.path_layout.addWidget(self.path_edit)
        self.path_layout.addWidget(self.open_directory_button)
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.main_layout.addLayout(self.path_layout)
        self.slice_num_label = QLabel(self)
        self.slice_num_label.setAlignment(Qt.AlignCenter)
        self.imageviewer = vtk.vtkImageViewer2()
        label_font = self.slice_num_label.font()
        label_font.setPointSize(20)
        self.slice_num_label.setFont(label_font)
        self.button_layout = QHBoxLayout()
        self.non_occluded_button = QPushButton("选取非闭塞段端点")
        self.non_occluded_button.clicked.connect(self.non_occluded_button_clicked)
        self.button_layout.addWidget(self.non_occluded_button)
        self.main_layout.addWidget(self.slice_num_label)
        self.main_layout.addWidget(self.vtkWidget)
        self.main_layout.addLayout(self.button_layout)
        # self.LastPickedActor = None
        # self.LastPickedProperty = vtk.vtkProperty()

        self.dicomreader = vtk.vtkDICOMImageReader()
        self.ren = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.actor = vtk.vtkImageActor()
        self.ren.ResetCamera()
        self.frame.setLayout(self.main_layout)
        self.setCentralWidget(self.frame)

        self.show()
        self.iren.Initialize()

    @Slot()
    def open_directory_button_clicked(self):
        dirname = QFileDialog.getExistingDirectory(self, "DICOM文件夹", '\home',
                                                             QFileDialog.ShowDirsOnly)
        while len(dirname) > 0:
            tempdir = QDir(dirname)
            allfile = tempdir.entryList(filters=QDir.Files)

            if len(allfile) <= 100:
                msgbtn = QMessageBox.warning(self, "警告", "文件夹中数量过少，是否重新选择？\n文件路径:"+dirname,
                                             QMessageBox.Ok|QMessageBox.Cancel)
                if msgbtn == QMessageBox.Ok:
                    dirname = QFileDialog.getExistingDirectory(self, "DICOM文件夹", '\home',
                                                               QFileDialog.ShowDirsOnly)
                else:
                    break
            else:
                break
        self.path_edit.setText(dirname)
        self.dicomreader.SetDirectoryName(dirname)
        self.imageviewer.SetInputConnection(self.dicomreader.GetOutputPort())
        self.imageviewer.SetupInteractor(self.iren)
        self.imageviewer.SetRenderWindow(self.vtkWidget.GetRenderWindow())
        self.iren.RemoveObservers('MouseWheelBackwardEvent')
        self.iren.RemoveObservers('MouseWheelForwardEvent')
        self.iren.AddObserver('LeftButtonPressEvent', self.left_pressed)
        self.iren.AddObserver('MouseWheelBackwardEvent', self.last_slice)
        self.iren.AddObserver('MouseWheelForwardEvent', self.next_slice)
        self.dicomreader.Update()
        self.ren.ResetCamera()
        self.imageviewer.GetImageActor().SetInterpolate(False)
        self.iren.Initialize()
        self.iren.Start()
        self.imageviewer.Render()
        self.slice_num_label.setText(str(self.imageviewer.GetSlice()))

    def last_slice(self, obj, ev):
        self.imageviewer.SetSlice((self.imageviewer.GetSlice() - 1) % self.imageviewer.GetSliceMax())
        self.slice_num_label.setText(str(self.imageviewer.GetSlice()))

    def next_slice(self, obj, ev):
        self.imageviewer.SetSlice((self.imageviewer.GetSlice() + 1) % self.imageviewer.GetSliceMax())
        self.slice_num_label.setText(str(self.imageviewer.GetSlice()))

    def left_pressed(self, obj, ev):
        clickpos = self.iren.GetEventPosition()
        # picker = vtk.vtkPointPicker()
        # picker.Pick(clickpos[0], clickpos[1], 0, self.ren)
        # self.NewPickedActor = picker.GetActor()
        # if self.NewPickedActor:
        #     if self.LastPickedActor:
        #         self.LastPickedActor.GetProperty.DeepCopy(self.LastPickedProperty)
        #     self.LastPickedProperty.DeepCopy(self.NewPickedActor.GetProperty())
        #     self.NewPickedActor.GetProperty().SetColor(vtk.vtkNamedColors().GetColor('Red'))
        #     self.LastPickedActor = self.NewPickedActor

    def non_occluded_button_clicked(self):
        image_data = numpy_support.vtk_to_numpy(self.dicomreader.GetOutput().GetPointData().GetScalars())
        capture_mouse(image_data)





if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()

    sys.exit(app.exec_())
