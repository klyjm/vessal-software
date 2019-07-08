import sys
import os
import PySide2
dirname = os.path.dirname(PySide2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
import vtk
from PySide2.QtWidgets import QMainWindow, QPushButton, QFrame, QVBoxLayout, QHBoxLayout, QFileDialog, QApplication, QMessageBox, QLineEdit, QLabel, QSlider
from PySide2.QtCore import Slot, QDir, Qt
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk.util import numpy_support
from vesselfit import get_vessel, display3d, slice_seg_contours
import SimpleITK as sitk
import cv2 as cv
import numpy as np



class MainWindow(QMainWindow):

    def __init__(self, parent=None):
        ###主窗体初始化###
        QMainWindow.__init__(self, parent)
        self.setWindowTitle("股浅动脉自动提取")
        self.setMinimumSize(800, 600)
        ###打开DICOM文件夹按钮###
        self.open_directory_button = QPushButton("打开文件夹")
        self.open_directory_button.clicked.connect(self.open_directory_button_clicked)
        ###路径框###
        self.path_edit = QLineEdit(self)
        ###slice计数标签###
        self.slice_num_label = QLabel(self)
        self.slice_num_label.setAlignment(Qt.AlignCenter)
        label_font = self.slice_num_label.font()
        label_font.setPointSize(20)
        self.slice_num_label.setFont(label_font)
        ###VTK显示控件###
        self.frame = QFrame()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.imageviewer = vtk.vtkImageViewer2()
        ###选取非闭塞段端点按钮###
        self.non_occluded_button = QPushButton("选取非闭塞段端点")
        self.non_occluded_button.clicked.connect(self.non_occluded_button_clicked)
        self.non_occluded_button.setEnabled(False)
        ###选取闭塞段控制点按钮###
        self.occluded_button = QPushButton("选取闭塞段控制点")
        self.occluded_button.clicked.connect(self.occluded_button_clicked)
        self.occluded_button.setEnabled(False)
        ###提取血管按钮###
        self.get_vessel_button = QPushButton("提取血管")
        self.get_vessel_button.clicked.connect(self.get_vessel_button_clicked)
        self.get_vessel_button.setEnabled(False)
        ###结果展示按钮###
        self.show_result_button = QPushButton("显示提取结果")
        self.show_result_button.clicked.connect(self.show_result_button_clicked)
        self.show_result_button.setVisible(False)
        ###patch保存按钮###
        self.save_patch_button = QPushButton("保存patch")
        self.save_patch_button.clicked.connect()
        self.save_patch_button.setVisible(False)
        ###滑块###
        self.sliber = QSlider()
        self.sliber.setVisible(False)
        self.sliber.setOrientation(Qt.Horizontal)
        self.sliber.valueChanged.connect(self.sliber_value_changed)
        ###路径布局###
        self.path_layout = QHBoxLayout()
        self.path_layout.addWidget(self.path_edit)
        self.path_layout.addWidget(self.open_directory_button)
        ###功能按钮布局###
        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(self.non_occluded_button)
        self.button_layout.addWidget(self.occluded_button)
        self.button_layout.addWidget(self.get_vessel_button)
        self.button_layout.addWidget(self.show_result_button)
        self.button_layout.addWidget(self.save_patch_button)
        ###主布局###
        self.main_layout = QVBoxLayout()
        self.main_layout.addLayout(self.path_layout)
        self.main_layout.addWidget(self.slice_num_label)
        self.main_layout.addWidget(self.sliber)
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
        self.seedpt = None
        self.seedpt_list = []
        self.seed_slice = 0
        self.seedpt3d_list = []
        self.end_points = []
        self.flag = False
        self.center = []
        self.dim = 0
        self.image_data = None
        self.numpy_label = np.zeros(self.dim, dtype=np.float32)

    ###打开文件夹按钮功能函数###
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
        if len(dirname) > 0:
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
            self.sliber.setMaximum(self.imageviewer.GetSliceMax())
            self.sliber.setMinimum(self.imageviewer.GetSliceMin())
            self.sliber.setValue(self.imageviewer.GetSlice())
            self.dim = self.dicomreader.GetOutput().GetDimensions()
            self.image_data = numpy_support.vtk_to_numpy(self.dicomreader.GetOutput().GetPointData().GetScalars())
            self.image_data = self.image_data.reshape(self.dim[2], self.dim[1], self.dim[0])
            self.image_data = np.flip(np.flip(self.image_data, axis=1), axis=0)
            self.numpy_label = np.zeros(self.dim, dtype=np.float32)
            self.sliber.setVisible(True)
            self.non_occluded_button.setEnabled(True)
            self.occluded_button.setEnabled(True)
            self.show_result_button.setVisible(False)
            self.save_patch_button.setVisible(False)

    ###滚轮向上滚动功能函数###
    def last_slice(self, obj, ev):
        slice_number = self.imageviewer.GetSlice()
        self.imageviewer.SetSlice((slice_number - 1) % self.imageviewer.GetSliceMax())
        self.slice_num_label.setText(str(self.imageviewer.GetSlice()))
        self.sliber.setValue(slice_number - 1)

    ###滚轮向下滚动功能函数###
    def next_slice(self, obj, ev):
        slice_number = self.imageviewer.GetSlice()
        self.imageviewer.SetSlice((slice_number + 1) % self.imageviewer.GetSliceMax())
        self.slice_num_label.setText(str(self.imageviewer.GetSlice()))
        self.sliber.setValue(slice_number + 1)

    ###左键选点功能函数###
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

    ###非闭塞段端点按钮功能函数###
    @Slot()
    def non_occluded_button_clicked(self):
        self.flag = False
        self.capture_mouse(self.image_data)
        if len(self.seedpt3d_list) >= 2:
            self.get_vessel_button.setEnabled(True)
            self.show_result_button.setVisible(False)
            self.save_patch_button.setVisible(False)

    ###滑块功能函数###
    @Slot()
    def sliber_value_changed(self):
        slice_number = self.sliber.value()
        self.imageviewer.SetSlice(slice_number)
        self.slice_num_label.setText(str(slice_number))

    ###opencv实现选点###
    def on_trace_bar_changed(self, args):
        pass

    def capture_mouse(self, image_array, win_max=1200, win_min=-100, seedname='left femur'):

        self.seedpt_list = []
        self.seedpt3d_list = []
        self.seed_slice = 0

        image_array = sitk.GetImageFromArray(image_array)
        intensityfilter = sitk.IntensityWindowingImageFilter()
        intensityfilter.SetWindowMaximum(np.double(win_max))
        intensityfilter.SetWindowMinimum(np.double(win_min))
        intensityfilter.SetOutputMaximum(255)
        intensityfilter.SetOutputMinimum(0)
        image_adjust = intensityfilter.Execute(image_array)

        image_data = sitk.GetArrayFromImage(image_adjust)
        image_data = np.uint8(image_data.transpose(1, 2, 0))

        h, w, s = image_data.shape
        window_name = 'Please choose the seed point of {}'.format(seedname)
        cv.namedWindow(window_name, cv.WINDOW_GUI_NORMAL)
        cv.createTrackbar('Slice', window_name, 0, s - 1, self.on_trace_bar_changed)
        cv.setMouseCallback(window_name, self.mouse_clip)
        cv.imshow(window_name, image_data[:, :, 0])

        while True:
            slice_num = cv.getTrackbarPos('Slice', window_name)
            self.seed_slice = slice_num
            img_slice = image_data[:, :, slice_num]
            if self.seedpt_list.__len__() is not 0:
                img_rgb = cv.cvtColor(img_slice, cv.COLOR_GRAY2RGB)
                for seeds in self.seedpt3d_list:
                    if slice_num == seeds[2]:
                        cv.circle(img_rgb, tuple(seeds[:2]), 2, (0, 0, 255), -1)
                cv.imshow(window_name, img_rgb)
            else:
                cv.imshow(window_name, img_slice)
                cv.waitKey(10)

            if 0xFF & cv.waitKey(10) == ord('q'):
                cv.destroyAllWindows()
                if len(self.seedpt3d_list) > 0:
                    x, y = self.seedpt
                    self.seedpt = None
                    print('{}: ({}, {}, {})'.format(seedname, x, y, slice_num))
                if not self.flag:
                    self.end_points = self.seedpt3d_list
                break

            elif 0xFF & cv.waitKey(10) == ord('r'):
                img_rgb = cv.cvtColor(img_slice, cv.COLOR_GRAY2RGB)
                self.seedpt = None
                cv.imshow(window_name, img_rgb)

    def mouse_clip(self, event, x, y, flags, param):

        if event == cv.EVENT_LBUTTONDOWN:
            self.seedpt = (x, y)
            self.seedpt_list.append(list(self.seedpt))
            self.seedpt3d_list.append([x, y, self.seed_slice])
            if self.flag:
                center_z = [temp[2] for temp in self.center]
                end_points_z = [temp[2] for temp in self.end_points]
                if self.seed_slice not in center_z and self.seed_slice not in end_points_z:
                    self.center.append([x, y, self.seed_slice])
                    print(self.center)
            else:
                print(self.seedpt3d_list)

        elif event == cv.EVENT_RBUTTONDOWN:
            if self.flag and self.center.__len__() != 0:
                    self.center.pop(-1)
                    # print(center)
            if self.seedpt_list.__len__() != 0:
                self.seedpt_list.pop(-1)
                self.seedpt3d_list.pop(-1)
                print(self.seedpt3d_list)

    ###闭塞段控制点按钮功能函数###
    def occluded_button_clicked(self):
        self.flag = True
        self.capture_mouse(self.image_data)
        if len(self.center) > 2:
            self.get_vessel_button.setEnabled(True)
            self.show_result_button.setVisible(False)
            self.save_patch_button.setVisible(False)

    ###提取血管按钮功能函数###
    def get_vessel_button_clicked(self):
        print(self.end_points)
        self.numpy_label = get_vessel(self)
        self.show_result_button.setVisible(True)
        self.save_patch_button.setVisible(True)


    ###结果展示按钮功能函数###
    def show_result_button_clicked(self):
        display3d(self.image_data, self.numpy_label.astype(np.uint8))
        slice_seg_contours(self.image_data, self.numpy_label)

    ###patch保存按钮功能函数###
    def save_patch_button_clicked(self):


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()

    sys.exit(app.exec_())
