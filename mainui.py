import sys
import os
import PySide2
dirname = os.path.dirname(PySide2.__file__)
plugin_path = os.path.join(dirname, 'plugins')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
import vtk
from PySide2.QtWidgets import QMainWindow, QPushButton, QFrame, QVBoxLayout, QHBoxLayout, QFileDialog, QApplication, QMessageBox, QLineEdit, QLabel, QSlider, QTextBrowser, QWidget, QRadioButton
from PySide2.QtCore import Slot, QDir, Qt
from PySide2.QtGui import QTextCursor
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk.util import numpy_support
from vesselfit import get_vessel
import SimpleITK as sitk
import cv2 as cv
import numpy as np
from copy import deepcopy
from classification import ImageClassifier



class MainWindow(QMainWindow):

    def __init__(self, parent=None):
        ###主窗体初始化###
        QMainWindow.__init__(self, parent)
        self.setWindowTitle("股浅动脉钙化分型软件")
        self.setMinimumSize(1280, 720)
        ###打开DICOM文件夹按钮###
        self.open_directory_button = QPushButton("打开文件夹")
        self.open_directory_button.clicked.connect(self.open_directory_button_clicked)
        self.button_font = self.open_directory_button.font()
        self.button_font.setPointSize(14)
        self.open_directory_button.setFont(self.button_font)
        ###路径框###
        self.path_edit = QLineEdit(self)
        ###滑块###
        self.sliber = QSlider()
        self.sliber.setVisible(False)
        self.sliber.setOrientation(Qt.Horizontal)
        self.sliber.valueChanged.connect(self.sliber_value_changed)
        ###slice计数标签###
        self.slice_num_label = QLabel(self)
        self.slice_num_label.setAlignment(Qt.AlignCenter)
        label_font = self.slice_num_label.font()
        label_font.setPointSize(20)
        self.slice_num_label.setFont(label_font)
        ###dicom-VTK显示控件###
        self.frame = QWidget(self)
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vtkWidget.setMinimumSize(512, 512)
        self.imageviewer = vtk.vtkImageViewer2()
        self.ren = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        ###分型结果显示标签###
        self.type_name_label = QLabel("分型结果")
        self.type_name_label.setFont(label_font)
        self.type_name_label.setFixedHeight(30)
        self.type_label = QLabel("无")
        self.type_label.setFont(label_font)
        self.type_label.setFixedHeight(30)
        ###patch-VTK显示控件###
        self.frame_1 = QFrame()
        self.patch_viewer = QVTKRenderWindowInteractor(self.frame)
        self.patch_viewer.setFixedSize(256, 256)
        self.imageviewer_1 = vtk.vtkImageViewer2()
        self.ren_1 = vtk.vtkRenderer()
        self.patch_viewer.GetRenderWindow().AddRenderer(self.ren_1)
        self.iren_1 = self.patch_viewer.GetRenderWindow().GetInteractor()
        ###选腿控件###
        self.left_button = QRadioButton('l')
        self.right_button = QRadioButton('r')
        self.left_button.setFont(self.button_font)
        self.right_button.setFont(self.button_font)
        self.right_button.setChecked(True)
        self.left_button.clicked.connect(self.select_leg_button_clicked)
        self.right_button.clicked.connect(self.select_leg_button_clicked)
        ###选取非闭塞段端点按钮###
        self.non_occluded_button = QPushButton("选取非闭塞段端点")
        self.non_occluded_button.clicked.connect(self.non_occluded_button_clicked)
        self.non_occluded_button.setEnabled(False)
        self.non_occluded_button.setFont(self.button_font)
        ###选取闭塞段控制点按钮###
        self.occluded_button = QPushButton("选取闭塞段控制点")
        self.occluded_button.clicked.connect(self.occluded_button_clicked)
        self.occluded_button.setEnabled(False)
        self.occluded_button.setFont(self.button_font)
        ###提取血管按钮###
        self.get_vessel_button = QPushButton("提取血管")
        self.get_vessel_button.clicked.connect(self.get_vessel_button_clicked)
        self.get_vessel_button.setEnabled(False)
        self.get_vessel_button.setFont(self.button_font)
        ###结果展示按钮###
        self.show_2D_result_button = QPushButton("显示提取结果（二维）")
        self.show_2D_result_button.clicked.connect(self.show_2D_result_button_clicked)
        self.show_2D_result_button.setVisible(False)
        self.show_2D_result_button.setFont(self.button_font)
        self.show_3D_result_button = QPushButton("显示提取结果（三维）")
        self.show_3D_result_button.clicked.connect(self.show_3D_result_button_clicked)
        self.show_3D_result_button.setVisible(False)
        self.show_3D_result_button.setFont(self.button_font)
        ###patch保存按钮###
        self.save_patch_button = QPushButton("保存patch")
        self.save_patch_button.clicked.connect(self.save_patch_button_clicked)
        self.save_patch_button.setVisible(False)
        self.save_patch_button.setFont(self.button_font)
        ###导入模型按钮###
        self.import_module_button = QPushButton("导入模型")
        self.import_module_button.clicked.connect(self.import_module_button_clicked)
        self.import_module_button.setVisible(False)
        self.import_module_button.setFont(self.button_font)
        ###分型按钮###
        self.decide_type_button = QPushButton("进行分型")
        self.decide_type_button.clicked.connect(self.decide_type_button_clicked)
        self.decide_type_button.setVisible(False)
        self.decide_type_button.setFont(self.button_font)
        ###信息框###
        self.info_browser = QTextBrowser()
        self.info_browser.setFont(self.button_font)
        self.info_browser.setMaximumHeight(100)
        self.info_browser.textChanged.connect(self.info_browser_text_changed)
        ###路径布局###
        self.path_layout = QHBoxLayout()
        self.path_layout.addWidget(self.path_edit)
        self.path_layout.addWidget(self.open_directory_button)
        ###分型结果标签布局###
        self.type_label_layout = QHBoxLayout()
        self.type_label_layout.addWidget(self.type_name_label)
        self.type_label_layout.addWidget(self.type_label)
        ###选腿按钮布局###
        self.select_leg_layout = QHBoxLayout()
        self.select_leg_layout.addWidget(self.left_button)
        self.select_leg_layout.addWidget(self.right_button)
        ###patch分型显示布局###
        self.patch_type_layout = QVBoxLayout()
        self.patch_type_layout.addLayout(self.type_label_layout)
        self.patch_type_layout.addWidget(self.patch_viewer)
        self.patch_type_layout.addLayout(self.select_leg_layout)
        ###显示控件布局###
        self.viewer_layout = QHBoxLayout()
        self.viewer_layout.setAlignment(Qt.AlignCenter)
        self.viewer_layout.addWidget(self.vtkWidget)
        self.viewer_layout.addLayout(self.patch_type_layout)
        ###功能按钮布局###
        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(self.non_occluded_button)
        self.button_layout.addWidget(self.occluded_button)
        self.button_layout.addWidget(self.get_vessel_button)
        self.button_layout.addWidget(self.show_2D_result_button)
        self.button_layout.addWidget(self.show_3D_result_button)
        self.button_layout.addWidget(self.save_patch_button)
        self.button_layout.addWidget(self.import_module_button)
        self.button_layout.addWidget(self.decide_type_button)
        ###主布局###
        self.main_layout = QVBoxLayout()
        self.main_layout.addLayout(self.path_layout)
        self.main_layout.addWidget(self.slice_num_label)
        self.main_layout.addWidget(self.sliber)
        # self.main_layout.addWidget(self.vtkWidget)
        self.main_layout.addLayout(self.viewer_layout)
        self.main_layout.addLayout(self.button_layout)
        self.main_layout.addWidget(self.info_browser)
        # self.LastPickedActor = None
        # self.LastPickedProperty = vtk.vtkProperty()

        self.dicomreader = vtk.vtkDICOMImageReader()
        # self.ren.ResetCamera()
        self.frame.setLayout(self.main_layout)
        self.setCentralWidget(self.frame)
        # self.frame_1.setLayout(self.main_layout)
        # self.setCentralWidget(self.frame_1)

        self.show()
        self.iren.Initialize()
        self.iren_1.Initialize()
        self.dir_name = None
        self.seedname = 'right leg'
        self.seedpt = None
        self.seedpt_list = []
        self.seed_slice = 0
        self.seedpt3d_list = []
        self.end_points = []
        self.flag = False
        self.center = []
        self.vessel_center = []
        self.dim = 0
        self.image_data = None
        self.numpy_label = np.zeros(self.dim, dtype=np.float32)
        self.patch_data = []
        self.patch_image = vtk.vtkImageData()
        self.patch_flag = False
        self.module_flag = False
        self.label_all = []
        self.conf_all = []
        self.type_flag = False

    ###打开文件夹按钮功能函数###
    @Slot()
    def open_directory_button_clicked(self):
        dirname = QFileDialog.getExistingDirectory(self, "DICOM文件夹", '\home',
                                                             QFileDialog.ShowDirsOnly)
        while len(dirname) > 0:
            tempdir = QDir(dirname)
            self.dir_name = tempdir.dirName()
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
            self.info_browser.insertPlainText("数据读取中......\n")
            self.repaint()
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
            self.ren_1.ResetCamera()
            self.iren.Initialize()
            self.iren.Start()
            self.imageviewer.Render()
            self.imageviewer.SetSlice(0)
            self.imageviewer.SetSize(512, 512)
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
            self.show_2D_result_button.setVisible(False)
            self.show_3D_result_button.setVisible(False)
            self.save_patch_button.setVisible(False)
            self.imageviewer.GetRenderWindow().SetSize(512, 512)
            self.type_flag = False
            self.patch_flag = False
            self.get_vessel_button.setEnabled(False)
            self.import_module_button.setVisible(False)
            self.decide_type_button.setVisible(False)
            self.type_label.setText("无")
            self.info_browser.insertPlainText("数据读取完成！\n")

    ###滚轮向上滚动功能函数###
    def last_slice(self, obj, ev):
        slice_number = self.imageviewer.GetSlice()
        self.imageviewer.SetSlice((slice_number - 1) % self.imageviewer.GetSliceMax())
        if self.patch_flag:
            self.imageviewer_1.SetSlice((slice_number - 1) % self.imageviewer_1.GetSliceMax())
        if self.type_flag:
            self.type_label.setText(self.label_all[self.imageviewer.GetSlice()])
        self.slice_num_label.setText(str(self.imageviewer.GetSlice()))
        self.sliber.setValue(slice_number - 1)

    ###滚轮向下滚动功能函数###
    def next_slice(self, obj, ev):
        slice_number = self.imageviewer.GetSlice()
        self.imageviewer.SetSlice((slice_number + 1) % self.imageviewer.GetSliceMax())
        if self.patch_flag:
            self.imageviewer_1.SetSlice((slice_number + 1) % self.imageviewer_1.GetSliceMax())
        if self.type_flag:
            self.type_label.setText(self.label_all[self.imageviewer.GetSlice()])
        self.slice_num_label.setText(str(self.imageviewer.GetSlice()))
        self.sliber.setValue(slice_number + 1)

    ###左键选点功能函数###
    def left_pressed(self, obj, ev):
        clickpos = self.iren.GetEventPosition()
        picker = vtk.vtkPointPicker()
        picker.Pick(clickpos[0], clickpos[1], 0, self.ren)
        # self.NewPickedActor = picker.GetActor()
        # if self.NewPickedActor:
        #     if self.LastPickedActor:
        #         self.LastPickedActor.GetProperty.DeepCopy(self.LastPickedProperty)
        #     self.LastPickedProperty.DeepCopy(self.NewPickedActor.GetProperty())
        #     self.NewPickedActor.GetProperty().SetColor(vtk.vtkNamedColors().GetColor('Red'))
        #     self.LastPickedActor = self.NewPickedActor

    ###选腿按钮功能函数###
    @Slot()
    def select_leg_button_clicked(self):
        self.repaint()
        button = self.sender()
        if button.text() == 'l':
            self.seedname = 'left leg'
        if button.text() == 'r':
            self.seedname = 'right leg'


    ###非闭塞段端点按钮功能函数###
    @Slot()
    def non_occluded_button_clicked(self):
        self.flag = False
        self.capture_mouse(self.image_data)
        if len(self.seedpt3d_list) >= 2:
            self.get_vessel_button.setEnabled(True)
            self.show_2D_result_button.setVisible(False)
            self.show_3D_result_button.setVisible(False)
            self.save_patch_button.setVisible(False)
            self.patch_flag = False

    ###滑块功能函数###
    @Slot()
    def sliber_value_changed(self):
        slice_number = self.sliber.value()
        self.imageviewer.SetSlice(slice_number)
        self.slice_num_label.setText(str(slice_number))
        if self.patch_flag:
            self.imageviewer_1.SetSlice(slice_number)
        if self.type_flag:
            self.type_label.setText(self.label_all[slice_number])

    ###opencv实现选点###
    def on_trace_bar_changed(self, args):
        pass

    def capture_mouse(self, image_array, win_max=1200, win_min=-100):

        self.seedpt_list = []
        self.seedpt3d_list = []
        self.seed_slice = 0
        seedname = self.seedname

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
                    print(self.center)
            if self.seedpt_list.__len__() != 0:
                self.seedpt_list.pop(-1)
                self.seedpt3d_list.pop(-1)
                print(self.seedpt3d_list)

    ###闭塞段控制点按钮功能函数###
    @Slot()
    def occluded_button_clicked(self):
        self.flag = True
        self.capture_mouse(self.image_data)
        if len(self.center) > 2:
            self.get_vessel_button.setEnabled(True)
            self.show_2D_result_button.setVisible(False)
            self.show_3D_result_button.setVisible(False)
            self.save_patch_button.setVisible(False)
            self.patch_flag = False

    ###提取血管按钮功能函数###
    def get_vessel_button_clicked(self):
        self.vessel_center = []
        self.info_browser.insertPlainText("血管提取中......\n")
        self.repaint()
        self.numpy_label = get_vessel(self)
        self.show_2D_result_button.setVisible(True)
        self.show_3D_result_button.setVisible(True)
        image = sitk.GetImageFromArray(self.image_data)
        filter_intensity = sitk.IntensityWindowingImageFilter()
        filter_intensity.SetOutputMaximum(255)
        filter_intensity.SetOutputMinimum(0)
        filter_intensity.SetWindowMinimum(-200)
        filter_intensity.SetWindowMaximum(900)
        image = filter_intensity.Execute(image)
        image_data = sitk.GetArrayFromImage(image)
        image_data = image_data.transpose(2, 1, 0)
        box_height = 32
        box_width = 32
        self.patch_data = np.zeros([box_height * 2, box_width * 2, self.dim[2]], dtype=np.float32)
        for center in self.vessel_center:
            slice_num = center[2]
            img_slice = image_data[:, :, slice_num]
            patch_slice = img_slice[center[0] - box_height + 1: center[0] + box_height + 1,
                          center[1] - box_width + 1: center[1] + box_width + 1]
            self.patch_data[:, :, slice_num] = patch_slice

        vtk_data_array = numpy_support.numpy_to_vtk(
            np.flip(np.flip(self.patch_data, axis=1), axis=2).transpose(2, 1, 0).ravel(), deep=True,
            array_type=vtk.VTK_FLOAT)
        self.patch_image = vtk.vtkImageData()
        self.patch_image.SetDimensions(box_height * 2, box_width * 2, self.dim[2])
        self.patch_image.GetPointData().SetScalars(vtk_data_array)
        self.imageviewer_1.SetInputData(self.dicomreader.GetOutput())
        self.imageviewer_1.SetupInteractor(self.iren_1)
        self.imageviewer_1.SetRenderWindow(self.patch_viewer.GetRenderWindow())
        self.ren_1.ResetCameraClippingRange()
        self.iren_1.Initialize()
        self.iren_1.Start()
        self.imageviewer_1.Render()
        self.imageviewer_1.SetInputData(self.patch_image)
        self.imageviewer_1.SetupInteractor(self.iren_1)
        self.imageviewer_1.SetRenderWindow(self.patch_viewer.GetRenderWindow())
        self.imageviewer_1.SetSize(256, 256)
        self.ren_1.ResetCamera()
        self.iren_1.Initialize()
        self.iren_1.Start()
        self.imageviewer_1.Render()
        self.patch_flag = True
        self.save_patch_button.setVisible(True)
        self.info_browser.insertPlainText("血管提取完成！可拖动滑块在小窗口中查看patch\n")
        self.import_module_button.setVisible(True)
        self.decide_type_button.setVisible(True)
        if not self.module_flag:
            self.decide_type_button.setEnabled(False)

    ###结果展示按钮功能函数###
    @Slot()
    def show_3D_result_button_clicked(self):
        self.display3d()

    def display3d(self):
        label = deepcopy(self.numpy_label.astype(np.uint8))
        img = sitk.GetImageFromArray(self.image_data)
        d, w, h = label.shape
        dicom_images = vtk.vtkImageImport()
        dicom_images.CopyImportVoidPointer(label.tostring(), len(label.tostring()))
        dicom_images.SetDataScalarTypeToUnsignedChar()
        dicom_images.SetNumberOfScalarComponents(1)

        px, py, s = img.GetSpacing()
        dicom_images.SetDataSpacing(s, py, px)
        dicom_images.SetDataExtent(0, h - 1, 0, w - 1, 0, d - 1)
        dicom_images.SetWholeExtent(0, h - 1, 0, w - 1, 0, d - 1)
        dicom_images.Update()

        render = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window.SetSize(1280, 800)
        render_window.AddRenderer(render)
        render_interact = vtk.vtkRenderWindowInteractor()
        render_interact.SetRenderWindow(render_window)

        threshold_dicom_image = vtk.vtkImageThreshold()
        threshold_dicom_image.SetInputConnection(dicom_images.GetOutputPort())
        threshold_dicom_image.Update()

        discrete_marching_cubes = vtk.vtkDiscreteMarchingCubes()
        discrete_marching_cubes.SetInputConnection(dicom_images.GetOutputPort())
        discrete_marching_cubes.GenerateValues(3, 1, 3)
        discrete_marching_cubes.ComputeNormalsOn()
        discrete_marching_cubes.Update()

        colorLookupTable = vtk.vtkLookupTable()
        colorLookupTable.SetNumberOfTableValues(3)
        colorLookupTable.Build()
        colorLookupTable.SetTableValue(0, 204 / 255.0, 84 / 255.0, 58 / 255.0, 1)
        # colorLookupTable.SetTableValue(1, 180/255.0, 160/255.0, 100/255.0, 1)
        colorLookupTable.SetTableValue(1, 218 / 255.0, 201 / 255.0, 166 / 255.0, 1)
        colorLookupTable.SetTableValue(2, 112 / 255.0, 160 / 255.0, 180 / 255.0, 1)

        dicom_data_mapper = vtk.vtkPolyDataMapper()
        dicom_data_mapper.SetInputConnection(discrete_marching_cubes.GetOutputPort())
        dicom_data_mapper.ScalarVisibilityOn()
        dicom_data_mapper.SetLookupTable(colorLookupTable)
        dicom_data_mapper.SetScalarRange(1, 3)
        dicom_data_mapper.Update()

        actor_dicom_3d = vtk.vtkActor()
        actor_dicom_3d.SetMapper(dicom_data_mapper)

        render.AddActor(actor_dicom_3d)
        render.ResetCamera()

        render_window.Render()
        render_interact.Start()

    def show_2D_result_button_clicked(self):
        self.slice_seg_contours()

    def slice_seg_contours(self, planes='t'):
        img = sitk.GetImageFromArray(self.image_data)
        label = sitk.GetImageFromArray(np.uint8(self.numpy_label.transpose(2, 1, 0)))
        label.CopyInformation(img)
        h, w, s = img.GetSize()
        filter_intensity = sitk.IntensityWindowingImageFilter()
        cv.namedWindow('Segmentation(exit with \'Q\')', cv.WINDOW_GUI_NORMAL)
        cv.createTrackbar('Slice', 'Segmentation(exit with \'Q\')', 0, s - 1, self.on_trace_bar_changed)
        filter_intensity.SetOutputMaximum(255)
        filter_intensity.SetOutputMinimum(0)
        filter_intensity.SetWindowMinimum(-200)
        filter_intensity.SetWindowMaximum(900)
        img_show = filter_intensity.Execute(img)
        while True:
            slice_num = cv.getTrackbarPos('Slice', 'Segmentation(exit with \'Q\')')
            if planes == 't':
                itk_display = sitk.LabelOverlay(img_show[:, :, slice_num], sitk.LabelContour(label[:, :, slice_num]),
                                                1.0)
            elif planes == 's':
                itk_display = sitk.LabelOverlay(img_show[:, slice_num, :], sitk.LabelContour(label[:, slice_num, :]),
                                                1.0)
            elif planes == 'c':
                itk_display = sitk.LabelOverlay(img_show[slice_num, :, :], sitk.LabelContour(label[slice_num, :, :]),
                                                1.0)
            else:
                break

            img_display = sitk.GetArrayFromImage(itk_display)
            cv.imshow('Segmentation(exit with \'Q\')', np.uint8(img_display))
            if 0xFF & cv.waitKey(10) == ord('q'):
                cv.destroyAllWindows()
                break

    ###patch保存按钮功能函数###
    @Slot()
    def save_patch_button_clicked(self):
        save_path = QFileDialog.getExistingDirectory(self, "保存路径", "\home", QFileDialog.ShowDirsOnly)
        if len(save_path) > 0:
            if self.seedname == 'left leg':
                leg = 'l'
            if self.seedname == 'right leg':
                leg = 'r'
            patch_dir = os.path.join(save_path, self.dir_name)
            if not os.path.exists(patch_dir):
                os.makedirs(patch_dir)
            for center in self.vessel_center:
                slice_num = center[2]
                patch_img_name = "{}_{}.jpeg".format(leg, slice_num)
                patch_path = os.path.join(patch_dir, patch_img_name)
                cv.imwrite(patch_path, np.uint8(self.patch_data[:, :, slice_num]))

    ###导入模型按钮功能函数###
    @Slot()
    def import_module_button_clicked(self):
        self.info_browser.insertPlainText("模型导入中......\n")
        self.repaint()
        model_root = 'merge12A_0628_001439'
        self.classifier = ImageClassifier(model_root)
        self.module_flag = True
        self.decide_type_button.setEnabled(True)
        self.info_browser.insertPlainText("模型导入完成！\n")

    ###分型按钮功能函数###
    @Slot()
    def decide_type_button_clicked(self):
        self.info_browser.insertPlainText("开始分型......\n")
        self.repaint()
        self.label_all, self.conf_all = self.classifier.infer(np.flip(np.flip(self.patch_data, axis=1), axis=2).transpose(2, 1, 0))
        self.type_flag = True
        self.info_browser.insertPlainText("分型完成！\n")

    ###信息框自动滚动###
    @Slot()
    def info_browser_text_changed(self):
        self.info_browser.moveCursor(QTextCursor().End)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("windowsxp")
    window = MainWindow()

    sys.exit(app.exec_())
    exec()
