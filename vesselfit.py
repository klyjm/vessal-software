import os

import SimpleITK as sitk
import cv2 as cv
import numpy as np
from vtk.util import numpy_support
from interp import interp1

seedpt = None
seedpt_list = []
seedpt3d_list = []
center = []
flag = False
seed_slice = 0


def slice_seg_contours(img, label, planes='t', Manualwin=True):
    h, w, s = img.GetSize()
    min_max_filter = sitk.MinimumMaximumImageFilter()
    min_max_filter.Execute(img)
    min_value = int(min_max_filter.GetMinimum())
    max_value = int(min_max_filter.GetMaximum())

    filter_intensity = sitk.IntensityWindowingImageFilter()

    cv.namedWindow('Segmentation', cv.WINDOW_GUI_NORMAL)
    cv.createTrackbar('Slice', 'Segmentation', 0, s - 1, on_trace_bar_changed)
    cv.createTrackbar('Win_min', 'Segmentation', 0, max_value - min_value, on_trace_bar_changed)
    cv.createTrackbar('Win_max', 'Segmentation', max_value - min_value, max_value - min_value, on_trace_bar_changed)
    win_max_old = cv.getTrackbarPos('Win_max', 'Segmentation')
    win_min_old = cv.getTrackbarPos('Win_min', 'Segmentation')
    img_show = img
    while True:
        slice_num = cv.getTrackbarPos('Slice', 'Segmentation')
        if Manualwin:
            win_max = cv.getTrackbarPos('Win_max', 'Segmentation')
            win_min = cv.getTrackbarPos('Win_min', 'Segmentation')
            if win_max != win_max_old or win_min != win_min_old:
                win_max_old = win_max
                win_min_old = win_min
                filter_intensity.SetOutputMaximum(255)
                filter_intensity.SetOutputMinimum(0)
                filter_intensity.SetWindowMinimum(win_min + min_value)
                filter_intensity.SetWindowMaximum(win_max + min_value)
                img_show = filter_intensity.Execute(img)
        else:
            filter_intensity.SetOutputMaximum(255)
            filter_intensity.SetOutputMinimum(0)
            filter_intensity.SetWindowMinimum(-100)
            filter_intensity.SetWindowMaximum(900)
            img_show = filter_intensity.Execute(img)
        if planes == 't':
            itk_display = sitk.LabelOverlay(img_show[:, :, slice_num], sitk.LabelContour(label[:, :, slice_num]), 1.0)
        elif planes == 's':
            itk_display = sitk.LabelOverlay(img_show[:, slice_num, :], sitk.LabelContour(label[:, slice_num, :]), 1.0)
        elif planes == 'c':
            itk_display = sitk.LabelOverlay(img_show[slice_num, :, :], sitk.LabelContour(label[slice_num, :, :]), 1.0)
        else:
            print('THe plane is wrong')
            break

        img_display = sitk.GetArrayFromImage(itk_display)
        cv.imshow('Segmentation', np.uint8(img_display))
        if 0xFF & cv.waitKey(10) == ord('q'):
            cv.destroyAllWindows()
            break


def slice_show(img, win_max=400, win_min=-200):
    if img.GetDepth() == 0:
        h, w = img.GetSize()
        s = 1
    else:
        h, w, s = img.GetSize()
    filter_intensity = sitk.IntensityWindowingImageFilter()
    filter_intensity.SetWindowMaximum(win_max)
    filter_intensity.SetWindowMinimum(win_min)
    filter_intensity.SetOutputMaximum(255)
    filter_intensity.SetOutputMinimum(0)
    img_ad = filter_intensity.Execute(img)
    cv.namedWindow('itk show', cv.WINDOW_GUI_NORMAL)
    cv.createTrackbar('Slice', 'itk show', 0, s - 1, on_trace_bar_changed)
    while True:
        slice_num = cv.getTrackbarPos('Slice', 'itk show')
        img_display = sitk.GetArrayFromImage(img_ad[:, :, slice_num])
        cv.imshow('itk show', np.uint8(img_display))
        if 0xFF & cv.waitKey(10) == ord('q'):
            cv.destroyAllWindows()
            break


def mouse_clip(event, x, y, flags, param):
    global seedpt
    global seedpt_list
    global seed_slice
    global seedpt3d_list
    global flag
    global center

    if event == cv.EVENT_LBUTTONDOWN:
        seedpt = (x, y)
        seedpt_list.append(list(seedpt))
        seedpt3d_list.append([x, y, seed_slice])
        if flag:
            centerz = [temp[2] for temp in center]
            if seed_slice not in centerz:
                center.append([x, y, seed_slice])
                print(center)
        else:
            print(seedpt3d_list)

    elif event == cv.EVENT_RBUTTONDOWN:
        if flag and center.__len__() != 0:
                center.pop(-1)
                print(center)
        if seedpt_list.__len__() != 0:
            seedpt_list.pop(-1)
            seedpt3d_list.pop(-1)
            print(seedpt3d_list)

def on_trace_bar_changed(args):
    pass

def capture_mouse(image_array, win_max=1200, win_min=-100, seedname='left femur'):
    global seedpt
    global seedpt_list
    global seed_slice
    global seedpt3d_list

    seedpt_list = []
    seedpt3d_list = []
    seed_slice = 0

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
    cv.createTrackbar('Slice', window_name, 0, s - 1, on_trace_bar_changed)
    cv.setMouseCallback(window_name, mouse_clip)
    cv.imshow(window_name, image_data[:, :, 0])

    while True:
        slice_num = cv.getTrackbarPos('Slice', window_name)
        seed_slice = slice_num
        img_slice = img_nda[:, :, slice_num]
        if seedpt_list.__len__() is not 0:
            img_rgb = cv.cvtColor(img_slice, cv.COLOR_GRAY2RGB)
            for seeds in seedpt3d_list:
                if slice_num == seeds[2]:
                    cv.circle(img_rgb, tuple(seeds[:2]), 2, (0, 0, 255), -1)
            cv.imshow(window_name, img_rgb)
        else:
            cv.imshow(window_name, img_slice)
            cv.waitKey(10)

        if 0xFF & cv.waitKey(10) == ord('q'):
            cv.destroyAllWindows()
            if len(seedpt3d_list) > 0:
                x, y = seedpt
                seedpt = None
                print('{}: ({}, {}, {})'.format(seedname, x, y, slice_num))
            global flag
            if not flag:
                return seedpt3d_list
            break

        elif 0xFF & cv.waitKey(10) == ord('r'):
            img_rgb = cv.cvtColor(img_slice, cv.COLOR_GRAY2RGB)
            seedpt = None
            cv.imshow(window_name, img_rgb)



def display3d(img, label):
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

    # volumn_dicom = vtk.vtkVolume()
    # volumn_dicom.SetMapper(dicom_data_mapper)
    # volumn_dicom.SetProperty(dicom_data_property)

    actor_dicom_3d = vtk.vtkActor()
    actor_dicom_3d.SetMapper(dicom_data_mapper)

    render.AddActor(actor_dicom_3d)
    # render.AddVolume(volumn_dicom)
    render.ResetCamera()

    render_window.Render()
    render_interact.Start()


def sort2(s):
    return s[2]


if __name__ == "__main__":
    patientid = 1
    side = 'r'
    filelist = os.listdir(data_root)
    file_path = os.path.join(data_root, filelist[patientid])
    dcm_sitk = read_dcm_series(file_path)

    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(file_path)
    reader.Update()
    image = reader.GetOutput()
    a = image.GetDimensions()
    numpy_label = np.zeros(a, dtype=np.float32)

    # Level Sets Initialization


    while True:
        points = capture_mouse(dcm_sitk, win_min=-100, win_max=1200)
        if len(points) % 2 == 0:
            break
        else:
            print('Only select two points for the beginning and the end!')

    dim = reader.GetOutput().GetDimensions()
    sourcepoints = []
    targetpoints = []
    for i in range(len(points)):
        if i % 2 == 0:
            sourcepoints.append(points[i])
        else:
            targetpoints.append(points[i])
    for i in range(len(sourcepoints)):
        vmtkImageInitialization = vmtkscripts.vmtkImageInitialization()
        vmtkImageInitialization.Interactive = 0
        vmtkImageInitialization.Image = reader.GetOutput()
        vmtkImageInitialization.UpperThreshold = 1000
        vmtkImageInitialization.LowerThreshold = 120
        vmtkImageInitialization.NegateImage = 0
        vmtkImageInitialization.SourcePoints = [sourcepoints[i][0], dim[1] - sourcepoints[i][1] - 1, dim[2] - sourcepoints[i][2] - 1]
        vmtkImageInitialization.TargetPoints = [targetpoints[i][0], dim[1] - targetpoints[i][1] - 1, dim[2] - targetpoints[i][2] - 1]
        vmtkImageInitialization.Execute()


            # Feature Image
        imageFeatures = vmtkscripts.vmtkImageFeatures()
        imageFeatures.Image = reader.GetOutput()
        imageFeatures.FeatureImageType = 'upwind'
        imageFeatures.SigmoidRemapping = 0
        imageFeatures.DerivativeSigma = 0.0
        imageFeatures.UpwindFactor = 1.0
        imageFeatures.FWHMRadius = [1.0, 1.0, 1.0]
        imageFeatures.FWHMBackgroundValue = 0.0
        imageFeatures.Execute()

        # # Level Sets
        levelset = vmtkscripts.vmtkLevelSetSegmentation()
        levelset.Image = reader.GetOutput()
        levelset.LevelSetsInput = vmtkImageInitialization.InitialLevelSets
        levelset.FeatureImage = imageFeatures.FeatureImage
        levelset.IsoSurfaceValue = 0.0
        levelset.LevelSetEvolution()

    # Display
    # imageview = vmtkscripts.vmtkImageViewer()
    # imageview.Image = reader.GetOutput()
    # imageview.Display = 1
    # imageview.Execute()
    #
    # vmtkRenderer = vmtkscripts.vmtkRenderer()
    # vmtkRenderer.Initialize()
    # vmtkRenderer.RegisterScript(vmtkImageInitialization)
    #
    # SurfaceViewer = vmtkscripts.vmtkSurfaceViewer()
    # SurfaceViewer.vmtkRenderer = vmtkRenderer
    #
    # marchingCubes = vtk.vtkMarchingCubes()
    # marchingCubes.SetInputData(levelset.LevelSetsOutput)
    # marchingCubes.SetValue(0, 0)
    # marchingCubes.Update()
    #
    # SurfaceViewer.Surface = marchingCubes.GetOutput()
    # SurfaceViewer.Display = 1
    # SurfaceViewer.Opacity = 0.5
    # SurfaceViewer.BuildView()
    # vmtkRenderer.Deallocate()

        # convert to Numpy data
        threshold = vtk.vtkImageThreshold()
        threshold.SetInputData(levelset.LevelSetsOutput)
        threshold.ThresholdByLower(0)
        threshold.ReplaceInOn()
        threshold.ReplaceOutOn()
        threshold.SetOutValue(0)
        threshold.SetInValue(1)
        threshold.Update()

        outVolumeData = vtk.vtkImageData()
        outVolumeData.DeepCopy(threshold.GetOutput())

            # numpy_label = numpy_support.vtk_to_numpy(outVolumeData.GetPointData().GetScalars())
            # numpy_label = numpy_label.reshape(dim[2], dim[1], dim[0])
            # numpy_label = numpy_label.transpose(2, 1, 0)
            # numpy_label = np.flip(np.flip(numpy_label, axis=1), axis=2)  # flip along the 2nd and 3rd axis
        temp_label = numpy_support.vtk_to_numpy(outVolumeData.GetPointData().GetScalars())
        temp_label = temp_label.reshape(dim[2], dim[1], dim[0])
        temp_label = temp_label.transpose(2, 1, 0)
        temp_label = np.flip(np.flip(temp_label, axis=1), axis=2)
        numpy_label = numpy_label + temp_label
    labelindex = np.argwhere(numpy_label == 1)
    center = points
    flag = True
    inputflag = input('是否选取闭塞段控制点:[y/n]')
    while inputflag != 'y' and inputflag != 'n':
        inputflag = input('是否选取闭塞段控制点:[y/n]')
    while inputflag == 'y':
        capture_mouse(dcm_sitk, win_min=-100, win_max=1200)
        inputflag = input('是否继续选取闭塞段控制点:[y/n]')
        while inputflag != 'y' and inputflag != 'n':
            inputflag = input('是否继续选取闭塞段控制点:[y/n]')
    pointsz = [temp[2] for temp in points]
    for i in range(dim[2]):
        if i not in pointsz:
            temp = labelindex[np.argwhere(labelindex[:, 2] == i), :]
            if temp.shape[0] >= 1:
                centerpoint = np.mean(temp, axis=0)
                center.append([int(centerpoint[0, 0]), int(centerpoint[0, 1]), int(centerpoint[0, 2])])
    center.sort(key=sort2)
    x = [temp[0] for temp in center]
    y = [temp[1] for temp in center]
    z = [temp[2] for temp in center]
    interpz = list(range(min(z), max(z) + 1))
    print(str(z))
    interpx, interpy= interp1(x, y, z, interpz)
    # for i in range(len(interpz)):
    #     numpy_label[int(interpx[i]), int(interpy[i]), int(interpz[i])] = 1
    # plt.figure()
    # plt.subplot(121)
    # plt.plot(interpz, interpx, 'o')
    # plt.subplot(122)
    # plt.plot(interpz, interpy, 'o')
    # plt.show()

    if len(sourcepoints) >= 1:
        if len(sourcepoints) > 1:
            for i in range(1, len(sourcepoints)):
                sourcepoint = np.asarray(sourcepoints[i])
                targetpoint = np.asarray(targetpoints[i - 1])
                deltaceter = sourcepoint - targetpoint
                deltaz = deltaceter[2]
                deltaceter = deltaceter[0:2]
                temp = np.squeeze(labelindex[np.argwhere(labelindex[:, 2] == targetpoint[2] - int(deltaz / abs(deltaz))), :])
                tz = targetpoint[2] - int(deltaz / abs(deltaz))
                while len(temp) <= 0:
                    temp = np.squeeze(labelindex[np.argwhere(labelindex[:, 2] == tz), :])
                    tz = tz - int(deltaz / abs(deltaz))
                targetpoint = np.mean(temp, axis=0)
                temp = temp[:, 0:2]
                sourcecircle = np.squeeze(labelindex[np.argwhere(labelindex[:, 2] == sourcepoint[2] + int(deltaz / abs(deltaz))), :])
                sz = sourcepoint[2] + int(deltaz / abs(deltaz))
                while len(sourcecircle) <= 0:
                    sourcecircle = np.squeeze(labelindex[np.argwhere(labelindex[:, 2] == sz), :])
                    sz = sz + int(deltaz / abs(deltaz))
                sourcepoint = np.mean(sourcecircle, axis=0)
                sourcecircle = sourcecircle[:, 0:2]
                deltaceter = sourcepoint - targetpoint
                deltaz = deltaceter[2]
                deltaceter = deltaceter[0:2]
                targetcircle = temp + deltaceter
                targetpointzindex = interpz.index(targetpoint[2])
                sourcepointzindex = interpz.index(sourcepoint[2])
                for j in range(len(targetcircle)):
                    d = []
                    for k in range(len(sourcecircle)):
                        d.append((abs(targetcircle[j, 0] - sourcecircle[k, 0]) ** 2 + abs(targetcircle[j, 1] - sourcecircle[k, 1]) ** 2) ** 0.5)
                    mind = min(d)
                    min_index = d.index(mind)
                    nearestpoint = sourcecircle[min_index, :]
                    delta = nearestpoint - targetcircle[j, :]
                    for k in range(1, int(abs(deltaz))):
                        temppoint = delta * (k / abs(deltaz)) + targetcircle[j, :]
                        realk = int(round(k / deltaz * abs(deltaz)))
                        realpoint = temppoint - (sourcepoint[0:2] - np.asarray([interpx[targetpointzindex + realk], interpy[targetpointzindex + realk]]))
                        numpy_label[int(realpoint[0]), int(realpoint[1]), interpz[targetpointzindex + realk]] = 2
            targetpoint = targetpoints[0]
            sourcepoint = sourcepoints[0]
            if targetpoint[-1] > sourcepoint[-1]:
                targetpoint = targetpoints[-1]
            else:
                sourcepoint = sourcepoints[-1]
            targetpoint = np.asarray(targetpoint)
            targetpoint = targetpoint.squeeze()
            sourcepoint = np.asarray(sourcepoint)
            sourcepoint = sourcepoint.squeeze()
        elif len(sourcepoints) == 1:
            targetpoint = np.asarray(targetpoints)
            targetpoint = targetpoint.squeeze()
            sourcepoint = np.asarray(sourcepoints)
            sourcepoint = sourcepoint.squeeze()
        deltaceter = sourcepoint - targetpoint
        deltaz = deltaceter[2]
        targetcircle = np.squeeze(labelindex[np.argwhere(labelindex[:, 2] == targetpoint[2] + int(deltaz / abs(deltaz))), :])
        tz = targetpoint[2] + int(deltaz / abs(deltaz))
        while len(targetcircle) <= 0:
            tz = tz + int(deltaz / abs(deltaz))
            targetcircle = np.squeeze(
                labelindex[np.argwhere(labelindex[:, 2] == tz), :])
        targetpoint = np.mean(targetcircle, axis=0)
        targetcircle = targetcircle[:, 0:2]
        sourcecircle = np.squeeze(
            labelindex[np.argwhere(labelindex[:, 2] == sourcepoint[2] - int(deltaz / abs(deltaz))), :])
        sz = sourcepoint[2] - int(deltaz / abs(deltaz))
        while len(sourcecircle) <= 0:
            sz = sz - int(deltaz / abs(deltaz))
            sourcecircle = np.squeeze(
                labelindex[np.argwhere(labelindex[:, 2] == sz), :])
        sourcepoint = np.mean(sourcecircle, axis=0)
        sourcecircle = sourcecircle[:, 0:2]
        targetpointzindex = interpz.index(targetpoint[2])
        sourcepointzindex = interpz.index(sourcepoint[2])
        if targetpointzindex > sourcepointzindex:
            for i in range(sourcepointzindex):
                delta = np.asarray([interpx[i], interpy[i]]) - sourcepoint[0:2]
                realcircle = sourcecircle + delta
                for j in range(len(realcircle)):
                    numpy_label[int(realcircle[j, 0]), int(realcircle[j, 1]), interpz[i]] = 2
            for i in range(targetpointzindex + 1, len(interpz)):
                delta = np.asarray([interpx[i], interpy[i]]) - targetpoint[0:2]
                realcircle = targetcircle + delta
                for j in range(len(realcircle)):
                    numpy_label[int(realcircle[j, 0]), int(realcircle[j, 1]), interpz[i]] = 2
        else:
            for i in range(targetpointzindex):
                delta = np.asarray([interpx[i], interpy[i]]) - targetpoint[0:2]
                realcircle = targetcircle + delta
                for j in range(len(realcircle)):
                    numpy_label[int(realcircle[j, 0]), int(realcircle[j, 1]), interpz[i]] = 2
            for i in range(sourcepointzindex + 1, len(interpz)):
                delta = np.asarray([interpx[i], interpy[i]]) - sourcepoint[0:2]
                realcircle = sourcecircle + delta
                for j in range(len(realcircle)):
                    numpy_label[int(realcircle[j, 0]), int(realcircle[j, 1]), interpz[i]] = 2
    else:
        circle = np.asarray([[0, 3], [0, 2], [0, 4], [1, 1], [1, 5], [2, 1], [2, 5], [3, 0], [3, 6], [4, 1], [4, 5], [5, 1],
                  [5, 5], [6, 3], [6, 2], [6, 4]])
        for i in range(len(interpz)):
            delta = np.c_[interpx[i], interpy[i]] - [3, 3]
            realcircle = circle + delta
            for j in range(len(circle)):
                numpy_label[int(realcircle[j, 0]), int(realcircle[j, 1]), interpz[i]] = 2





    display3d(dcm_sitk, numpy_label.astype(np.uint8))

    result_sitk = sitk.GetImageFromArray(np.uint8(numpy_label.transpose(2, 1, 0)))
    result_sitk.CopyInformation(dcm_sitk)
    slice_seg_contours(dcm_sitk, result_sitk)
    # exit()

    thres = False
    if thres:
        intensityfilter = sitk.IntensityWindowingImageFilter()
        intensityfilter.SetWindowMaximum(np.double(900))
        intensityfilter.SetWindowMinimum(np.double(-100))
        intensityfilter.SetOutputMaximum(255)
        intensityfilter.SetOutputMinimum(0)
        dcm_thres = intensityfilter.Execute(dcm_sitk)
    else:
        dcm_thres = dcm_sitk

    numpy_img = np.transpose(sitk.GetArrayFromImage(dcm_thres), (2, 1, 0))
    index = np.where((numpy_label > 0))
    slices = np.unique(index[2])
    patch_dir = os.path.join(patch_root, filelist[patientid])
    if not os.path.exists(patch_dir):
        os.makedirs(patch_dir)
    #
    # box_height = 32
    # box_width = 32
    for s in range(slices.size):
        slice_num = slices[s]
        img_slice = numpy_img[:, :, slice_num]
        index_slice = np.where((numpy_label[:, :, slice_num] > 0))
        center_slice = [np.mean(index_slice[0]).astype(np.int), np.mean(index_slice[1]).astype(np.int)]
        # patch_slice = img_slice[center_slice[0] - box_height + 1: center_slice[0] + box_height + 1,
        #               center_slice[1] - box_width + 1: center_slice[1] + box_width + 1]

        patch_img_name = filelist[patientid] + "_{}_{}.bmp".format(side, slice_num)
        patch_path = os.path.join(patch_dir, patch_img_name)
        filter_intensity = sitk.IntensityWindowingImageFilter()
        filter_intensity.SetOutputMaximum(255)
        filter_intensity.SetOutputMinimum(0)
        filter_intensity.SetWindowMinimum(-100)
        filter_intensity.SetWindowMaximum(900)
        img_show = filter_intensity.Execute(dcm_sitk)
        itk_display = sitk.LabelOverlay(img_show[:, :, int(slice_num)], sitk.LabelContour(result_sitk[:, :, int(slice_num)]), 1.0)
        img_display = sitk.GetArrayFromImage(itk_display)
        cv.imwrite(patch_path, np.uint8(img_display))
        # cv.imshow('Segmentation', np.uint8(img_display))
        # np.savetxt(patch_path, patch_slice, fmt='%d', delimiter=' ')
        #
        # plt.figure()
        # plt.imshow(patch_slice, cmap='gray')
        # plt.show()
    exit()

