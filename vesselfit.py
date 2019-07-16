# import os
from vmtk import vmtkscripts
# import SimpleITK as sitk
# import cv2 as cv
import numpy as np
import vtk
from vtk.util import numpy_support
from interp import interp1
import copy


def sort2(s):
    return s[2]


def get_vessel(window):
    points = copy.deepcopy(window.end_points)
    numpy_label = copy.deepcopy(window.numpy_label)
    dim = copy.deepcopy(window.dim)
    imageport = window.dicomreader.GetOutputPort()
    sourcepoints = []
    targetpoints = []
    if len(points) >= 2:
        points.sort(key=sort2)
        for i in range(len(points)):
            if i % 2 == 0:
                sourcepoints.append(points[i])
            else:
                targetpoints.append(points[i])
        for i in range(len(sourcepoints)):
            vmtkImageInitialization = vmtkscripts.vmtkImageInitialization()
            vmtkImageInitialization.Interactive = 0
            evoi = vtk.vtkExtractVOI()
            evoi.SetInputConnection(imageport)
            evoi.SetVOI(0, dim[0] - 1, 0, dim[1] - 1, min(dim[2] - sourcepoints[i][2] - 1, dim[2] - targetpoints[i][2] - 1), max(dim[2] - sourcepoints[i][2] - 1, dim[2] - targetpoints[i][2] - 1))
            evoi.Update()
            voiimage = evoi.GetOutput()
            vmtkImageInitialization.Image = voiimage
            vmtkImageInitialization.UpperThreshold = 1000
            vmtkImageInitialization.LowerThreshold = 120
            vmtkImageInitialization.NegateImage = 0
            vmtkImageInitialization.SourcePoints = [sourcepoints[i][0], dim[1] - sourcepoints[i][1] - 1,
                                                    dim[2] - sourcepoints[i][2] - 1]
            vmtkImageInitialization.TargetPoints = [targetpoints[i][0], dim[1] - targetpoints[i][1] - 1,
                                                    dim[2] - targetpoints[i][2] - 1]
            vmtkImageInitialization.Execute()

            # Feature Image
            imageFeatures = vmtkscripts.vmtkImageFeatures()
            imageFeatures.Image = voiimage
            imageFeatures.FeatureImageType = 'upwind'
            imageFeatures.SigmoidRemapping = 0
            imageFeatures.DerivativeSigma = 0.0
            imageFeatures.UpwindFactor = 1.0
            imageFeatures.FWHMRadius = [1.0, 1.0, 1.0]
            imageFeatures.FWHMBackgroundValue = 0.0
            imageFeatures.Execute()

            # # Level Sets
            levelset = vmtkscripts.vmtkLevelSetSegmentation()
            levelset.Image = voiimage
            levelset.LevelSetsInput = vmtkImageInitialization.InitialLevelSets
            levelset.FeatureImage = imageFeatures.FeatureImage
            levelset.IsoSurfaceValue = 0.0
            levelset.LevelSetEvolution()

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
            temp_label = numpy_support.vtk_to_numpy(outVolumeData.GetPointData().GetScalars())
            temp_label = temp_label.reshape(abs(sourcepoints[i][2] - targetpoints[i][2]) + 1, dim[1], dim[0])
            temp_label = temp_label.transpose(2, 1, 0)
            temp_label = np.flip(np.flip(temp_label, axis=1), axis=2)
            numpy_label[:, :, min(sourcepoints[i][2], targetpoints[i][2]):(max(sourcepoints[i][2], targetpoints[i][2]) + 1)] = numpy_label[:, :, min(sourcepoints[i][2], targetpoints[i][2]):(max(sourcepoints[i][2], targetpoints[i][2]) + 1)] + temp_label
    labelindex = np.argwhere(numpy_label == 1)
    control_points = points
    pointsz = [temp[2] for temp in points]
    for i in range(dim[2]):
        if i not in pointsz:
            temp = labelindex[np.argwhere(labelindex[:, 2] == i), :]
            if temp.shape[0] >= 1:
                centerpoint = np.mean(temp, axis=0)
                control_points.append([int(centerpoint[0, 0]), int(centerpoint[0, 1]), int(centerpoint[0, 2])])
    for centerpoint in window.center:
        control_points.append(centerpoint)
    control_points.sort(key=sort2)
    x = [temp[0] for temp in control_points]
    y = [temp[1] for temp in control_points]
    z = [temp[2] for temp in control_points]
    interpz = list(range(min(z), max(z) + 1))
    interpx, interpy = interp1(x, y, z, interpz)
    for i in range(len(interpz)):
        window.vessel_center.append([int(interpx[i]), int(interpy[i]), int(interpz[i])])
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
                temp = np.squeeze(
                    labelindex[np.argwhere(labelindex[:, 2] == targetpoint[2] - int(deltaz / abs(deltaz))), :])
                tz = targetpoint[2] - int(deltaz / abs(deltaz))
                while len(temp) <= 0:
                    temp = np.squeeze(labelindex[np.argwhere(labelindex[:, 2] == tz), :])
                    tz = tz - int(deltaz / abs(deltaz))
                targetpoint = np.mean(temp, axis=0)
                temp = temp[:, 0:2]
                sourcecircle = np.squeeze(
                    labelindex[np.argwhere(labelindex[:, 2] == sourcepoint[2] + int(deltaz / abs(deltaz))), :])
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
                        d.append((abs(targetcircle[j, 0] - sourcecircle[k, 0]) ** 2 + abs(
                            targetcircle[j, 1] - sourcecircle[k, 1]) ** 2) ** 0.5)
                    mind = min(d)
                    min_index = d.index(mind)
                    nearestpoint = sourcecircle[min_index, :]
                    delta = nearestpoint - targetcircle[j, :]
                    for k in range(1, int(abs(deltaz))):
                        temppoint = delta * (k / abs(deltaz)) + targetcircle[j, :]
                        realk = int(round(k / deltaz * abs(deltaz)))
                        realpoint = temppoint - (sourcepoint[0:2] - np.asarray(
                            [interpx[targetpointzindex + realk], interpy[targetpointzindex + realk]]))
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
        targetcircle = np.squeeze(
            labelindex[np.argwhere(labelindex[:, 2] == targetpoint[2] + int(deltaz / abs(deltaz))), :])
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
        circle = np.asarray(
            [[0, 3], [0, 2], [0, 4], [1, 1], [1, 5], [2, 1], [2, 5], [3, 0], [3, 6], [4, 1], [4, 5], [5, 1],
             [5, 5], [6, 3], [6, 2], [6, 4]])
        for i in range(len(interpz)):
            delta = np.c_[interpx[i], interpy[i]] - [3, 3]
            realcircle = circle + delta
            for j in range(len(circle)):
                numpy_label[int(realcircle[j, 0]), int(realcircle[j, 1]), interpz[i]] = 2
    return numpy_label


# if __name__ == "__main__":
#     patientid = 1
#     side = 'r'
#     filelist = os.listdir(data_root)
#     file_path = os.path.join(data_root, filelist[patientid])
#     dcm_sitk = read_dcm_series(file_path)
#
#     reader = vtk.vtkDICOMImageReader()
#     reader.SetDirectoryName(file_path)
#     reader.Update()
#     image = reader.GetOutput()
#     a = image.GetDimensions()
#     numpy_label = np.zeros(a, dtype=np.float32)
#
#     # Level Sets Initialization
#
#
#     while True:
#         points = capture_mouse(dcm_sitk, win_min=-100, win_max=1200)
#         if len(points) % 2 == 0:
#             break
#         else:
#             print('Only select two points for the beginning and the end!')
#
#     get_vessel(points)
#
#     dim = reader.GetOutput().GetDimensions()
#     sourcepoints = []
#     targetpoints = []
#     for i in range(len(points)):
#         if i % 2 == 0:
#             sourcepoints.append(points[i])
#         else:
#             targetpoints.append(points[i])
#     for i in range(len(sourcepoints)):
#         vmtkImageInitialization = vmtkscripts.vmtkImageInitialization()
#         vmtkImageInitialization.Interactive = 0
#         vmtkImageInitialization.Image = reader.GetOutput()
#         vmtkImageInitialization.UpperThreshold = 1000
#         vmtkImageInitialization.LowerThreshold = 120
#         vmtkImageInitialization.NegateImage = 0
#         vmtkImageInitialization.SourcePoints = [sourcepoints[i][0], dim[1] - sourcepoints[i][1] - 1, dim[2] - sourcepoints[i][2] - 1]
#         vmtkImageInitialization.TargetPoints = [targetpoints[i][0], dim[1] - targetpoints[i][1] - 1, dim[2] - targetpoints[i][2] - 1]
#         vmtkImageInitialization.Execute()
#
#
#             # Feature Image
#         imageFeatures = vmtkscripts.vmtkImageFeatures()
#         imageFeatures.Image = reader.GetOutput()
#         imageFeatures.FeatureImageType = 'upwind'
#         imageFeatures.SigmoidRemapping = 0
#         imageFeatures.DerivativeSigma = 0.0
#         imageFeatures.UpwindFactor = 1.0
#         imageFeatures.FWHMRadius = [1.0, 1.0, 1.0]
#         imageFeatures.FWHMBackgroundValue = 0.0
#         imageFeatures.Execute()
#
#         # # Level Sets
#         levelset = vmtkscripts.vmtkLevelSetSegmentation()
#         levelset.Image = reader.GetOutput()
#         levelset.LevelSetsInput = vmtkImageInitialization.InitialLevelSets
#         levelset.FeatureImage = imageFeatures.FeatureImage
#         levelset.IsoSurfaceValue = 0.0
#         levelset.LevelSetEvolution()
#
#     # Display
#     # imageview = vmtkscripts.vmtkImageViewer()
#     # imageview.Image = reader.GetOutput()
#     # imageview.Display = 1
#     # imageview.Execute()
#     #
#     # vmtkRenderer = vmtkscripts.vmtkRenderer()
#     # vmtkRenderer.Initialize()
#     # vmtkRenderer.RegisterScript(vmtkImageInitialization)
#     #
#     # SurfaceViewer = vmtkscripts.vmtkSurfaceViewer()
#     # SurfaceViewer.vmtkRenderer = vmtkRenderer
#     #
#     # marchingCubes = vtk.vtkMarchingCubes()
#     # marchingCubes.SetInputData(levelset.LevelSetsOutput)
#     # marchingCubes.SetValue(0, 0)
#     # marchingCubes.Update()
#     #
#     # SurfaceViewer.Surface = marchingCubes.GetOutput()
#     # SurfaceViewer.Display = 1
#     # SurfaceViewer.Opacity = 0.5
#     # SurfaceViewer.BuildView()
#     # vmtkRenderer.Deallocate()
#
#         # convert to Numpy data
#         threshold = vtk.vtkImageThreshold()
#         threshold.SetInputData(levelset.LevelSetsOutput)
#         threshold.ThresholdByLower(0)
#         threshold.ReplaceInOn()
#         threshold.ReplaceOutOn()
#         threshold.SetOutValue(0)
#         threshold.SetInValue(1)
#         threshold.Update()
#
#         outVolumeData = vtk.vtkImageData()
#         outVolumeData.DeepCopy(threshold.GetOutput())
#
#             # numpy_label = numpy_support.vtk_to_numpy(outVolumeData.GetPointData().GetScalars())
#             # numpy_label = numpy_label.reshape(dim[2], dim[1], dim[0])
#             # numpy_label = numpy_label.transpose(2, 1, 0)
#             # numpy_label = np.flip(np.flip(numpy_label, axis=1), axis=2)  # flip along the 2nd and 3rd axis
#         temp_label = numpy_support.vtk_to_numpy(outVolumeData.GetPointData().GetScalars())
#         temp_label = temp_label.reshape(dim[2], dim[1], dim[0])
#         temp_label = temp_label.transpose(2, 1, 0)
#         temp_label = np.flip(np.flip(temp_label, axis=1), axis=2)
#         numpy_label = numpy_label + temp_label
#     labelindex = np.argwhere(numpy_label == 1)
#     center = points
#     flag = True
#     inputflag = input('是否选取闭塞段控制点:[y/n]')
#     while inputflag != 'y' and inputflag != 'n':
#         inputflag = input('是否选取闭塞段控制点:[y/n]')
#     while inputflag == 'y':
#         capture_mouse(dcm_sitk, win_min=-100, win_max=1200)
#         inputflag = input('是否继续选取闭塞段控制点:[y/n]')
#         while inputflag != 'y' and inputflag != 'n':
#             inputflag = input('是否继续选取闭塞段控制点:[y/n]')
#     pointsz = [temp[2] for temp in points]
#     for i in range(dim[2]):
#         if i not in pointsz:
#             temp = labelindex[np.argwhere(labelindex[:, 2] == i), :]
#             if temp.shape[0] >= 1:
#                 centerpoint = np.mean(temp, axis=0)
#                 center.append([int(centerpoint[0, 0]), int(centerpoint[0, 1]), int(centerpoint[0, 2])])
#     center.sort(key=sort2)
#     x = [temp[0] for temp in center]
#     y = [temp[1] for temp in center]
#     z = [temp[2] for temp in center]
#     interpz = list(range(min(z), max(z) + 1))
#     print(str(z))
#     interpx, interpy= interp1(x, y, z, interpz)
#     # for i in range(len(interpz)):
#     #     numpy_label[int(interpx[i]), int(interpy[i]), int(interpz[i])] = 1
#     # plt.figure()
#     # plt.subplot(121)
#     # plt.plot(interpz, interpx, 'o')
#     # plt.subplot(122)
#     # plt.plot(interpz, interpy, 'o')
#     # plt.show()
#
#     if len(sourcepoints) >= 1:
#         if len(sourcepoints) > 1:
#             for i in range(1, len(sourcepoints)):
#                 sourcepoint = np.asarray(sourcepoints[i])
#                 targetpoint = np.asarray(targetpoints[i - 1])
#                 deltaceter = sourcepoint - targetpoint
#                 deltaz = deltaceter[2]
#                 deltaceter = deltaceter[0:2]
#                 temp = np.squeeze(labelindex[np.argwhere(labelindex[:, 2] == targetpoint[2] - int(deltaz / abs(deltaz))), :])
#                 tz = targetpoint[2] - int(deltaz / abs(deltaz))
#                 while len(temp) <= 0:
#                     temp = np.squeeze(labelindex[np.argwhere(labelindex[:, 2] == tz), :])
#                     tz = tz - int(deltaz / abs(deltaz))
#                 targetpoint = np.mean(temp, axis=0)
#                 temp = temp[:, 0:2]
#                 sourcecircle = np.squeeze(labelindex[np.argwhere(labelindex[:, 2] == sourcepoint[2] + int(deltaz / abs(deltaz))), :])
#                 sz = sourcepoint[2] + int(deltaz / abs(deltaz))
#                 while len(sourcecircle) <= 0:
#                     sourcecircle = np.squeeze(labelindex[np.argwhere(labelindex[:, 2] == sz), :])
#                     sz = sz + int(deltaz / abs(deltaz))
#                 sourcepoint = np.mean(sourcecircle, axis=0)
#                 sourcecircle = sourcecircle[:, 0:2]
#                 deltaceter = sourcepoint - targetpoint
#                 deltaz = deltaceter[2]
#                 deltaceter = deltaceter[0:2]
#                 targetcircle = temp + deltaceter
#                 targetpointzindex = interpz.index(targetpoint[2])
#                 sourcepointzindex = interpz.index(sourcepoint[2])
#                 for j in range(len(targetcircle)):
#                     d = []
#                     for k in range(len(sourcecircle)):
#                         d.append((abs(targetcircle[j, 0] - sourcecircle[k, 0]) ** 2 + abs(targetcircle[j, 1] - sourcecircle[k, 1]) ** 2) ** 0.5)
#                     mind = min(d)
#                     min_index = d.index(mind)
#                     nearestpoint = sourcecircle[min_index, :]
#                     delta = nearestpoint - targetcircle[j, :]
#                     for k in range(1, int(abs(deltaz))):
#                         temppoint = delta * (k / abs(deltaz)) + targetcircle[j, :]
#                         realk = int(round(k / deltaz * abs(deltaz)))
#                         realpoint = temppoint - (sourcepoint[0:2] - np.asarray([interpx[targetpointzindex + realk], interpy[targetpointzindex + realk]]))
#                         numpy_label[int(realpoint[0]), int(realpoint[1]), interpz[targetpointzindex + realk]] = 2
#             targetpoint = targetpoints[0]
#             sourcepoint = sourcepoints[0]
#             if targetpoint[-1] > sourcepoint[-1]:
#                 targetpoint = targetpoints[-1]
#             else:
#                 sourcepoint = sourcepoints[-1]
#             targetpoint = np.asarray(targetpoint)
#             targetpoint = targetpoint.squeeze()
#             sourcepoint = np.asarray(sourcepoint)
#             sourcepoint = sourcepoint.squeeze()
#         elif len(sourcepoints) == 1:
#             targetpoint = np.asarray(targetpoints)
#             targetpoint = targetpoint.squeeze()
#             sourcepoint = np.asarray(sourcepoints)
#             sourcepoint = sourcepoint.squeeze()
#         deltaceter = sourcepoint - targetpoint
#         deltaz = deltaceter[2]
#         targetcircle = np.squeeze(labelindex[np.argwhere(labelindex[:, 2] == targetpoint[2] + int(deltaz / abs(deltaz))), :])
#         tz = targetpoint[2] + int(deltaz / abs(deltaz))
#         while len(targetcircle) <= 0:
#             tz = tz + int(deltaz / abs(deltaz))
#             targetcircle = np.squeeze(
#                 labelindex[np.argwhere(labelindex[:, 2] == tz), :])
#         targetpoint = np.mean(targetcircle, axis=0)
#         targetcircle = targetcircle[:, 0:2]
#         sourcecircle = np.squeeze(
#             labelindex[np.argwhere(labelindex[:, 2] == sourcepoint[2] - int(deltaz / abs(deltaz))), :])
#         sz = sourcepoint[2] - int(deltaz / abs(deltaz))
#         while len(sourcecircle) <= 0:
#             sz = sz - int(deltaz / abs(deltaz))
#             sourcecircle = np.squeeze(
#                 labelindex[np.argwhere(labelindex[:, 2] == sz), :])
#         sourcepoint = np.mean(sourcecircle, axis=0)
#         sourcecircle = sourcecircle[:, 0:2]
#         targetpointzindex = interpz.index(targetpoint[2])
#         sourcepointzindex = interpz.index(sourcepoint[2])
#         if targetpointzindex > sourcepointzindex:
#             for i in range(sourcepointzindex):
#                 delta = np.asarray([interpx[i], interpy[i]]) - sourcepoint[0:2]
#                 realcircle = sourcecircle + delta
#                 for j in range(len(realcircle)):
#                     numpy_label[int(realcircle[j, 0]), int(realcircle[j, 1]), interpz[i]] = 2
#             for i in range(targetpointzindex + 1, len(interpz)):
#                 delta = np.asarray([interpx[i], interpy[i]]) - targetpoint[0:2]
#                 realcircle = targetcircle + delta
#                 for j in range(len(realcircle)):
#                     numpy_label[int(realcircle[j, 0]), int(realcircle[j, 1]), interpz[i]] = 2
#         else:
#             for i in range(targetpointzindex):
#                 delta = np.asarray([interpx[i], interpy[i]]) - targetpoint[0:2]
#                 realcircle = targetcircle + delta
#                 for j in range(len(realcircle)):
#                     numpy_label[int(realcircle[j, 0]), int(realcircle[j, 1]), interpz[i]] = 2
#             for i in range(sourcepointzindex + 1, len(interpz)):
#                 delta = np.asarray([interpx[i], interpy[i]]) - sourcepoint[0:2]
#                 realcircle = sourcecircle + delta
#                 for j in range(len(realcircle)):
#                     numpy_label[int(realcircle[j, 0]), int(realcircle[j, 1]), interpz[i]] = 2
#     else:
#         circle = np.asarray([[0, 3], [0, 2], [0, 4], [1, 1], [1, 5], [2, 1], [2, 5], [3, 0], [3, 6], [4, 1], [4, 5], [5, 1],
#                   [5, 5], [6, 3], [6, 2], [6, 4]])
#         for i in range(len(interpz)):
#             delta = np.c_[interpx[i], interpy[i]] - [3, 3]
#             realcircle = circle + delta
#             for j in range(len(circle)):
#                 numpy_label[int(realcircle[j, 0]), int(realcircle[j, 1]), interpz[i]] = 2
#
#
#
#
#
#     display3d(dcm_sitk, numpy_label.astype(np.uint8))
#
#     result_sitk = sitk.GetImageFromArray(np.uint8(numpy_label.transpose(2, 1, 0)))
#     result_sitk.CopyInformation(dcm_sitk)
#     slice_seg_contours(dcm_sitk, result_sitk)
#     # exit()
#
#     thres = False
#     if thres:
#         intensityfilter = sitk.IntensityWindowingImageFilter()
#         intensityfilter.SetWindowMaximum(np.double(900))
#         intensityfilter.SetWindowMinimum(np.double(-100))
#         intensityfilter.SetOutputMaximum(255)
#         intensityfilter.SetOutputMinimum(0)
#         dcm_thres = intensityfilter.Execute(dcm_sitk)
#     else:
#         dcm_thres = dcm_sitk
#
#     numpy_img = np.transpose(sitk.GetArrayFromImage(dcm_thres), (2, 1, 0))
#     index = np.where((numpy_label > 0))
#     slices = np.unique(index[2])
#     patch_dir = os.path.join(patch_root, filelist[patientid])
#     if not os.path.exists(patch_dir):
#         os.makedirs(patch_dir)
#     #
#     # box_height = 32
#     # box_width = 32
#     for s in range(slices.size):
#         slice_num = slices[s]
#         img_slice = numpy_img[:, :, slice_num]
#         index_slice = np.where((numpy_label[:, :, slice_num] > 0))
#         center_slice = [np.mean(index_slice[0]).astype(np.int), np.mean(index_slice[1]).astype(np.int)]
#         # patch_slice = img_slice[center_slice[0] - box_height + 1: center_slice[0] + box_height + 1,
#         #               center_slice[1] - box_width + 1: center_slice[1] + box_width + 1]
#
#         patch_img_name = filelist[patientid] + "_{}_{}.bmp".format(side, slice_num)
#         patch_path = os.path.join(patch_dir, patch_img_name)
#         filter_intensity = sitk.IntensityWindowingImageFilter()
#         filter_intensity.SetOutputMaximum(255)
#         filter_intensity.SetOutputMinimum(0)
#         filter_intensity.SetWindowMinimum(-200)
#         filter_intensity.SetWindowMaximum(900)
#         img_show = filter_intensity.Execute(dcm_sitk)
#         itk_display = sitk.LabelOverlay(img_show[:, :, int(slice_num)], sitk.LabelContour(result_sitk[:, :, int(slice_num)]), 1.0)
#         img_display = sitk.GetArrayFromImage(itk_display)
#         cv.imwrite(patch_path, np.uint8(img_display))
#         # cv.imshow('Segmentation', np.uint8(img_display))
#         # np.savetxt(patch_path, patch_slice, fmt='%d', delimiter=' ')
#         #
#         # plt.figure()
#         # plt.imshow(patch_slice, cmap='gray')
#         # plt.show()
#     exit()


