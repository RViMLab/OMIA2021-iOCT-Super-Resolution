import SimpleITK as sitk
import time
import os
import numpy as np
import cv2
from pathlib import Path



def registration(fixed,moving,cubebool=True):
    start = time.time()

    fixedImage=sitk.GetImageFromArray(fixed)

    movingImage = sitk.GetImageFromArray(moving)

    parameterMapVector = sitk.VectorOfParameterMap()


    if(cubebool):
        # Rigid Transformation
        parameterMap1 = sitk.GetDefaultParameterMap("rigid")

        parameterMapVector.append(parameterMap1)

        # Non Rigid Transformation
        parameterMap = sitk.GetDefaultParameterMap("bspline")
        parameterMap['GridSpacingSchedule'] = ['50','30.', '30.', '30.', '20.'] #for Cube 450x350


        parameterMap['NumberOfResolutions']=['5']
        parameterMap['MaximumNumberOfIterations'] = ['800']

        parameterMapVector.append(parameterMap)
    else:
        # Rigid Transformation
        parameterMap1 = sitk.GetDefaultParameterMap("rigid")
        parameterMap1['NumberOfResolutions'] = ['7']
        parameterMap1['FixedImagePyramidSchedule'] = [ '64','64','64','64','32', '32', '16', '16', '8', '8', '8',
                                                        '8', '8', '8']
        parameterMap1['MovingSmoothingImagePyramid'] = [ '64','64','64','64','40', '40', '30', '30', '20', '20', '16',
                                                        '16', '2', '2']
        parameterMapVector.append(parameterMap1)

        # Non Rigid Transformation
        parameterMap = sitk.GetDefaultParameterMap("bspline")
        parameterMap['GridSpacingSchedule'] = ['100', '80', '50', '30.', '30.', '30.', '20.']  # for 1024x1024 Cube
        # parameterMap['GridSpacingSchedule'] = ['5.', '5.', '3.', '0.5']

        parameterMap['NumberOfResolutions'] = ['7']
        parameterMap['FixedImagePyramidSchedule'] = ['128', '128', '64', '64', '32', '32', '16', '16', '8', '8', '4',
                                                     '4', '1', '1']
        parameterMap['MovingSmoothingImagePyramid'] = ['128', '128', '64', '64', '32', '32', '16', '16', '8', '8', '4',
                                                      '4', '1', '1']
        parameterMap['MaximumNumberOfIterations'] = ['800']
        parameterMapVector.append(parameterMap)


    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixedImage)
    elastixImageFilter.SetMovingImage(movingImage)
    elastixImageFilter.SetParameterMap(parameterMapVector)


    elastixImageFilter.Execute()

    ### Get Deformation Field
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
    transformixImageFilter.ComputeDeformationFieldOn()
    #transformixImageFilter.ComputeSpatialJacobianOn()
    transformixImageFilter.ComputeDeterminantOfSpatialJacobianOn()


    transformixImageFilter.Execute()
    deformationField = transformixImageFilter.GetDeformationField()
    resultImage = elastixImageFilter.GetResultImage()


    end = time.time()

    print("=================================================================")

    print("Registration done in:", end - start, "s")
    #elastixImageFilter.PrintParameterMap(parameterMap1)
    #elastixImageFilter.PrintParameterMap(parameterMap)


    return resultImage

def applyRegistration(target_path,path_fixed_folder,path_moving_folder,dim):


    # Apply Registration:
    for filename in os.listdir(path_fixed_folder):

        path_video = os.path.join(path_fixed_folder, filename)

        # Read,resize and numpy Video
        imgVideo = cv2.imread(path_video, 0)
        resizedVideo = cv2.resize(imgVideo, dim, interpolation=cv2.INTER_LINEAR)
        resizedVideo = resizedVideo.astype(np.float64)

        path_cube = os.path.join(path_moving_folder, filename)  # check video

        # Read,resize and numpy Cube
        img = cv2.imread(path_cube, 0)
        resizedCube = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
        resizedCube = resizedCube.astype(np.float64)

        # Define Fixed:Video, Moving:Cube
        fixed = resizedVideo
        moving = resizedCube

        # Run Registration
        movedITK = registration(fixed, moving)
        movedNumpy = sitk.GetArrayViewFromImage(movedITK)

        print("#End of Registration")

        # Plot and Write Images

        # plt.imshow(movedNumpy,cmap='gray')
        # plt.show()

        movedImage = movedNumpy
        moved_name = 'moved' + filename + '.jpg'
        path_registered = os.path.join(target_path, moved_name)
        cv2.imwrite(path_registered, movedImage)



def registrationLow(fixed,moving,apply_rigid):
    start = time.time()
    fixedImage = sitk.GetImageFromArray(fixed)
    movingImage = sitk.GetImageFromArray(moving)
    parameterMapVector = sitk.VectorOfParameterMap()

    if apply_rigid==False:
        parameterMap = sitk.GetDefaultParameterMap("bspline")
        parameterMap['Metric']= ['AdvancedMattesMutualInformation']
    else:
        parameterMap = sitk.GetDefaultParameterMap("rigid")
        parameterMap['Metric'] = ['AdvancedNormalizedCorrelation']

    parameterMap['GridSpacingSchedule'] = ['10'] #for Cube 450x350
    parameterMap['NumberOfResolutions']=['1']
    parameterMap['MaximumNumberOfIterations'] = ['500']
    parameterMapVector.append(parameterMap)

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixedImage)
    elastixImageFilter.SetMovingImage(movingImage)
    elastixImageFilter.SetParameterMap(parameterMap)

    elastixImageFilter.Execute()
    resultImage = elastixImageFilter.GetResultImage()
    #prm=elastixImageFilter.GetParameterMap()
    #elastixImageFilter.PrintParameterMap((prm))
    end = time.time()

    print("=================================================================")

    print("Registration done in:", end - start, "s")

    return resultImage

def applyExtraRegistration(target_path,path_extra_fixed_folder,path_moving_folder,apply_rigid,dim):
    dim=(390,150)

    registrArray=[]
    oldname='000'
    Path(target_path + "/test_target").mkdir(parents=True, exist_ok=True)

    for filename_mov in os.listdir(path_moving_folder):


        if filename_mov[0:4]==oldname:

            registrArray.append(filename_mov)
        else:

            ####################################
            ### End of finding same name frames,
            ### Registration using [0] as fixed
            ### and the rest as moving images
            if (np.size(registrArray) > 0):
                print(registrArray)
                path_fix = os.path.join(path_extra_fixed_folder, registrArray[0])
                imgFix = cv2.imread(path_fix, 0)
                resFix = imgFix.astype(np.float64)
                fixed = resFix

                fixed_name = registrArray[0][0:-4] + '.png'
                fixed_name = os.path.join(target_path, "test_target", fixed_name)
                cv2.imwrite(fixed_name, fixed)
                for reg_img in range(1,np.size(registrArray)):

                    path_mov = os.path.join(path_moving_folder, registrArray[reg_img])
                    print(path_fix)
                    print(path_mov)
                    imgMov = cv2.imread(path_mov, 0)
                    # resMov = cv2.resize(imgMov, dim, interpolation=cv2.INTER_LINEAR)
                    resMov = imgMov.astype(np.float64)
                    moving = resMov

                    movedITK = registrationLow(fixed, moving, apply_rigid)
                    movedNumpy = sitk.GetArrayViewFromImage(movedITK)
                    print("##########################End of Registration")

                    '''error = mse(resMov,fixed)
                    print(error)
                    print("after registration error: ",mse(movedNumpy.astype(np.uint8),fixed))
                    print(imgMov.dtype)
                    print(movedNumpy.dtype)
                    #cv2.imshow('abc',fixed.astype(np.uint8))
                    cv2.imshow('cde', imgMov)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()'''
                    print("##########################Saving Registration Result")
                    movedImage = movedNumpy

                    moved_name = registrArray[reg_img][0:-4] + '.png'

                    path_registered = os.path.join(target_path, "test_target", moved_name)
                    cv2.imwrite(path_registered, movedImage)

            # End of Registration
            ###################################




            registrArray=[]
            registrArray.append(filename_mov)
            oldname=filename_mov[0:4]



        '''
        path_mov = os.path.join(path_moving_folder, filename_mov)
        imgMov = cv2.imread(path_mov, 0)
        #resMov = cv2.resize(imgMov, dim, interpolation=cv2.INTER_LINEAR)
        resMov = imgMov.astype(np.float64)
        moving=resMov
        
        
        for filename_fix in os.listdir(path_extra_fixed_folder):
            print(filename_fix)
            print(filename_mov)

            #if(filename_mov[0:3]==filename_fix[0:3]):
            if (filename_mov[0:1] == filename_fix[0:1]):

                #Define Fixed Image
                path_fix = os.path.join(path_extra_fixed_folder, filename_fix)
                imgFix = cv2.imread(path_fix, 0)
                #resFix = cv2.resize(imgFix, dim, interpolation=cv2.INTER_LINEAR)
                resFix = imgFix.astype(np.float64)
                fixed=resFix

                # Run Registration
                movedITK = registrationLow(fixed, moving,apply_rigid)
                movedNumpy = sitk.GetArrayViewFromImage(movedITK)
                print("#End of Registration")

                print("Saving Registration Result")
                movedImage = movedNumpy

                #moved_name = filename_fix[0:7] + '.jpg'
                moved_name = filename_mov[0:6] + 'moved.jpg'
                Path(target_path + "/test_target").mkdir(parents=True, exist_ok=True)
                path_registered = os.path.join(target_path, "test_target", moved_name)
                cv2.imwrite(path_registered, movedImage)'''