import numpy as np
import cv2
import os
from scipy.ndimage import measurements
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import metrics
from layerSegmentation import refineBottom,findUpperWhitePixel,findUp
from registration import registration
from scipy.spatial import distance
from myplot import showManyImages

def open_files(target_path):

    #Save First Check:
    FCtxt = os.path.join(target_path, 'FirstCheck.txt')
    text_First_Check = open(FCtxt, "w")

    #Save Euclideian:
    Eutxt = os.path.join(target_path, 'Euclideian.txt')
    text_Eucl = open(Eutxt, "w")

    #Save Hausdorff:
    Htxt = os.path.join(target_path, 'Hausdorff.txt')
    text_H = open(Htxt, "w")

    #Save Final check (Hausdorff):
    HChecktxt = os.path.join(target_path, 'FinalCheckH.txt')
    text_Final_Check = open(HChecktxt, "w")

    #Save Crop Dimensions:
    minxtxt = os.path.join(target_path, 'Minx.txt')
    text_minx = open(minxtxt, "w")

    maxxtxt = os.path.join(target_path, 'Maxx.txt')
    text_maxx = open(maxxtxt, "w")

    minytxt = os.path.join(target_path, 'Miny.txt')
    text_miny = open(minytxt, "w")

    maxytxt = os.path.join(target_path, 'Maxy.txt')
    text_maxy = open(maxytxt, "w")

    #Save RPEFixed:
    RPEFtxt = os.path.join(target_path, 'RPEF.txt')
    text_RPEF = open(RPEFtxt, "w")

    #Save ILMFixed:
    ILMFtxt = os.path.join(target_path, 'ILMF.txt')
    text_ILMF = open(ILMFtxt, "w")

    return text_First_Check,text_Eucl,text_H,text_Final_Check,text_minx,text_maxx,text_miny,text_maxy,text_RPEF,text_ILMF


def read_crop_moved(target_path):
    movedImages = []
    AssignImages = []
    RPEmoved = []
    MinX = []
    MinY = []
    MaxX = []
    MaxY = []
    NumberOfIMAGES = 200

    ### Crop Details ###
    marginup = 100
    margindown = 50
    print('Path_register: ', target_path)
    CheckPair = np.ones((NumberOfIMAGES), dtype=bool)
    cnt = 0

    # Read and Crop MOVED IMAGES
    for filename in os.listdir(target_path):
        if (filename[0:8] == 'moved042'):
            path_register = os.path.join(target_path, filename)

            imgReg = cv2.imread(path_register, 0)
            imgReg = imgReg.astype(np.float64)

            movingImageCopy = imgReg.copy()
            movingImagergb = imgReg.astype(np.uint8)
            movingImagergb = cv2.cvtColor(movingImagergb, cv2.COLOR_GRAY2BGR)

            RPEpixelsMoved, maxRPEcol, minRPEcol = refineBottom(imgReg, True)

            if (maxRPEcol == 0):
                CheckPair[cnt] = False
                AssignImages.append(movingImagergb)
                MinX.append(0)
                MaxX.append(0)
                MinY.append(0)
                MaxY.append(0)
                RPEmoved.append(np.arange(1))
                movedImages.append(np.zeros((150, 380)))

            else:
                x_range = RPEpixelsMoved[0][minRPEcol:maxRPEcol]
                y_range = np.arange(maxRPEcol - minRPEcol)
                fit = np.polyfit(y_range, x_range, 5)
                lspace = np.linspace(0, maxRPEcol - minRPEcol, maxRPEcol - minRPEcol)

                draw_x = lspace
                draw_y = np.polyval(fit, draw_x)  # evaluate the polynomial

                draw_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32)  # needs to be int32 and transposed

                cv2.polylines(movingImagergb[:, minRPEcol:maxRPEcol], [draw_points], False, (0, 255, 0),
                              2)  # args: image, points, closed, color

                ############ Show test#######
                mindraw = np.amin(draw_y)
                maxdraw = np.amax(draw_y)

                # Find if repeated black pixels exist:
                imgCr = imgReg.astype(np.uint8)
                if (maxdraw > marginup):
                    cropimgReg = imgCr[int(maxdraw) - marginup:int(maxdraw) + margindown, minRPEcol:maxRPEcol]
                else:
                    cropimgReg = imgCr[:int(maxdraw) + margindown, minRPEcol:maxRPEcol]
                    CheckPair[cnt] = False

                cropimgRegrgb = cropimgReg.astype(np.uint8)
                cropimgRegrgb = cv2.cvtColor(cropimgRegrgb, cv2.COLOR_GRAY2BGR)

                contArr = []
                for i in range(cropimgReg.shape[1]):

                    testcol = cropimgReg[:, i]
                    testcol[testcol > 0] = 1
                    indices_one = testcol == 1
                    indices_zero = testcol == 0
                    testcol[indices_one] = 0
                    testcol[indices_zero] = 1

                    cluster, freq = measurements.label(testcol)
                    if (freq > 0):

                        maxcontinuousblack = np.amax(list(Counter(cluster).values())[1:])
                        if (maxcontinuousblack > 30):
                            contArr.append(0)
                        else:
                            contArr.append(1)
                    else:
                        contArr.append(1)

                nonZeroIndices = np.nonzero(contArr)
                extramn = nonZeroIndices[0][0]
                extramx = nonZeroIndices[0][-1]

                if (maxdraw > marginup):
                    mnx = int(maxdraw) - marginup
                else:
                    mnx = 0
                mxx = int(maxdraw) + margindown
                mny = (extramn + 5) + minRPEcol
                mxy = minRPEcol + extramx

                MinX.append(mnx)
                MaxX.append(mxx)
                MinY.append(mny)
                MaxY.append(mxy)

                imgRegRGB = movingImagergb[mnx:mxx, mny:mxy]
                draw_y_cropped = draw_y[extramn + 5:extramx]

                AssignImages.append(imgRegRGB)
                # plt.imshow(imgRegRGB)
                # plt.show()
                RPEmoved.append(draw_y_cropped)

                movingImage = movingImageCopy[mnx:mxx, mny:mxy]
                movedImages.append(movingImage)
            cnt = cnt + 1

    return movedImages, AssignImages, RPEmoved, MinX, MinY, MaxX, MaxY,CheckPair

def validateReg(target_path,path_fixed_folder,dim):

    movedImages, AssignImages, RPEmoved, MinX, MinY, MaxX, MaxY, CheckPair = read_crop_moved(target_path)


    fixedImages = []
    RPEfixed = []
    ILMfixed= []
    cnt = 0
    cntvalidimag = 0
    saveImages=[]

    for filename in os.listdir(target_path):
        if (CheckPair[cnt]):
            video_file = os.path.join(path_fixed_folder, filename[5:5 + 7])

            imgVid = cv2.imread(video_file, 0)
            imgVid = imgVid.astype(np.float64)
            imgVid = cv2.resize(imgVid, dim, interpolation=cv2.INTER_LINEAR)

            croppedFixedGray = imgVid[MinX[cnt]:MaxX[cnt], MinY[cnt]:MaxY[cnt]]
            croppedFixedGrayCopy = croppedFixedGray.copy()
            croppedFixedRGB = imgVid[MinX[cnt]:MaxX[cnt], MinY[cnt]:MaxY[cnt]]


            RPEpixelsFixed, maxRPEcol2, minRPEcol2 = refineBottom(croppedFixedGray, False)

            ## check whether we can fit a line
            x_range = RPEpixelsFixed[0]
            y_range = np.arange((croppedFixedGray).shape[1])

            fit = np.polyfit(y_range, x_range, 5)
            lspace = np.linspace(0, (croppedFixedGray).shape[1], (croppedFixedGray).shape[1])

            draw_x = lspace
            draw_y = np.polyval(fit, draw_x)  # evaluate the polynomial


            draw_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32)  # needs to be int32 and transposed

            cropped_TO_PLOT = croppedFixedRGB.astype(np.uint8)
            cropped_TO_PLOT = cv2.cvtColor(cropped_TO_PLOT, cv2.COLOR_GRAY2BGR)
            cv2.polylines(cropped_TO_PLOT, [draw_points], False, (0, 255, 0),2)  # args: image, points, closed, color


            cv2.polylines(AssignImages[cntvalidimag], [draw_points], False, (0, 255, 0),2)  # args: image, points, closed, color

            ILMpixelsFixed, binary = findUp(croppedFixedGray,saveImages,RPEpixelsFixed[0])

            ## check whether we can fit a line
            x_ILM = ILMpixelsFixed[0]
            y_ILM = np.arange((croppedFixedGray).shape[1])

            fit_ILM = np.polyfit(y_ILM, x_ILM, 5)
            lspace_ILM = np.linspace(0, (croppedFixedGray).shape[1], (croppedFixedGray).shape[1])

            draw_x_ILM = lspace_ILM
            draw_y_ILM = np.polyval(fit_ILM, draw_x_ILM)  # evaluate the polynomial

            draw_points_ILM = (np.asarray([draw_x_ILM, draw_y_ILM]).T).astype(np.int32)  # needs to be int32 and transposed
            cv2.polylines(croppedFixedRGB, [draw_points_ILM], False, (255, 255, 0),2)


            justsave=croppedFixedRGB
            res_movingImagergb = cv2.resize(justsave, (380, 150), interpolation=cv2.INTER_LINEAR)
            saveImages.append(res_movingImagergb)

            draw_y_ILM=draw_y_ILM+MinX[cnt]
            ILMfixed.append(draw_y_ILM)

            draw_y=draw_y+MinX[cnt]
            RPEfixed.append(draw_y)

            fixedImages.append(croppedFixedGrayCopy)
            cntvalidimag = cntvalidimag + 1
            cnt = cnt + 1

        else:
            cnt = cnt + 1
            RPEfixed.append(np.arange(1))
            ILMfixed.append(np.arange(1))
            fixedImages.append(np.zeros((150,380)))


    print("#########################################")
    print("RPEFixed Size", np.array(RPEfixed).shape)
    print("ILMFixed Size", np.array(ILMfixed).shape)
    print("RPEMoved Size", np.array(RPEmoved).shape)

    SumDistances = isValid(RPEfixed, ILMfixed, RPEmoved, MinX, MaxX, MinY, MaxY, CheckPair, fixedImages, movedImages, target_path,AssignImages)

    return SumDistances, AssignImages


def isValid(RPEfixed,ILMfixed,RPEmoved,MinX,MaxX,MinY,MaxY,CheckPair,fixedImages,movedImages,target_path,AssignImages):

    text_First_Check,text_Eucl,text_H,text_Final_Check,text_minx,text_maxx,text_miny,text_maxy,text_RPEF,text_ILMF=open_files(target_path)

    SumDistances = 0
    cntDistances = 0
    countvalidimages = 0


    for i in range(np.array(np.array(RPEfixed).shape[0])):
        print("Register Pair: ",i)

        d=300
        h=300
        val=False

        set_ay = RPEfixed[i]-MinX[i]
        set_ax = np.arange(set_ay.shape[0])
        set_by = RPEmoved[i]-MinX[i]

        if(CheckPair[i]==True):
            set_bx = np.arange(set_by.shape[0])

            coords_a = np.zeros((set_ay.shape[0], fixedImages[i].shape[0]), dtype=bool)
            coords_b = np.zeros((set_ay.shape[0], fixedImages[i].shape[0]), dtype=bool)

            for x, y in zip(set_ax, set_ay):
                coords_a[(int(x), int(y))] = True

            for x, y in zip(set_bx, set_by):
                coords_b[(int(x), int(y))] = True

            h=metrics.hausdorff_distance(coords_a, coords_b)
            print("Hausdorff Distance: ", h)
            if (h < 20):
                SumDistances = SumDistances + h
                cntDistances += 1
                if(fixedImages[i].shape[1]>380):
                    countvalidimages=countvalidimages+1
                    val = True


            d = distance.euclidean(RPEfixed[i], RPEmoved[i])
            print("Euclidean distance: ", d)

            Path(target_path + "/cropped_fixed").mkdir(parents=True, exist_ok=True)
            path_cropped_fixed = os.path.join(target_path, "cropped_fixed", str("{:03}".format(i)) + '.jpg')

            Path(target_path + "/cropped_moved").mkdir(parents=True, exist_ok=True)
            path_cropped_moved = os.path.join(target_path, "cropped_moved", str("{:03}".format(i)) + '.jpg')

            if(val):
                fImage=fixedImages[i]
                fImage = cv2.resize(fImage, (380,150), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(path_cropped_fixed, fImage)

                mImage=movedImages[i]
                mImage = cv2.resize(mImage, (380,150), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(path_cropped_moved, mImage)

                text_minx.write("%s\n" % str(MinX[i]))
                text_maxx.write("%s\n" % str(MaxX[i]))
                text_miny.write("%s\n" % str(MinY[i]))
                text_maxy.write("%s\n" % str(MaxY[i]))
                np.set_printoptions(linewidth = np.inf)
                text_RPEF.write("%s\n" % str(RPEfixed[i]))
                text_ILMF.write("%s\n" % str(ILMfixed[i]))
            else:
                fImage=np.zeros((150,380))
                cv2.imwrite(path_cropped_fixed, fImage)
                mImage = np.zeros((150, 380))
                cv2.imwrite(path_cropped_moved, mImage)
                text_minx.write("%s\n" % str(np.inf))
                text_maxx.write("%s\n" % str(np.inf))
                text_miny.write("%s\n" % str(np.inf))
                text_maxy.write("%s\n" % str(np.inf))
                temprpe=np.zeros(RPEfixed[i].shape[0])
                text_RPEF.write("%s\n" % str(temprpe))
                text_ILMF.write("%s\n" % str(temprpe))


        Path(target_path + "/RPE").mkdir(parents=True, exist_ok=True)
        path_RPE = os.path.join(target_path, "RPE", str(i) + '.jpg')
        cv2.imwrite(path_RPE, AssignImages[i])

        Path(target_path + "/ILM").mkdir(parents=True, exist_ok=True)
        path_ILM = os.path.join(target_path, "ILM", str(i) + '.jpg')
        #cv2.imwrite(path_ILM, saveImages[i])

        text_First_Check.write("%s\n" % str(CheckPair[i]))
        text_Eucl.write("%s\n" % str(d))
        text_H.write("%s\n" % str(h))
        text_Final_Check.write("%s\n" % str(val))


    print("#########################################")
    print("End of Process")
    print("Total number of paired images", np.array(RPEfixed).shape[0])
    print("Mean Hausdorff Distance: ", SumDistances / cntDistances)
    print("Number of Valid Pairs: ", cntDistances)
    print("Number of final valid pairs",countvalidimages)
    print("#########################################")

    return SumDistances


