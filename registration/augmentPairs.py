import os
import cv2
from distutils.util import strtobool
import matplotlib.pyplot as plt
import numpy as np
from skimage import metrics
from validate import refineBottom,refineUp,findUp
from pathlib import Path
from myplot import showManyImages

def getMoreFrames(videoname,target_path,path_root):

    #Define crop_fixed folder
    path_cr_fixed=os.path.join(target_path, 'cropped_fixed')

    # Define valid Pairs - Read valid list
    HChecktxt = os.path.join(target_path, 'FinalCheckH.txt')
    text_Final_Check = open(HChecktxt, "r")

    dim=(450,300)
    # Define Crop Margins - Read txt files
    minxtxt = os.path.join(target_path, 'Minx.txt')
    text_minx = open(minxtxt, "r")

    maxxtxt = os.path.join(target_path, 'Maxx.txt')
    text_maxx = open(maxxtxt, "r")

    minytxt = os.path.join(target_path, 'Miny.txt')
    text_miny = open(minytxt, "r")

    maxytxt = os.path.join(target_path, 'Maxy.txt')
    text_maxy = open(maxytxt, "r")

    RPEFtxt = os.path.join(target_path, 'RPEF.txt')
    text_RPEF = open(RPEFtxt, "r")

    ILMFtxt = os.path.join(target_path, 'ILMF.txt')
    text_ILMF = open(ILMFtxt, "r")

    RPE_FIXED=[]
    ILM_FIXED=[]
    i=0
    cntframerarr = []
    numbclosepairs = []
    saveIMages=[]
    pltImages=[]

    for filename in os.listdir(path_cr_fixed):
        ############################################################
        path_register = os.path.join(path_cr_fixed, filename)
        imgReg = cv2.imread(path_register, 0)
        imgReg = imgReg.astype(np.float64)
        previous_saved_img=imgReg
        #############################################################

        close_pairs = 0
        print(filename)
        # Define Video
        videoFilename = os.path.join(path_root, videoname)
        cap = cv2.VideoCapture(videoFilename)
        w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)

        str_valid=(text_Final_Check.readline())
        print(str_valid)
        if (str_valid[0:4] == 'True'):
            valid= True
        elif (str_valid[0:5] == 'False'):
            valid= False

        mnx = text_minx.readline()
        mxx = text_maxx.readline()
        mny = text_miny.readline()
        mxy = text_maxy.readline()
        RPE_str = text_RPEF.readline()
        ILM_str=text_ILMF.readline()

        # RPE txt to RPE matrix
        RPE_frame=editRPETxt(RPE_str)
        ILM_frame=editRPETxt(ILM_str)

        #test_frame = myDrawPoints(RPE_frame, imgReg)


        RPE_FIXED.append(RPE_frame)
        ILM_FIXED.append(ILM_frame)

        if(valid ):
            mnx = int(mnx)
            #print("minx: ",mnx)
            mxx = int(mxx)
            #print("maxx: ", mxx)
            mny = int(mny)
            #print("miny: ", mny)
            mxy = int(mxy)
            #print("maxx: ",mxy)
            RPE_frame = RPE_frame-mnx
            ILM_frame=ILM_frame-mnx

            cntframe=0
            while (cap.isOpened()):
                ret, frame = cap.read()
                if (ret == False):
                    plt.imshow(np.zeros((350,350)))
                    numbclosepairs.append(close_pairs)
                    break


                # Grayscale, Resize and Crop]
                #frame2 = cv2.resize(frame, (1024, 1024), interpolation=cv2.INTER_LINEAR)

                #frame2=frame[mnx:mxx]
                edit_frame=editVideoFrame(frame,dim,mnx,mxx,mny,mxy)
                our_frame=edit_frame.copy()

                # Find checked frame RPE:
                RPE_check, maxRPEcol2, minRPEcol2 = refineBottom(edit_frame, False)

                if(maxRPEcol2==0 and minRPEcol2==0):
                    RPE_checked = np.zeros(mxy-mny)
                    show_frame=edit_frame
                else:
                    show_frame,RPE_checked=calcRPE(RPE_check,edit_frame)

                h=computeHausdorff(RPE_frame,RPE_checked)

                RPE_frame=RPE_frame.astype(int)

                #show_frame=myDrawPoints(RPE_frame,show_frame)
                #show_frame = myDrawPoints(RPE_frame, show_frame)

                print("frame",cntframe)
                print("Hausdorff Distance RPE: ", h)
                print("Close pairs", close_pairs)

                # Find checked frame ILM
                #refineUp(edit_frame, RPE_check)
                #pltImages.append(show_frame)


                #if(cntframe/50==1):
                    #showManyImages(pltImages)


                if(h<15):
                    #cv2.imshow('frame2', show_frame)
                    #cv2.imshow('frame', show_frame)
                    #plt.imshow(show_frame)
                    #plt.show()
                    exists = cntframe in cntframerarr


                    # Find checked frame ILM:
                    ##ILM_check, binary = findUp(edit_frame, saveIMages, RPE_check[0])
                    ##show_frame2, ILM_checked = calcILM(ILM_check, edit_frame)
                    ##myDrawPoints(ILM_frame, show_frame2)
                    #cv2.imshow('frame', show_frame2)
                    ##h_ilm = computeHausdorff(ILM_frame, ILM_checked)
                    h_ilm=5
                    print("Hausdorff Distance ILM: ", h_ilm)
                    if(h_ilm<10):

                        if(exists==False):
                            mse_error=mse(previous_saved_img,our_frame)
                            print("MSE",mse_error)
                            diff_frame=previous_saved_img-our_frame
                            plt.imshow(diff_frame,cmap='gray')
                            plt.show()
                            if(mse_error>0.03):
                                cntframerarr.append(cntframe)
                                Path(target_path + "/extrafixedDeleted").mkdir(parents=True, exist_ok=True)
                                path_extrafixed = os.path.join(target_path, "extrafixedDeleted", filename[0:3]+'-'+str("{:03}".format(close_pairs))+'-' + str(cntframe)+'.jpg')
                                cv2.imwrite(path_extrafixed, our_frame)
                                previous_saved_img=our_frame
                                close_pairs = close_pairs + 1

                if(close_pairs>10):
                    break

                cntframe = cntframe + 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if(ret==False):
                    break
            print("end of video")
            cap.release()
            cv2.destroyAllWindows()


    print(cntframerarr)
    print(numbclosepairs)
    print(np.sum(numbclosepairs))

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def calcRPE(RPE_checked,edit_frame):

    x_range = RPE_checked[0]
    print("RPE",RPE_checked[0])
    #print("RPE_CHECKED SHAPE",RPE_checked.shape[1])
    y_range = np.arange(RPE_checked.shape[1])

    fit = np.polyfit(y_range, x_range, 2)

    lspace = np.linspace(0,RPE_checked.shape[1], RPE_checked.shape[1])

    draw_x = lspace
    draw_y = np.polyval(fit, draw_x)  # evaluate the polynomial
    #print("RPE checking", draw_y)
    draw_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32)  # needs to be int32 and transposed

    cv2.polylines(edit_frame, [draw_points], False, (255, 255, 255), 1)  # args: image, points, closed, color


    return edit_frame,draw_y


def calcILM(RPE_checked,edit_frame):

    x_range = RPE_checked[0]
    #print("RPE_CHECKED SHAPE",RPE_checked.shape[1])
    y_range = np.arange(RPE_checked.shape[1])
    fit = np.polyfit(y_range, x_range, 5)

    lspace = np.linspace(0,RPE_checked.shape[1], RPE_checked.shape[1])

    draw_x = lspace

    draw_y = np.polyval(fit, draw_x)  # evaluate the polynomial

    draw_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32)  # needs to be int32 and transposed

    cv2.polylines(edit_frame, [draw_points], False, (255, 255, 255), 1)  # args: image, points, closed, color


    return edit_frame,draw_y


def computeHausdorff(RPEfixed,RPEchecked):

    set_ay = RPEfixed
    print("maximum balue in coords a is:",np.amax(set_ay))
    print("first value in coords a is:",set_ay[0])
    set_ax = np.arange(set_ay.shape[0])

    set_by = RPEchecked
    print("maximum balue in coords b is:",np.amax(set_by))
    print("first value in coords b is:",set_by[0])

    set_bx = np.arange(set_by.shape[0])

    coords_a = np.zeros((set_ay.shape[0], 150), dtype=bool)
    coords_b = np.zeros((set_ay.shape[0], 150), dtype=bool)

    for x, y in zip(set_ax, set_ay):
        coords_a[(int(x), int(y))] = True

    for x, y in zip(set_bx, set_by):
        if (y > 150):
            dy=y-149
            y=y-dy
        coords_b[(int(x), int(y))] = True

    h = metrics.hausdorff_distance(coords_a, coords_b)

    return h

def editRPETxt(RPE_str):

    split = str.split(RPE_str[1:-2])
    #print("RPE SHAPE:",np.array(split).shape[0])
    RPE_frame=np.zeros(np.array(split).shape[0])
    for i in range(np.array(split).shape[0]):
        RPE_frame[i]=float(split[i])

    return RPE_frame



def editVideoFrame(frame,dim,mnx,mxx,mny,mxy):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    res_frame = cv2.resize(gray_frame, (512, 1024), interpolation=cv2.INTER_LINEAR)
    res_frame=res_frame[145:,:]
    res_frame = cv2.resize(res_frame, dim, interpolation=cv2.INTER_LINEAR)
    print("dimensions of resframe",res_frame.shape)
    crop_frame=res_frame[mnx:mxx,mny:mxy]
    print("dimensions of cropframe", crop_frame.shape)

    return crop_frame

def myDrawPoints(pointList,Image):
    x_range = pointList
    # print("RPE_CHECKED SHAPE",RPE_checked.shape[1])
    y_range = np.arange(Image.shape[1])

    fit = np.polyfit(y_range, x_range, 5)
    lspace = np.linspace(0, Image.shape[1], Image.shape[1])

    draw_x = lspace
    draw_y = np.polyval(fit, draw_x)  # evaluate the polynomial

    draw_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32)  # needs to be int32 and transposed

    cv2.polylines(Image, [draw_points], False, (255, 0, 0), 1)  # args: image, points, closed, color

    return Image