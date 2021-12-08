import numpy as np
import cv2
from sklearn.cluster import KMeans
import sys
import matplotlib.pyplot as plt


def findFirstWhitePixel(th4,ILMpixels):
    cols=th4.shape[1]
    first_white_pixel=0
    for l in range(0,cols):
        white_pixels_init = np.array(np.where(th4[:, l] == 255))
        if (white_pixels_init.shape[1] == 0):
            first_white_pixel = first_white_pixel
        else:
            first_white_pixel = white_pixels_init[0, 0]
        ILMpixels.append(first_white_pixel)

def useKmeans(ILM_fixed):
    data_up = np.array(ILM_fixed)
    cols=np.array(ILM_fixed).shape[0]
    labels = np.zeros((1, cols), dtype=np.uint8)
    ILMpixels = np.zeros((1, cols), dtype=np.uint8)

    kmeans = KMeans(n_clusters=2).fit(data_up.reshape(-1, 1))
    labels = kmeans.predict(data_up.reshape(-1, 1))

    if (kmeans.cluster_centers_[1] < kmeans.cluster_centers_[0]):
        labels = 1 - labels


    return labels


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx


def findUp(img,saveImages,RPEpixels):
    u8 = img.astype(np.uint8)
    blur = cv2.GaussianBlur(u8, (5, 5), 0)

    ret3, th3 = cv2.threshold(blur, 35,255,  cv2.THRESH_BINARY)
    blur2 = cv2.GaussianBlur(th3, (11, 11), 0)
    opening = cv2.morphologyEx(blur2, cv2.MORPH_OPEN, (15,15))
    sobely = cv2.Sobel(opening, cv2.CV_64F, 0, 1, ksize=3)
    sobely = 255*(sobely - sobely.min()) / (sobely.max() - sobely.min())
    np.set_printoptions(linewidth=np.inf)
    np.set_printoptions(threshold=sys.maxsize)
    ret, th4 = cv2.threshold(sobely, 220, 255, cv2.THRESH_BINARY)


    movingImagergb = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    ILMpixelsTest=[]

    # First Iteration
    findFirstWhitePixel(th4, ILMpixelsTest)
    # Second Iteration
    labels=useKmeans(ILMpixelsTest)

    cols=np.array(ILMpixelsTest).shape[0]
    sum = 0
    count = 0
    sumup = 0
    countup = 0
    ILMpixels = np.zeros((1, cols), dtype=np.uint8)
    for l in range(cols):
        if (labels[l] == 0):
            movingImagergb[ILMpixelsTest[l], l, 0] = 255
            movingImagergb[ILMpixelsTest[l], l, 1] = 0
            movingImagergb[ILMpixelsTest[l], l, 2] = 0
            sumup = sumup + ILMpixelsTest[l]
            countup = countup + 1

        if (labels[l] == 1):
            movingImagergb[ILMpixelsTest[l], l, 0] = 0
            movingImagergb[ILMpixelsTest[l], l, 1] = 255
            movingImagergb[ILMpixelsTest[l], l, 2] = 0
            lmax = l
            max0 = ILMpixelsTest[l]
            sum = sum + max0
            count = count + 1

    val, idx = find_nearest(ILMpixelsTest, (sumup / countup))

    dist_to_RPE = abs(val - RPEpixels[idx])
    if (dist_to_RPE < 50):
        val, idx = find_nearest(ILMpixelsTest, (sumup / countup))

    maxIndex = val
    lmax = idx

    ILMpixels[0, lmax] = maxIndex

    movingImagergb = movingImagergb.copy()
    dest_or = th4+blur
    findFirstWhitePixel(dest_or, ILMpixelsTest)
    ret5, th5 = cv2.threshold(dest_or, 35, 255, cv2.THRESH_BINARY)

    goLeft(lmax, maxIndex, blur, ILMpixels, ILMpixelsTest)
    maxIndex = val
    goRight(lmax, maxIndex, blur, ILMpixels, ILMpixelsTest)

    for l in range(0,cols):
        movingImagergb[ILMpixels[0][l], l,0] = 0
        movingImagergb[ILMpixels[0][l], l,1] = 255
        movingImagergb[ILMpixels[0][l], l,2] = 0

    return ILMpixels,th3




def goLeft(lmax,maxIndex,movingImage2,ILMpixels,ILM_fixed):
    blur = cv2.GaussianBlur(movingImage2, (5, 5), 0)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=11)
    movingImage = sobely

    for k in range(lmax, 0, -1):
        margin2 = 5
        if(margin2 > maxIndex):
            margin2=maxIndex

        submax = np.amax(movingImage[maxIndex - margin2:maxIndex + margin2, k])
        #if (abs(ILM_fixed[k] - maxIndex) < margin2):
            #maxIndex = ILM_fixed[k]
        if (submax > 50):
            maxIndex = np.argmax(movingImage[maxIndex - margin2:maxIndex + margin2, k]) + maxIndex - margin2
        else:
            maxIndex = maxIndex
        ILMpixels[0, k] = maxIndex



def goRight(lmax,maxIndex,movingImage2,ILMpixels,ILM_fixed):
    blur = cv2.GaussianBlur(movingImage2, (5, 5), 0)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=11)
    movingImage=sobely

    cols=movingImage.shape[1]

    for k in range(lmax, cols):
        margin2 = 5
        if(maxIndex==0):
            ILMpixels[0, k] = maxIndex
        else:
            if(maxIndex<margin2):
                margin2=maxIndex
            submax = np.amax(movingImage[maxIndex - margin2:maxIndex + margin2, k])
            #if (abs(ILM_fixed[k] - maxIndex) < margin2):
                #maxIndex = ILM_fixed[k]
            if (submax > 50):
                maxIndex = np.argmax(movingImage[maxIndex - margin2:maxIndex + margin2, k]) + maxIndex - margin2
            else:
                maxIndex = maxIndex

        ILMpixels[0, k] = maxIndex



def findBottom(imag):

    RPEpixels=[]
    im = cv2.GaussianBlur(imag, (9, 9), 0)

    im=imag
    shape=im.shape
    #The brightest pixel in each column is assigned as an estimate of the retinal pigment epithelium (RPE). (Lazaridis supplem)
    min_index = np.zeros((1, shape[1]))
    maxIndex = np.argmax(im[:, 0])

    RPEpixels.append(maxIndex)

    mincol=0
    maxcol=shape[1]
    for i in range(1,shape[1]):

        # RPE pixel
        maxVal = np.amax(im[:, i])
        if(maxVal>80):
            #maxIndex=np.argmax(im[maxIndex-3:maxIndex+3, i])+maxIndex-3
            maxIndex = np.argmax(im[:, i])
        else:
            maxIndex=0

        RPEpixels.append(maxIndex)

    return RPEpixels

def refineBottom(movingImage,moving):

    movingImagergb = movingImage.astype(np.uint8)

    movingImagergb = cv2.cvtColor(movingImagergb, cv2.COLOR_GRAY2BGR)

    maxrows,maxcols=movingImage.shape

    RPE_fixed = findBottom(movingImage)

    # Crop RPE keep from first non zero to last non zero element:

    nonZeroIndices=np.nonzero(RPE_fixed)
    mn=nonZeroIndices[0][0]
    mx=nonZeroIndices[0][-1]

    if (mn==0 and mx==0):

        return RPE_fixed,0,0


    data = np.array(RPE_fixed[mn:mx])
    if(data.shape[0]<2):
        return RPE_fixed,0,0

    kmeans = KMeans(n_clusters=2).fit(data.reshape(-1, 1))
    labels = kmeans.predict(data.reshape(-1, 1))
    tempcenter1=kmeans.cluster_centers_[1]
    tempcenter0=kmeans.cluster_centers_[0]

    if (kmeans.cluster_centers_[1] < kmeans.cluster_centers_[0]):
        labels = 1 - labels
        tempcenter0=kmeans.cluster_centers_[1]
        tempcenter1=kmeans.cluster_centers_[0]


    if (len(np.where( labels  == 0)[0]) > len(np.where(labels == 1)[0])):
        labels = 1 - labels
        tempcenter0 = kmeans.cluster_centers_[1]
        tempcenter1 = kmeans.cluster_centers_[0]

    if(moving):

        if (kmeans.cluster_centers_.shape[0] < 2):

            return RPE_fixed, 0, 0

        if (abs(kmeans.cluster_centers_[1] - kmeans.cluster_centers_[0])<10):
            return RPE_fixed, 0, 0

        if (abs(tempcenter1) < 20):
            return RPE_fixed, 0, 0

    #rows, cols = np.array(RPE_fixed[mn:mx]).shape
    cols=mx-mn
    max0 = maxcols
    lmax = 0
    sum = 0
    count = 0

    for l in range(mn,mx):
        if (labels[l-mn] == 0):
            movingImagergb[RPE_fixed[l], l, 0] = 250
            movingImagergb[RPE_fixed[l], l, 1] = 0
            movingImagergb[RPE_fixed[l], l, 2] = 250

        if (labels[l-mn] == 1):
            movingImagergb[RPE_fixed[l], l, 0] = 0
            movingImagergb[RPE_fixed[l], l, 1] = 250
            movingImagergb[RPE_fixed[l], l, 2] = 0
            lmax = l
            max0 = RPE_fixed[l]
            sum = sum + max0
            count = count + 1


    val, idx = find_nearest(RPE_fixed, (sum / count))
    maxIndex = val


    plt.imshow(movingImagergb)
    plt.show()

    lmax = idx
    RPEpixels = np.zeros((1, maxcols), dtype=np.uint8)
    helpRPEpixels = np.zeros((1, maxcols), dtype=np.uint8)
    margin2 = 5
    submax = 0

    movingImagergb = movingImagergb.copy()
    for k in range(lmax, mn, -1):

        # RPE pixel
        # maxVal = np.amax(im[maxIndex-20:maxIndex+20, i])
        if (margin2 > maxIndex):
            submax = np.amax(movingImage[maxIndex:maxIndex + margin2, k])
            if (abs(RPE_fixed[k] - maxIndex) < margin2):

                maxIndex = RPE_fixed[k]
            elif (submax > 70):
                maxIndex = np.argmax(movingImage[maxIndex:maxIndex + margin2, k]) + maxIndex
            else:
                maxIndex = maxIndex

        else:
            submax = np.amax(movingImage[maxIndex - margin2:maxIndex + margin2, k])
            if (abs(RPE_fixed[k] - maxIndex) < margin2):

                maxIndex = RPE_fixed[k]
            elif (submax > 70):
                maxIndex = np.argmax(movingImage[maxIndex - margin2:maxIndex + margin2, k]) + maxIndex - margin2
            else:
                maxIndex = maxIndex
        RPEpixels[0, k] = maxIndex
        
        if((k+1<=mx)):
            if(RPEpixels[0, k+1]==maxIndex ):
                helpRPEpixels[0,k]=0
            else:
                helpRPEpixels[0, k] = 1
        else:
            helpRPEpixels[0, k] = 1

    maxIndex = val
    for k in range(lmax, mx):

        # RPE pixel
        # maxVal = np.amax(im[maxIndex-20:maxIndex+20, i])
        if (margin2 > maxIndex):
            msubmax = np.amax(movingImage[maxIndex:maxIndex + margin2, k])
            if (abs(RPE_fixed[k] - maxIndex) < margin2):

                maxIndex = RPE_fixed[k]
            elif (submax > 70):
                maxIndex = np.argmax(movingImage[maxIndex:maxIndex + margin2, k]) + maxIndex
            else:
                maxIndex = maxIndex
        else:
            submax = np.amax(movingImage[maxIndex - margin2:maxIndex + margin2, k])
            if (abs(RPE_fixed[k] - maxIndex) < margin2):
                maxIndex = RPE_fixed[k]
            elif (submax > 70):
                maxIndex = np.argmax(movingImage[maxIndex - margin2:maxIndex + margin2, k]) + maxIndex - margin2
            else:
                maxIndex = maxIndex


        RPEpixels[0, k] = maxIndex

        if(RPEpixels[0, k-1]==maxIndex):
            helpRPEpixels[0,k]=0
        else:
            helpRPEpixels[0, k] = 1


    nonZero = np.nonzero(helpRPEpixels[0])

    if(np.array(nonZero).shape[1]>0):
        mnn = nonZero[0][0]
        mxx = nonZero[0][-1]
    else:
        mnn = 0
        mxx = 0

    return RPEpixels,mxx,mnn