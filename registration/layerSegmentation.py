import numpy as np
import cv2
from sklearn.cluster import KMeans
import sys
import matplotlib.pyplot as plt


def findFirstWhitePixel(th4,ILMpixels):
    print(th4.shape)
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
    print(ILM_fixed)
    labels = np.zeros((1, cols), dtype=np.uint8)
    ILMpixels = np.zeros((1, cols), dtype=np.uint8)

    kmeans = KMeans(n_clusters=2).fit(data_up.reshape(-1, 1))
    labels = kmeans.predict(data_up.reshape(-1, 1))

    if (kmeans.cluster_centers_[1] < kmeans.cluster_centers_[0]):
        labels = 1 - labels


    return labels

def fitPolionym(ILMpixelsTest,labels,RPEpixels):
    x_range=[]
    y_range=[]
    #x_range = ILMpixelsTest
    # y_range = np.arange(cols)
    cols=np.array(ILMpixelsTest).shape[0]
    print(labels[0])
    for i in range(cols):
        if (labels[i]==0):
            x_range.append(ILMpixelsTest[i])
            y_range.append(i)


    fit = np.polyfit(y_range, x_range, 3)
    lspace = np.linspace(0, cols, cols)

    draw_x = lspace
    draw_y = np.polyval(fit, draw_x)  # evaluate the polynomial

    draw_ILM = (np.asarray([draw_x, draw_y]).T).astype(np.int32)
    draw_RPE = (np.asarray([draw_x, RPEpixels]).T).astype(np.int32)

    return draw_ILM,draw_RPE

def compareRPE(ILMpixelsTest,RPEpixels):
    print(RPEpixels[0])
    cols = np.array(ILMpixelsTest).shape[0]
    labels=[]

    for i in range(cols):
        d=abs(ILMpixelsTest[i]-RPEpixels[i])
        if (d<5):
            labels.append(False)
        else:
            labels.append(True)

    return labels

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx


def findUpperWhitePixel(img):
    u8 = img.astype(np.uint8)

    blur = cv2.GaussianBlur(u8, (9, 9), 0)
    ret, th3 = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY)
    white_pixels_init = np.array(np.where(th3[:, :] == 255))

    return th3

def findUp(img,saveImages,RPEpixels):
    u8 = img.astype(np.uint8)
    blur = cv2.GaussianBlur(u8, (5, 5), 0)


    #sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=11)
    #sobely=sobely.astype(np.uint8)
    #clean = cv2.fastNlMeansDenoising(blur)
    #blur = cv2.GaussianBlur(blur1, (15, 15), 0)
    ret3, th3 = cv2.threshold(blur, 35,255,  cv2.THRESH_BINARY)
    blur2 = cv2.GaussianBlur(th3, (11, 11), 0)
    opening = cv2.morphologyEx(blur2, cv2.MORPH_OPEN, (15,15))
    sobely = cv2.Sobel(opening, cv2.CV_64F, 0, 1, ksize=3)
    sobely = 255*(sobely - sobely.min()) / (sobely.max() - sobely.min())
    np.set_printoptions(linewidth=np.inf)
    np.set_printoptions(threshold=sys.maxsize)
    ret, th4 = cv2.threshold(sobely, 220, 255, cv2.THRESH_BINARY)

    #contours, hierarchy = cv2.findContours(np.uint8(th4 ), cv2.RETR_LIST  , cv2.CHAIN_APPROX_SIMPLE)
    #print(contours)
    #canny1 = cv2.Canny(np.uint8(th4 ), 100, 200)
    #cv2.drawContours(u8, sobely, 0, (255, 255, 0), 5)
    movingImagergb = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    ILMpixelsTest=[]

    # First Iteration
    findFirstWhitePixel(th4, ILMpixelsTest)
    # Second Iteration
    labels=useKmeans(ILMpixelsTest)
    #labels=compareRPE(ILMpixelsTest,RPEpixels)
    # Second Order Polyonym
    #draw_ILM,draw_RPE=fitPolionym(ILMpixelsTest,labels,RPEpixels)

    #cv2.polylines(movingImagergb, [draw_ILM], False, (255, 0, 0),1)  # args: image, points, closed, color

    cols = np.array(RPEpixels).shape[0]


    #cv2.polylines(movingImagergb, [draw_RPE], False, (0, 255, 0),1)

    res_movingImagergb = cv2.resize(movingImagergb, (380,150), interpolation=cv2.INTER_LINEAR)

    #saveImages.append(res_movingImagergb)

    cols=np.array(ILMpixelsTest).shape[0]
    sum = 0
    count = 0
    sumup = 0
    countup = 0
    ILMpixels = np.zeros((1, cols), dtype=np.uint8)
    print("cols in findup",th4.shape)
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
    # print("dist_to_RPE", dist_to_RPE)
    if (dist_to_RPE < 50):
        val, idx = find_nearest(ILMpixelsTest, (sumup / countup))

    maxIndex = val
    lmax = idx
    margin2 = 10
    submax = 0

    ILMpixels[0, lmax] = maxIndex

    movingImagergb = movingImagergb.copy()
    dest_or = th4+blur
    findFirstWhitePixel(dest_or, ILMpixelsTest)
    ret5, th5 = cv2.threshold(dest_or, 35, 255, cv2.THRESH_BINARY)

    goLeft(lmax, maxIndex, blur, ILMpixels, ILMpixelsTest)
    maxIndex = val
    goRight(lmax, maxIndex, blur, ILMpixels, ILMpixelsTest)
    #cols=np.array(ILMpixelsTest).shape[0]
    print(ILMpixels)
    for l in range(0,cols):
        movingImagergb[ILMpixels[0][l], l,0] = 0
        movingImagergb[ILMpixels[0][l], l,1] = 255
        movingImagergb[ILMpixels[0][l], l,2] = 0


    res_movingImagergb = cv2.resize(movingImagergb, (380, 150), interpolation=cv2.INTER_LINEAR)

    #saveImages.append(res_movingImagergb)


    '''sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=11)
    soby = cv2.cvtColor(sobely, cv2.COLOR_BGR2GRAY)

    ret3, th3 = cv2.threshold(soby, 0, 255, cv2.cv2.THRESH_OTSU)
    #ret, thresh1 = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    ret4, th4 = cv2.threshold(blur, 0, 255, cv2.cv2.THRESH_OTSU)

    edge4 = cv2.Canny(soby, 60, 180)

    contours, hierarchy = cv2.findContours(th3,  cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (150, 150, 160), 3)'''
    #contours, hierarchy = cv2.findContours(th3,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(th3, contours, -1, (150, 9, 160), 3)

    #plt.subplot(121), plt.imshow(th3, cmap='gray')
    #plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    #plt.subplot(122), plt.imshow(clean, cmap='gray')
    #plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    #plt.show()

    '''ILMpixels=[]
    ILMBool=[]
    shape=th3.shape

    #(used)white_pixels_init = np.array(np.where(th3[:,15] == 255))
    white_pixels_init = np.array(np.where(th3[:,:] == 255))

    first_white_pixel = white_pixels_init[0,0]
    ILMpixels.append(first_white_pixel)
    ILMBool.append(True)'''

    margin=5

    #cx,cy=centerMass(th3)
    #plt.subplot(121), plt.imshow(th3, cmap='gray')
    #plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    #plt.scatter(cx, cy, s=160, c='C0', marker='+')
    #plt.show()

    '''for i in range(1,shape[1]):
        if(margin>first_white_pixel):
            white_pixels = np.array(np.where(th3[first_white_pixel:first_white_pixel + margin, i] == 255))
        else:
            white_pixels = np.array(np.where(th3[first_white_pixel-margin:first_white_pixel+margin,i] == 255))
        shapeWhite=white_pixels.shape

        if (shapeWhite[1]==0):
            first_white_pixel=first_white_pixel
        else:
            if (margin > first_white_pixel):
                first_white_pixel = white_pixels[0, 0] + first_white_pixel
            else:
                first_white_pixel = white_pixels[0,0] +first_white_pixel-margin

        ILMpixels.append(first_white_pixel)'''

    '''for i in range(1,shape[1]):
        if(margin>first_white_pixel):
            white_pixels = np.array(np.where(th3[:, i] == 255))
        else:
            white_pixels = np.array(np.where(th3[:,i] == 255))
        shapeWhite=white_pixels.shape

        if (shapeWhite[1]==0):
            first_white_pixel=first_white_pixel
        else:
            if (margin > first_white_pixel):
                first_white_pixel = white_pixels[0, 0]
            else:
                first_white_pixel = white_pixels[0,0]

        ILMpixels.append(first_white_pixel)'''

    return ILMpixels,th3


def goLeftUp(lmax,maxIndex,movingImage2,ILMpixels,ILM_fixed):
    blur = cv2.GaussianBlur(movingImage2, (5, 5), 0)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=11)
    movingImage = sobely


    for k in range(lmax, 0, -1):
        margin2 = 5
        if (margin2 > maxIndex):
            margin2 = maxIndex

        submax = np.amax(movingImage[maxIndex - margin2:maxIndex + margin2, k])
        # if (abs(ILM_fixed[k] - maxIndex) < margin2):
        # maxIndex = ILM_fixed[k]
        if (submax > 50):
            maxIndex = np.argmax(movingImage[maxIndex - margin2:maxIndex + margin2, k]) + maxIndex - margin2
        else:
            maxIndex = maxIndex
        ILMpixels[0, k] = maxIndex

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

def goRightUp(lmax,maxIndex,movingImage2,ILMpixels,ILM_fixed):
    #blur = cv2.GaussianBlur(movingImage2, (5, 5), 0)
    #sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=11)
    #movingImage=sobely


    cols=movingImage2.shape[1]

    for k in range(lmax, cols):
        margin2 = 5
        #if (margin2 > maxIndex):
            #margin2 = maxIndex
        nonZeroIndices=np.nonzero(movingImage2[maxIndex - margin2:maxIndex + margin2, k])
        print(np.array(nonZeroIndices).shape[1])
        if(np.array(nonZeroIndices).shape[1]==0):
            maxIndex=maxIndex
        else:
            maxIndex=nonZeroIndices[0][0]+ maxIndex - margin2
        #submax = np.amax(movingImage2[maxIndex - margin2:maxIndex + margin2, k])
        #if (submax > 50):
           # maxIndex = np.argmax(movingImage2[maxIndex - margin2:maxIndex + margin2, k]) + maxIndex - margin2
        #else:
            #maxIndex = maxIndex

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

def plotSubSegment(movingImagergb,ILMpixels):

    cols=np.array(ILMpixels).shape[0]
    for l in range(cols):
        movingImagergb[ILMpixels[l], l, 0] = 0
        movingImagergb[ILMpixels[l], l, 1] = 255
        movingImagergb[ILMpixels[l], l, 2] = 0


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

    print('data for kmeans',(data.reshape(-1, 1)).shape)
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
            print("ena class")

            return RPE_fixed, 0, 0

        if (abs(kmeans.cluster_centers_[1] - kmeans.cluster_centers_[0])<10):
            print("MIKRI DIAFORA METAXY TWN CLUSTERS")
            return RPE_fixed, 0, 0

        if (abs(tempcenter1) < 20):
            print("konta sto anw orio")
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

    # WRITE
    #KMEAN_path=os.path.join("./../DATA_REG/",'RIDE_006_REG',"Registered")
    #kmean_im = os.path.join(KMEAN_path, "RPE/RefRPE",filename + '.jpg')
    #cv2.imwrite(kmean_im, movingImagergb)

    # Draw the started point
    #movingImagergb[RPE_fixed[idx], idx, 0] = 250
    #movingImagergb[RPE_fixed[idx], idx, 1] = 0
    #movingImagergb[RPE_fixed[idx], idx, 2] = 0

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

    #temp = movingImagergb[:, mn:mx, :]


    # plot_image = np.zeros([maxrows,maxcols, 300, 3], dtype=np.uint8)
    # plot_image.fill(255)  # or img[:] = 255
    # for l in range(mn,mx):
    #     plot_image[RPEpixels[0][l], l, 0] = 0
    #     plot_image[RPEpixels[0][l], l, 1] = 0
    #     plot_image[RPEpixels[0][l], l, 2] = 255
    #
    # plt.imshow(plot_image)
    # plt.show()
    nonZero = np.nonzero(helpRPEpixels[0])

    if(np.array(nonZero).shape[1]>0):
        mnn = nonZero[0][0]
        mxx = nonZero[0][-1]
    else:
        mnn = 0
        mxx = 0

    return RPEpixels,mxx,mnn