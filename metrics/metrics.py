from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import numpy as np
#from util import util
import os
import cv2
from cleanfid import fid
from l_feat.cal_vgg import calc_vgg
from l_feat.compare_vgg import compare
from Global_Contrast_Factor import compute_global_contrast_factor



def getPSNR(source_images,reference_images):

    num_images=len(source_images) # get number of images in the list
    psnr_list=[]
    for i in range(0, num_images):
        ref_img=reference_images[i]
        src_img=source_images[i]
        psnr_list.append(peak_signal_noise_ratio(ref_img,src_img)) #first input: true image, second input: test image

    mean_PSNR=np.mean(psnr_list)
    return mean_PSNR

def getSSIM(source_images,reference_images):

    num_images=len(source_images) # get number of images in the list
    ssim_list=[]
    for i in range(0, num_images):
        img_1=reference_images[i]
        img_2=source_images[i]
        ssim_list.append(structural_similarity(img_1,img_2,multichannel=True)) # order of input images does not matter (obviously)

    mean_SSIM=np.mean(ssim_list)
    return mean_SSIM

def getFID(source_images,reference_images):
    score = fid.compute_fid(source_images, reference_images, mode="clean",num_workers=0)

    return score

def getVGG(path_source_images,path_reference_images):

    source_stats=calc_vgg(path_source_images)
    ref_stats=calc_vgg(path_reference_images)

    score=compare(source_stats,ref_stats,image_to_image=False)

    return score

def getGCF(source_images):
    num_images = len(source_images)  # get number of images in the list
    gcf_list = []
    for i in range(0, num_images):
        img= source_images[i]
        gcf_list.append(
            compute_global_contrast_factor(img))

    mean_GCF = np.mean(gcf_list)
    return mean_GCF

def path_to_list(path_imgs):
    list_imgs=[]
    for fname in os.listdir(path_imgs):
        src_image = path_imgs + fname
        img = cv2.imread(src_image, 0)
        list_imgs.append(img)

    return list_imgs



def getMetrics(path_source_images,path_reference_images):
    # input: all inputs paths of images
    # outuput: float numbers of every metric (6 metrics)

    source_images=path_to_list(path_source_images)
    reference_images=path_to_list(path_reference_images)

    # get PSNR
    PSNR=getPSNR(source_images,reference_images)
    #print(PSNR)
    # get SSIM
    SSIM=getSSIM(source_images,reference_images)
    #print(SSIM)
    # get VGG
    VGG=getVGG(path_source_images,path_reference_images)

    # get FID
    FID=getFID(path_source_images,path_reference_images)
    #print(FID)

    # get GCF
    GCF=getGCF(source_images)

    return PSNR, SSIM, VGG, FID, GCF, 0






