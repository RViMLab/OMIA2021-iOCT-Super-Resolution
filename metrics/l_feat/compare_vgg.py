from keras.preprocessing import image
#from scipy.misc import imread
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from os import walk
import os
import pickle as pickle
import scipy
from keras.models import Model
from scipy import spatial


def compare_data_to_data(calculated_vgg_path1,calculated_vgg_path2):

    # Read calculated features from dataset 1
    fp = open(calculated_vgg_path1, 'rb')
    data1 = pickle.load(fp)
    calc_features1 = np.array(data1)
    print(calc_features1.shape)
    fp.close()

    # Read calculated features from dataset 2
    fp = open(calculated_vgg_path2, 'rb')
    data2 = pickle.load(fp)
    calc_features2 = np.array(data2)
    print(calc_features2.shape)
    fp.close()

    calc_features1 = calc_features1.reshape(1, -1)

    calc_features2 = calc_features2.reshape(1, -1)

    feature_dist = spatial.distance.cdist(calc_features1, calc_features2, 'euclidean')
    return feature_dist

def compare_image_to_image(image1, image2):
    img_data1 = image.img_to_array(image1)
    img_data1 = np.expand_dims(img_data1, axis=0)
    img_data1 = preprocess_input(img_data1)

    vgg16_feature1 = model.predict(img_data1)
    vgg16_feature_np1 = np.array(vgg16_feature1)
    image_features1 = vgg16_feature_np1.flatten()


    img_data2 = image.img_to_array(image2)
    img_data2 = np.expand_dims(img_data2, axis=0)
    img_data2 = preprocess_input(img_data2)

    vgg16_feature2 = model.predict(img_data2)
    vgg16_feature_np2 = np.array(vgg16_feature2)
    image_features2 = vgg16_feature_np2.flatten()


    image_features1 = image_features1.reshape(1, -1)

    image_features2 = image_features2.reshape(1, -1)

    feature_dist = spatial.distance.cdist(image_features1, image_features2, 'euclidean')
    return feature_dist


#Define if Image to Image or Dataset to Dataset
COMPARE_IMAGE_TO_IMAGE=False
COMPARE_DATA_TO_DATA=True

# if COMPARE_DATA_TO_DATA==True
# Define the folders containing the calculated averaged feature maps
calculated_vgg_path1='C:\PhD\SR\SR_averaged_paper_results/vgg_features_hr_averaged_equal_hist'
calculated_vgg_path2='C:\PhD\SR\SR_averaged_paper_results/vgg_features_snn_averaged'

# if COMPARE_IMAGE_TO_IMAGE==True
# Define the folders containing the images

input_path = 'C:/PhD/SR/CYCLEGAN/results/iOCT3D/Quantitative_Data'
Directory_ref = "/CUBE/"
Directory_1 = "/SR_VIDEO_CycleGAN/"


# Read VGG16 model
model = VGG16(weights='imagenet', include_top=False)
# redefine model to output right after the first hidden layer
ixs = [2, 5, 9, 13, 17,18]
'''17 is used'''
model = Model(inputs=model.inputs, outputs=model.layers[ixs[4]].output)
model.summary()

def compare(source_stats,ref_stats,image_to_image=True):


    COMPARE_IMAGE_TO_IMAGE = image_to_image

    if(COMPARE_IMAGE_TO_IMAGE):
        print("Processing image to image: ")

        for (dirpath, dirnames, filenames) in walk(input_path+Directory_1):
            for i, fname in enumerate(filenames):

                flnm_ref = os.path.join(input_path + Directory_ref + fname)
                img_ref = image.load_img(flnm_ref, target_size=(224, 224))

                flnm1 = os.path.join(input_path + Directory_1 + fname)
                img1 = image.load_img(flnm1, target_size=(224, 224))

                distance = compare_image_to_image(img1, img_ref)
                print("Distance between Image1 and Image2",distance)

    elif(COMPARE_DATA_TO_DATA):

        distance = compare_data_to_data(source_stats, ref_stats)
        print("VGG Distance Between Dataset 1 and Dataset 2 is: ",distance[0][0])

    return distance
