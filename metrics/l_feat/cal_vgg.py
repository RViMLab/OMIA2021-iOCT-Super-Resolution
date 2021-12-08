from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from os import walk
import os, sys, getopt
import pickle as pickle
import scipy
from keras.models import Model

# Read VGG16 model
model = VGG16(weights='imagenet', include_top=False)
# redefine model to output right after the first hidden layer
ixs = [2, 5, 9, 13, 17,18]
'''17 is used'''
model = Model(inputs=model.inputs, outputs=model.layers[ixs[4]].output)
model.summary()

def calc_vgg(path_images):

    # path for where to read the images
    input_path =path_images
    type=path_images[-7:-1]

    output_path=path_images+'../vgg_features_'+type

    vgg16_feature_list=[]

    #input_path = 'C:/PhD/Registration/DATA_REG/RIDE_'+RIDE+'_REG/Reg_450_300_cube/' + TYPES_LIST[j]
    for (dirpath, dirnames,filenames) in walk(input_path):
        for i, fname in enumerate(filenames):

            # Load images
            flnm=os.path.join(input_path + "/" + fname)
            img = image.load_img(flnm, target_size=(224, 224))
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)

            vgg16_feature = model.predict(img_data)
            vgg16_feature_np = np.array(vgg16_feature)
            vgg16_feature_list.append(vgg16_feature_np.flatten())

        vgg16_feature_list_np = np.array(vgg16_feature_list)
        print(vgg16_feature_list_np.shape)
        vgg16_feature_list_means = vgg16_feature_list_np.mean(axis=0)

        pickled_db_path=output_path

        with open(pickled_db_path, 'wb') as fp:
            #general: vgg16_feature_list_means isntead of vgg16_feature_np
            pickle.dump(vgg16_feature_list_means, fp)
        fp.close()


    return output_path

