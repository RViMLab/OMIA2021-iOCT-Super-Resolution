import numpy as np
import os
from registration import applyRegistration,applyExtraRegistration
from validate_registration import validateReg
from data_augmentation import getMoreFrames

np.random.seed(1000)


def main():

    applyReg = False
    checkValidity=False

    dataAugmentation=False
    register_augmented_frames=False

    path_to_fixed_folder = '/fixed/'   # folder that contains the fixed images during the registration
    path_to_moving_folder = '/moving/' # folder that contains the moving images during the registration
    path_to_moved_folder = '/moved/'   # folder that contains the moved images, results of the registration

    dim = (450, 300)

    ## 1) We apply Non-Rigid registration between Dataset 1 and
    ## Dataset 2 to create partially aligned pairs.
    if(applyReg):
        applyRegistration(path_to_moved_folder,path_to_fixed_folder,path_to_moving_folder,dim)

    ## 2) We validate the registration of the previous step.
    if (checkValidity):
         validateReg(path_to_moved_folder, path_to_fixed_folder, dim)

    ## 3) Data Augmentation
    videoname='video_i' # video that will be processed for data augmentation
    if (dataAugmentation):
        getMoreFrames(videoname,path_to_moved_folder)

    ## 4)  Registration for the augmented frames:
    if(register_augmented_frames):
        apply_rigid=True
        applyExtraRegistration(path_to_moved_folder, path_to_fixed_folder, path_to_moving_folder,apply_rigid)


if __name__ == '__main__':
	main()
