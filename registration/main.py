import numpy as np
import os
from registration import applyRegistration,applyExtraRegistration
from validate_registration import validateReg
from data_augmentation import getFrames
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-t',  '--task', type=str, help='Set task')
    parser.add_argument('-fp', '--fixed_path', type=str, help='Set path for fixed images')
    parser.add_argument('-mp', '--moving_path', type=str, help='Set path for moving images')
    parser.add_argument('-sp', '--saved_path', type=str, help='Set path for saved (results of registration) images')
    parser.add_argument('-v', '--vid_file', type=str, help='Set video for data augmentation')

    args = parser.parse_args()

    if args.task:
        task=args.task

    if args.fixed_path:
        path_to_fixed_folder=args.fixed_path

    if args.moving_path:
        path_to_moving_folder=args.moving_path

    if args.saved_path:
        path_to_moved_folder=args.saved_path

    if args.vid_file:
        videoname=args.vid_file


    # path_to_fixed_folder = '/fixed/'   # folder that contains the fixed images during the registration
    # path_to_moving_folder = '/moving/' # folder that contains the moving images during the registration
    # path_to_moved_folder = '/moved/'   # folder in which the moved images, results of the registration, will be saved

    dim = (450, 300)                   # image dimensions

    ## 1) Non-Rigid registration between Dataset 1 and
    ## Dataset 2 to create partially aligned pairs.
    if(task=='registration'):
        applyRegistration(path_to_moved_folder,path_to_fixed_folder,path_to_moving_folder,dim)

    ## 2) Validation of the registration.
    if (task=='validation'):
         validateReg(path_to_moved_folder, path_to_fixed_folder, dim)

    ## 3) Data Augmentation
    #videoname='video_i' # video that will be processed for data augmentation
    if (task=='augmentation'):
        getFrames(videoname,path_to_moved_folder)

    ## 4) Registration for the augmented frames:
    if(task=='reg_augmentation'):
        apply_rigid=True
        applyExtraRegistration(path_to_moved_folder, path_to_fixed_folder, path_to_moving_folder,apply_rigid)




