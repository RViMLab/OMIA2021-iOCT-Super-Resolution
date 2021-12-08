import numpy as np
import os
from registration import applyRegistration,creatCheckboards,applyExtraRegistration
from validate import validateReg
from augmentPairs import getMoreFrames

np.random.seed(1000)


def main():

    applyReg = False
    checkValidity=False

    findMoreVidFrames=False

    registerExtra=True

    path_to_fixed_folder = '/fixed/'   # folder that contains the fixed images during the registration
    path_to_moving_folder = '/moving/' # folder that contains the moving images during the registration
    path_to_moved_folder = '/moved/'   # folder that contains the moved images, results of the registration

    dim = (450, 300)

    ## 1) We apply Non-Rigid registration between Dataset 1 and
    ## Dataset 2 to create partially aligned pairs.

    if(applyReg):
        applyRegistration(path_to_moved_folder,path_to_fixed_folder,path_to_moving_folder,dim)

    ## 2) We validate the registration of the previous step.
    ## We check

    if (checkValidity):
         validateReg(path_to_moved_folder, path_to_fixed_folder, dim)

    ## 3) Thirdly, we augment our dataset

    # 3) Find more video frames to increase the image pairs
    if (findMoreVidFrames):
        getMoreFrames(videoname,target_path, path_root)



    #  Register the extra Video Frames:
    if(registerExtra):
        # RIDE FOLDERS EXTRA PAIRS
        apply_rigid=False
        #applyExtraRegistration(target_path, path_fixed_folder, path_moving_folder,apply_rigid dim=[390, 150])

        # VIDEO REGISTRATION
        # Fixed Path
        path_fixed_folder = 'C:/PhD/SR/video_registration/fixed'
        # Moving Path
        path_moving_folder = 'C:/PhD/SR/video_registration/moving'

        # Moved Path
        array_ids = [ '026', '027', '028', '029', '030']
        RIDE_id='066'

        for i in range(0,np.array(array_ids).shape[0]):
            RIDE='RIDE_'+array_ids[i]+'_iOCT'

            ## DOWN
            updown='DOWN'
            VIDEO='Retina_OCT.mpg'

            target_path = "C:/PhD/DATA/" + RIDE + "/vid_registration/" + VIDEO + '/'+updown+'/Registered'
            if not os.path.exists(str(target_path)):
                os.makedirs(str(target_path))

            #target_path = 'C:/PhD/SR/video_registration/moved'
            path_fixed_folder="C:/PhD/DATA/" + RIDE + "/vid_registration/" + VIDEO + '/'+updown+'/'
            path_moving_folder="C:/PhD/DATA/" + RIDE + "/vid_registration/" + VIDEO + '/'+updown+'/'
            print(path_moving_folder)
            apply_rigid=True

            applyExtraRegistration(target_path, path_fixed_folder, path_moving_folder,apply_rigid, dim=[306,451])

            ## UP
            updown = 'UP'

            target_path = "C:/PhD/DATA/" + RIDE + "/vid_registration/" + VIDEO + '/' + updown + '/Registered'
            if not os.path.exists(str(target_path)):
                os.makedirs(str(target_path))

            # target_path = 'C:/PhD/SR/video_registration/moved'
            path_fixed_folder = "C:/PhD/DATA/" + RIDE + "/vid_registration/" + VIDEO + '/' + updown + '/'
            path_moving_folder = "C:/PhD/DATA/" + RIDE + "/vid_registration/" + VIDEO + '/' + updown + '/'
            print(path_moving_folder)
            apply_rigid = True

            applyExtraRegistration(target_path, path_fixed_folder, path_moving_folder, apply_rigid, dim=[306, 451])


if __name__ == '__main__':
	main()
