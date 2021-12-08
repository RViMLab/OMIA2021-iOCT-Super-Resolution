import os
from PIL import Image
#import torch
#import scipy.misc
import numpy as np
from metrics import getMetrics
from skimage.exposure import match_histograms
import pathlib
import cv2
import imageio

restults_dir='C:\PhD\SR\CYCLEGAN_PIX2PIX_TORCH\pytorch-CycleGAN-and-pix2pix/results/'
name_dir='oct_cyclegan'
phase_dir='val'



if __name__ == '__main__':

    latest = False  # todo define if we want only the latest saved model or all
    min_epochs = 5  # todo define the minimum checkpoints we have
    max_epochs = 30  # todo define the maximum checkpoints we have
    epoch = []


    if (latest):
        epoch.append('latest')

    else:
        num_of_saved_chkpoints = (max_epochs-min_epochs) / 5
        for i in range(min_epochs, max_epochs, 5):
            epoch.append(str(i))

    output_dir = restults_dir + name_dir
    with open(output_dir + '/evaluation_results.txt', 'w') as f:

        for k in range(len(epoch)):
            epoch_dir = epoch[k]
            # Read folder with all images
            path_to_results_epoch=pathlib.Path(restults_dir+name_dir+'/'+phase_dir +'_'+epoch_dir+'/images/')

            norm_real_A_images = []
            norm_real_B_images = []
            norm_fake_B_images = []

            path_to_norm_real_A=restults_dir+name_dir+'/'+phase_dir +'_'+epoch_dir+'/norm_real_A'
            path_to_norm_real_B=restults_dir+name_dir+'/'+phase_dir +'_'+epoch_dir+'/norm_real_B'
            path_to_norm_fake_B=restults_dir+name_dir+'/'+phase_dir +'_'+epoch_dir+'/norm_fake_B'

            if not os.path.exists(path_to_norm_real_A):
                os.makedirs(path_to_norm_real_A)
            if not os.path.exists(path_to_norm_real_B):
                os.makedirs(path_to_norm_real_B)
            if not os.path.exists(path_to_norm_fake_B):
                os.makedirs(path_to_norm_fake_B)

            for i in path_to_results_epoch.glob('*real_B.png'):
                img = cv2.imread(str(i), 0)
                norm_real_B_images.append(img)
                imageio.imwrite(path_to_norm_real_B+'/'+str(i.stem)+'.png', img)

            counter_fkb=0 # this counter helps to find the corresponding image from real_b so we can normalize the intensities for real_a
            for i in path_to_results_epoch.glob('*real_A.png'):
                img = cv2.imread(str(i), 0)
                ref=norm_real_B_images[counter_fkb]
                matched = match_histograms(img, ref, multichannel=True)
                imageio.imwrite(path_to_norm_real_A+'/'+str(i.stem)+'.png', matched)
                counter_fkb=counter_fkb+1


            counter_fkb=0 # this counter helps to find the corresponding image from real_b so we can normalize the intensities for fake_b
            for i in path_to_results_epoch.glob('*fake_B.png'):
                img = cv2.imread(str(i), 0)
                ref=norm_real_B_images[counter_fkb]
                matched = match_histograms(img, ref, multichannel=True)
                imageio.imwrite(path_to_norm_fake_B+'/'+str(i.stem)+'.png', matched)
                counter_fkb=counter_fkb+1

            ### Calculate metrics, low-resolution - high resolution, we pass paths as inputs
            print(path_to_norm_real_A)
            PSNR_l, SSIM_l, l_feat_l, FID_l, GCF_l, NIQE_l = getMetrics(path_to_norm_real_A+'/', path_to_norm_real_B+'/')
            print( PSNR_l, SSIM_l, l_feat_l, FID_l,GCF_l)
            ### Calculate metrics, super-resolution - high resolution, we pass paths as inputs
            PSNR_sr, SSIM_sr, l_feat_sr, FID_sr, GCF_sr, NIQE_sr = getMetrics(path_to_norm_fake_B+'/', path_to_norm_real_B+'/')
            print(PSNR_sr, SSIM_sr, l_feat_sr, FID_sr,GCF_sr)

            f.write('Epoch: %s SR_PSNR: %f SR_SSIM: %f SR_l_feat: %f SR_FID: %f SR_GCF: %f LR_PSNR: %f LR_SSIM: %f LR_l_feat: %f LR_FID: %f LR_GCF: %f  \n'
                    % (epoch[k], PSNR_sr, SSIM_sr, l_feat_sr, FID_sr, GCF_sr, PSNR_l, SSIM_l, l_feat_l, FID_l, GCF_l))



        # for i in path_to_results_epoch.glob('*real_A.png'):
        # for i in path_to_results_epoch.glob('*real_B.png'):

            #for real_A_file in i.glob('*Angiography*3x3*.avi'):
                # for label, im_data in visuals.items():
                #     im = util.tensor2im(im_data)
                #     image_name = '%s_%s.png' % (name, label)
                #     save_path = os.path.join(image_dir, image_name)
                #     util.save_image(im, save_path, aspect_ratio=aspect_ratio)