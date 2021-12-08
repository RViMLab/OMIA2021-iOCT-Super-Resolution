import cv2
import numpy as np

def compute_image_average_contrast(k, gamma=2.2):
    L = 100 * np.sqrt((k / 255) ** gamma )
    # pad image with border replicating edge values
    L_pad = np.pad(L,1,mode='edge')

    # compute differences in all directions
    left_diff = L - L_pad[1:-1,:-2]
    right_diff = L - L_pad[1:-1,2:]
    up_diff = L - L_pad[:-2,1:-1]
    down_diff = L - L_pad[2:,1:-1]

    # create matrix with number of valid values 2 in corners, 3 along edges and 4 in the center
    num_valid_vals = 3 * np.ones_like(L)
    num_valid_vals[[0,0,-1,-1],[0,-1,0,-1]] = 2
    num_valid_vals[1:-1,1:-1] = 4

    pixel_avgs = (np.abs(left_diff) + np.abs(right_diff) + np.abs(up_diff) + np.abs(down_diff)) / num_valid_vals

    return np.mean(pixel_avgs)

def compute_global_contrast_factor(img):
    if img.ndim != 2:
        gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gr = img

    superpixel_sizes = [1, 2, 4, 8, 16, 25, 50, 100, 200]

    gcf = 0

    for i,size in enumerate(superpixel_sizes,1):
        wi =(-0.406385 * i / 9 + 0.334573) * i/9 + 0.0877526
        im_scale = cv2.resize(gr, (0,0), fx=1/size, fy=1/size,
                              interpolation=cv2.INTER_LINEAR)
        avg_contrast_scale = compute_image_average_contrast(im_scale)
        gcf += wi * avg_contrast_scale

    return gcf