from scipy.io import loadmat
import matplotlib.pyplot as plt
import sys
sys.path.append('../util/')
from rx_anomaly import rx_anomaly
from get_RGB import get_RGB
"""
Demo that runs all anomaly detectors in the hsi_toolkit using example sub MUUFL Gulfport data

inputs:
	hsi_image - n_row x n_col x n_band hyperspectral image
	mask - binary image limiting detector operation to pixels where mask is true
		   if not present or empty, no mask restrictions are used
	wavelengths - 1 x n_band vector listing wavelength values for hsi_img in nm

outputs:
	det_out - cell array of detector output images

5/5/2018 - Alina Zare
10/1/2018 - Python Implementation by Yutai Zhou
"""
an_hsi_image_sub_for_demo = loadmat('../datasets/an_hsi_image_sub_for_demo.mat')
hsi_img_sub = an_hsi_image_sub_for_demo['hsi_img_sub']
mask_sub = an_hsi_image_sub_for_demo['mask_sub']
wavelengths = an_hsi_image_sub_for_demo['wavelengths']

guard_win = 3; bg_win = 3
rx_out = rx_anomaly(hsi_img_sub, guard_win, bg_win, mask_sub)

# print(mask_sub.shape, hsi_img_sub.shape, wavelengths.shape)
n_row = 4; n_col = 3
plt.subplot(n_row, n_col,1);
plt.imshow(get_RGB(hsi_img_sub, wavelengths), extent=[0, 1, 0, 1]); plt.title('RGB Image')
plt.subplot(n_row, n_col,2);
plt.imshow(mask_sub, extent=[0, 1, 0, 1]); plt.title('Valid Mask')
plt.subplot(n_row, n_col,3);
plt.imshow(rx_out, extent=[0, 1, 0, 1]); plt.title('rx anomaly')
plt.show()
