import sys
from scipy.io import loadmat
import matplotlib.pyplot as plt
sys.path.append('../')
sys.path.append('../util/')
from hsi_toolkit.anomaly_detectors import *
from get_RGB import get_RGB
"""
Demo script that runs all anomaly detectors in hsi_toolkit_py using example sub MUUFL Gulfport data

Inputs:
	hsi_img_sub - n_row x n_col x n_band hyperspectral image
	wavelengths - n_band x 1 vector listing wavelength values for hsi_img in nm
	mask_sub - n_row x n_col binary image limiting detector operation to pixels where mask is true
		       if not present or empty, no mask restrictions are used
Outputs:
	det_out - dictionary of detector output images

5/5/2018 - Alina Zare
10/1/2018 - Python Implementation by Yutai Zhou
"""
an_hsi_image_sub_for_demo = loadmat('an_hsi_image_sub_for_demo.mat')
hsi_img_sub = an_hsi_image_sub_for_demo['hsi_img_sub']
wavelengths = an_hsi_image_sub_for_demo['wavelengths']
mask_sub = an_hsi_image_sub_for_demo['mask_sub']

det_out = {}
det_out['RGB'] = get_RGB(hsi_img_sub, wavelengths)
det_out['Valid Mask'] = mask_sub

# init detector args
guard_win = 3; bg_win = 3; n_dim_ss = 3

# call detectors
# beta_out = beta_anomaly(hsi_img_sub, mask_sub) <--- moved into dev
# det_out['Beta Anomaly'] = beta_out
cbad_out, _ = cbad_anomaly(hsi_img_sub, 8, mask_sub)
det_out['CBAD Anomaly'] = cbad_out
csd_out = csd_anomaly(hsi_img_sub, 3, None, True)
det_out['CSD Anomaly'] = csd_out
# fcbad_out, _ = fcbad_anomaly(hsi_img_sub, 8, mask_sub)
# det_out['FCBAD Anomaly'] = fcbad_out
gmm_out = gmm_anomaly(hsi_img_sub, n_comp = 8, mask = mask_sub)
det_out['GMM Anomaly'] = gmm_out
# gmrx_out = gmrx_anomaly(hsi_img_sub, n_comp = 8, mask = mask_sub) <--- moved into dev
# det_out['GMRX Anomaly'] = gmrx_out
md_out = md_anomaly(hsi_img_sub, mask_sub)
det_out['MD Anomaly'] = md_out
rx_out = rx_anomaly(hsi_img_sub, guard_win, bg_win, mask_sub)
det_out['RX Anomaly'] = rx_out
# ssrx_out = ssrx_anomaly(hsi_img_sub, n_dim_ss, guard_win, bg_win) <--- moved into dev
# det_out['SSRX Anomaly'] = ssrx_out

# visualization
plt.figure(figsize=(10, 15))
plt.subplots_adjust(hspace=.5)
n_row = 4; n_col = 3

i = 1
for key, value in det_out.items():
	plt.subplot(n_row, n_col, i);
	plt.imshow(value); plt.title(key)
	i += 1
# plt.imshow(gmrx_out)
plt.show()
