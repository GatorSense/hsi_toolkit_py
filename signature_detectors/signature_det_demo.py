import sys
sys.path.append('../util/')
from scipy.io import loadmat
import matplotlib.pyplot as plt
from ace_detector import ace_detector
from ace_rx_detector import ace_rx_detector
from smf_detector import smf_detector
from get_RGB import get_RGB
"""
Demo that runs all signature detectors in the hsi_toolkit

Inputs:
	img - n_row x n_col x n_band hyperspectral image
	tgt_sig - n_band x 1 target signature vector
	mask - binary image limiting detector operation to pixels where mask is true
	     if not present or empty, no mask restrictions are used
	wavelengths - 1 x n_band vector listing wavelength values for hsi_img in nm

Outputs:
	det_out - cell array of detector outputs

6/2/2018 - Alina Zare
10/12/2018 - Python Implementation by Yutai Zhou
"""
# Load data
an_hsi_img_for_tgt_det_demo = loadmat('an_hsi_img_for_tgt_det_demo.mat')
hsi_sub = an_hsi_img_for_tgt_det_demo['hsi_sub']
tgt_spectra = an_hsi_img_for_tgt_det_demo['tgt_spectra']
wavelengths = an_hsi_img_for_tgt_det_demo['wavelengths']
gt_img_sub = an_hsi_img_for_tgt_det_demo['gtImg_sub']

# init detector args
guard_win = 1; bg_win = 3; beta = 0.001

# call detectors
ace_out, _, _ = ace_detector(hsi_sub, tgt_spectra)
ace_rx_out, _ = ace_rx_detector(hsi_sub, tgt_spectra, guard_win = guard_win, bg_win = bg_win, beta = beta)
smf_out, _, _ = smf_detector(hsi_sub, tgt_spectra)

# visualization
plt.figure(figsize=(10,10))
plt.subplots_adjust(hspace=.5)

n_row = 4; n_col = 3
plt.subplot(n_row, n_col,1);
plt.imshow(get_RGB(hsi_sub, wavelengths)); plt.title('RGB')
plt.subplot(n_row, n_col,2);
plt.imshow(gt_img_sub); plt.title('Ground Truth')

plt.subplot(n_row, n_col,3);
plt.imshow(ace_out); plt.title('ACE Squared')
plt.subplot(n_row, n_col,4);
plt.imshow(ace_rx_out); plt.title('ACE RX Squared')
plt.subplot(n_row, n_col,5);
plt.imshow(smf_out); plt.title('Spectral Matched Filter', fontsize = 12)
plt.show()
