import sys
from scipy.io import loadmat
import matplotlib.pyplot as plt
sys.path.append('../')
sys.path.append('../util/')
from signature_detectors import *
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

det_out = {}
# call detectors
ace_out, _, _ = ace_detector(hsi_sub, tgt_spectra)
det_out['ACE Squared'] = ace_out
ace_rx_out, _ = ace_rx_detector(hsi_sub, tgt_spectra, guard_win = guard_win, bg_win = bg_win, beta = beta)
det_out['ACE RX Squared'] = ace_rx_out
smf_out, _, _ = smf_detector(hsi_sub, tgt_spectra)
det_out['Spectral Matched Filter'] = smf_out
smf_rx_out    = smf_rx_detector(hsi_sub, tgt_spectra)
det_out['Spectral Matched Filter RX'] = smf_rx_out

# visualization
plt.figure(figsize=(10, 15))
plt.subplots_adjust(hspace=.5)

n_row = 4; n_col = 3
plt.subplot(n_row, n_col,1);
plt.imshow(get_RGB(hsi_sub, wavelengths)); plt.title('RGB')
plt.subplot(n_row, n_col,2);
plt.imshow(gt_img_sub); plt.title('Ground Truth')

i = 3
for key, value in det_out.items():
	plt.subplot(n_row, n_col,i);
	plt.imshow(value); plt.title(key)
	i += 1

plt.show()
