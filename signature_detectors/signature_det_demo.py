import sys
from scipy.io import loadmat
import matplotlib.pyplot as plt
sys.path.append('../')
sys.path.append('../util/')
from signature_detectors import *
from get_RGB import get_RGB
"""
Demo script that runs all signature detectors in hsi_toolkit_py

Inputs:
	hsi_sub - n_row x n_col x n_band hyperspectral image
	tgt_spectra - n_band x 1 target signature vector
	wavelengths - n_band x 1 vector listing wavelength values for hsi_sub in nm
	gt_img_sub - n_row x n_col ground truths
	mask - binary image limiting detector operation to pixels where mask is true
	 	if not present or empty, no mask restrictions are used
Outputs:
	det_out - dictionary of RGB image, ground truth image, and detector outputs

6/2/2018 - Alina Zare
10/12/2018 - Python Implementation by Yutai Zhou
"""
# Load data
an_hsi_img_for_tgt_det_demo = loadmat('an_hsi_img_for_tgt_det_demo.mat')
hsi_sub = an_hsi_img_for_tgt_det_demo['hsi_sub']
tgt_spectra = an_hsi_img_for_tgt_det_demo['tgt_spectra']
wavelengths = an_hsi_img_for_tgt_det_demo['wavelengths']
gt_img_sub = an_hsi_img_for_tgt_det_demo['gtImg_sub']

det_out = {}
det_out['RGB'] = get_RGB(hsi_sub, wavelengths)
det_out['Ground Truth'] = gt_img_sub

# init detector args
guard_win = 2; bg_win = 4; beta = 0.001

# call detectors
ace_out, _, _ = ace_detector(hsi_sub, tgt_spectra)
det_out['ACE Squared'] = ace_out
ace_rx_out, _ = ace_rx_detector(hsi_sub, tgt_spectra, guard_win = guard_win, bg_win = bg_win, beta = beta)
det_out['ACE RX Squared'] = ace_rx_out
smf_out, _, _ = smf_detector(hsi_sub, tgt_spectra)
det_out['Spectral Matched Filter'] = smf_out
smf_rx_out    = smf_rx_detector(hsi_sub, tgt_spectra, guard_win = guard_win, bg_win = bg_win)
det_out['Spectral Matched Filter RX'] = smf_rx_out

# visualization
plt.figure(figsize=(10, 15))
plt.subplots_adjust(hspace=.5)
n_row = 4; n_col = 3

i = 1
for key, value in det_out.items():
	plt.subplot(n_row, n_col, i);
	plt.imshow(value); plt.title(key)
	i += 1

plt.show()
