import sys
from scipy.io import loadmat
import matplotlib.pyplot as plt
sys.path.append('../')
sys.path.append('../util/')
from signature_detectors import *
from get_RGB import get_RGB

"""
Demo script that runs all spectral indices in hsi_toolkit_py and visualizes the results. 

Inputs:
	hsi_sub - n_row x n_col x n_band hyperspectral image
	tgt_spectra - n_band x 1 target signature vector
	wavelengths - n_band x 1 vector listing wavelength values for hsi_sub in nm
	gt_img_sub - n_row x n_col ground truths
	mask - binary image limiting detector operation to pixels where mask is true
	 	if not present or empty, no mask restrictions are used
Outputs:
	det_out - dictionary of RGB image and spectral indices outputs

02/2020 - Susan Meerdink
"""

# Load data
an_hsi_img_for_tgt_det_demo = loadmat('an_hsi_img_for_tgt_det_demo.mat')
hsi_sub = an_hsi_img_for_tgt_det_demo['hsi_sub']
tgt_spectra = an_hsi_img_for_tgt_det_demo['tgt_spectra']
wavelengths = an_hsi_img_for_tgt_det_demo['wavelengths']
gt_img_sub = an_hsi_img_for_tgt_det_demo['gtImg_sub']

# Set up True Color or Red/Green/Blue Image
det_out = {}
det_out['RGB'] = get_RGB(hsi_sub, wavelengths)

# Call Spectral Indices
ndvi_out = ndvi_vi(hsi_sub, wavelengths)
det_out['NDVI'] = ndvi_out
evi_out = evi_vi(hsi_sub, wavelengths)
det_out['EVI'] = evi_out

# Visualization
# plt.figure(figsize=(10,15))



