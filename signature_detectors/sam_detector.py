import sys
sys.path.append('../util/')
from img_det import img_det
import numpy as np

def sam_detector(hsi_img, tgt_sig, mask = None):
	"""
	Spectral Angle Mapper

	Inputs:
	 hsi_image - n_row x n_col x n_band hyperspectral image
	 tgt_sig - target signature (n_band x 1 - column vector)
	 mask - binary image limiting detector operation to pixels where mask is true
	        if not present or empty, no mask restrictions are used

	Outputs:
	 sam_out - detector image

	8/8/2012 - Taylor C. Glenn
	6/2/2018 - Edited by Alina Zare
	12/2018 - Python Implementation by Yutai Zhou
	"""
	sam_out, kwargsout = img_det(sam_helper, hsi_img, tgt_sig, mask)

	return sam_out

def sam_helper(hsi_data, tgt_sig, kwargs):
	prod = tgt_sig.T @ hsi_data
	mag = np.sqrt(tgt_sig.T @ tgt_sig * np.sum(hsi_data ** 2, 0))
	sam_data = prod / mag
	return sam_data.squeeze(), {}
