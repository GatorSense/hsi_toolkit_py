import sys
sys.path.append('../util/')
from img_det import img_det
from unmix import unmix
import numpy as np
def abd_detector(hsi_img, tgt_sig, ems, mask = None):
	"""
	Abundance of Target when unmixed with background endmembers

	Inputs:
	 hsi_image - n_row x n_col x n_band hyperspectral image
	 tgt_sig - target signature (n_band x 1 - column vector)
	 mask - binary image limiting detector operation to pixels where mask is true
	        if not present or empty, no mask restrictions are used
	 ems - background endmembers

	Outputs:
	 abd_out - detector image

	8/19/2012 - Taylor C. Glenn
	6/2/2018 - Edited by Alina Zare
	12/2018 - Python Implementation by Yutai Zhou
	"""
	abd_out, kwargsout = img_det(abd_helper,hsi_img,tgt_sig,mask,ems = ems);

	return abd_out

def abd_helper(hsi_data, tgt_sig, kwargs):
	ems = kwargs['ems']
	# unmix data with target signature and background
	targ_P = unmix(hsi_data, np.hstack((tgt_sig, ems)))

	abd_data = targ_P[:,0]

	return abd_data, {'None': None}
