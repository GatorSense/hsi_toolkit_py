import sys
sys.path.append('../util/')
from img_det import img_det
import numpy as np
from sklearn.mixture import GaussianMixture

def gmrx_anomaly(hsi_img, n_comp, mask = None):
	"""
	Gaussian Mixture RX Anomaly Detector
	 fits GMM assuming entire image is background
	 assigns pixels to highest posterior probability mixture component
	 computes pixel Mahlanobis distance to component mean

	Inputs:
	 hsi_image - n_row x n_col x n_band hyperspectral image
	 mask - binary image limiting detector operation to pixels where mask is true
	        if not present or empty, no mask restrictions are used
	 n_comp - number of Gaussian components to use

	Outputs:
	  gmrx_out - detector output image

	8/7/2012 - Taylor C. Glenn - tcg@cise.ufl.edu
	5/5/2018 - Edited by Alina Zare
	11/2018 - Python Implementation by Yutai Zhou
	"""
	gmm_out, kwargsout = img_det(gmrx_helper, hsi_img, None, mask, n_comp = n_comp)
	return gmm_out

def gmrx_helper(hsi_data, tgt_sig, kwargs):
	n_comp = kwargs['n_comp']
	n_pixel = hsi_data.shape[1]
	gmm = GaussianMixture(n_components = n_comp, max_iter = 1, init_params = 'random').fit(hsi_data.T)
