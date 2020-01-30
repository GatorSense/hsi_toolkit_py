import sys
from hsi_toolkit.util.img_det import img_det
import numpy as np
from sklearn.mixture import GaussianMixture

def gmm_anomaly(hsi_img, n_comp, mask = None):
	"""
	Gaussian Mixture Model Anomaly Detector
	 fits GMM assuming entire image is background
	 computes negative log likelihood of each pixel in the fit model

	Inputs:
	 hsi_image - n_row x n_col x n_band hyperspectral image
	 mask - binary image limiting detector operation to pixels where mask is true
	        if not present or empty, no mask restrictions are used
	 n_comp - number of Gaussian components to use

	Outputs:
	  gmm_out - detector output image

	8/7/2012 - Taylor C. Glenn
	5/5/2018 - Edited by Alina Zare
	11/2018 - Python Implementation by Yutai Zhou
	"""
	gmm_out, kwargsout = img_det(gmm_helper, hsi_img, None, mask, n_comp = n_comp)
	return gmm_out

def gmm_helper(hsi_data, tgt_sig, kwargs):
	n_comp = kwargs['n_comp']
	gmm = GaussianMixture(n_components = n_comp, max_iter = 1, init_params = 'random').fit(hsi_data.T)
	gmm_data = -gmm.score_samples(hsi_data.T)
	return gmm_data, {}
