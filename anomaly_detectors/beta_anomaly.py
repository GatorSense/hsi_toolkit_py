import sys
sys.path.append('../util/')
from img_det import img_det
import numpy as np
from scipy.stats import beta
from math import log

def beta_anomaly(hsi_img, mask):
	"""
	Beta Distribution Anomaly Detector
	 fits beta distribution to each band assuming entire image is background
	 computes negative log likelihood of each pixel in the model

	Inputs:
	 hsi_image - n_row x n_col x n_band hyperspectral image
	 mask - binary image limiting detector operation to pixels where mask is true
	        if not present or empty, no mask restrictions are used

	Outputs:
	  beta_out - detector output image

	8/24/2012 - Taylor C. Glenn
	5/5/2018 - Edited by Alina Zare
	11/2018 - Python Implementation by Yutai Zhou
	"""
	beta_out, kwargsout = img_det(beta_helper, hsi_img, None, mask)
	return beta_out

def beta_helper(hsi_data, tgt_sig, kwargs):
	n_band, n_pixel = hsi_data.shape
	hsi_data[hsi_data <= 0] = 1e-6
	hsi_data[hsi_data >= 1] = 1 - 1e-6

	alphas = np.zeros(n_band)
	betas = np.zeros(n_band)

	# fit the model
	loc = 0; scale = 1
	for i in range(n_band):
		params = beta.fit(hsi_data[i,:])
		alphas[i] = params[0]
		betas[i] = params[1]

	# compute likelihood of each pixel
	likelihood = np.zeros((n_band, n_pixel))
	for i in range(n_band):
		likelihood[i,:] = beta.logpdf(hsi_data[i,:], alphas[i], betas[i])
	beta_data = - np.sum(likelihood, 0)
	return beta_data[:, np.newaxis], {'None': None}
