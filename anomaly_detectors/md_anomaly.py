import sys
sys.path.append('../util/')
from img_det import img_det
import numpy as np

def md_anomaly(hsi_img, mask = None):
	"""
	Mahalanobis Distance anomaly detector
	uses global image mean and covariance as background estimates

	Inputs:
	 hsi_image - n_row x n_col x n_band hyperspectral image
	 mask - binary image limiting detector operation to pixels where mask is true
	        if not present or empty, no mask restrictions are used

	Outputs:
	  dist_img - detector output image

	8/7/2012 - Taylor C. Glenn
	5/5/2018 - Edited by Alina Zare
	11/2018 - Python Implementation by Yutai Zhou
	"""
	dist_img, kwargsout = img_det(md_helper, hsi_img, None, mask)
	return dist_img

def md_helper(hsi_data, tgt_sig, kwargs):
	n_pixel = hsi_data.shape[1]

	mu = np.mean(hsi_data, 1)
	sigma = np.cov(hsi_data.T, rowvar = False)

	z = hsi_data - mu[:,np.newaxis]
	sig_inv = np.linalg.pinv(sigma)

	dist_data = np.zeros(n_pixel)
	for i in range(n_pixel):
		dist_data[i] = z[:,i].T @ sig_inv @ z[:,i]
	return dist_data[:,np.newaxis], {'None', None}
