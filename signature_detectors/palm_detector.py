import sys
sys.path.append('../util/')
from img_det import img_det
import numpy as np
from sklearn.mixture import GaussianMixture

def palm_detector(hsi_img, tgt_sig, mask = None, n_comp = 5):
	"""
	Pairwise Adaptive Linear Matched Filter

	Inputs:
	 hsi_image - n_row x n_col x n_band hyperspectral image
	 tgt_sig - target signature (n_band x 1 - column vector)
	 mask - binary image limiting detector operation to pixels where mask is true
	        if not present or empty, no mask restrictions are used
	 n_comp - number of Gaussian components to use

	Outputs:
	 palm_out - detector image

	8/8/2012 - Taylor C. Glenn
	6/2/2018 - Edited by Alina Zare
	12/2018 - Python Implementation by Yutai Zhou
	"""
	palm_out, kwargsout = img_det(palm_helper, hsi_img, tgt_sig, mask, n_comp = n_comp)
	return palm_out

def palm_helper(hsi_data, tgt_sig, kwargs):
	n_comp = kwargs['n_comp']
	n_pixel = hsi_data.shape[1]

	# fit the model
	gmm = GaussianMixture(n_components = n_comp, max_iter = 1, init_params = 'random').fit(hsi_data.T)
	means = gmm.means_
	covariances = gmm.covariances_
	sig_inv = np.zeros(covariances.shape)
	filt = np.zeros(means.shape)

	for i in range(n_comp):
		sig_inv[i,:,:] = np.linalg.pinv(covariances[i,:,:])

		s = tgt_sig - means[i,:][:,np.newaxis]
		filt[i,:] = s.T @ sig_inv[i,:,:] / np.sqrt(s.T @ sig_inv[i,:,:] @ s)
	# print(filt.shape)
	# print(tgt_sig.shape, means.shape, covariances.shape, sig_inv.shape)

	palm_data = np.zeros(n_pixel)

	for i in range(n_pixel):
		dists = np.zeros(n_comp)
		for j in range(n_comp):
			dists[j] = (filt[j,:] @ (hsi_data[:,i] - means[j,:])) ** 2

		palm_data[i] = np.min(dists)

	return palm_data, {'None': None}
