from hsi_toolkit.util import img_det
import numpy as np
from sklearn.mixture import GaussianMixture

def ccmf_detector(hsi_img, tgt_sig, mask = None, n_comp = 5, gmm = None):
	"""
	Class Conditional Matched Filters

	inputs:
	 hsi_image - n_row x n_col x n_band hyperspectral image
	 tgt_sig - target signature (n_band x 1 - column vector)
	 mask - binary image limiting detector operation to pixels where mask is true
	        if not present or empty, no mask restrictions are used
	 n_comp - number of Gaussian components to use
	 gmm - optional mixture model structure from previous training data

	outputs:
	 ccmf_out - detector image
	 gmm - mixture model learned from input image

	8/8/2012 - Taylor C. Glenn
	6/2/2018 - Edited by Alina Zare
	12/2018 - Python Implementation by Yutai Zhou
	"""
	if tgt_sig.ndim == 1:
		tgt_sig = tgt_sig[:, np.newaxis]

	ccmf_out, kwargsout = img_det(ccmf_helper, hsi_img, tgt_sig, mask, n_comp = n_comp, gmm = gmm)

	return ccmf_out, kwargsout['gmm']

def ccmf_helper(hsi_data, tgt_sig, kwargs):
	n_comp = kwargs['n_comp']
	gmm = kwargs['gmm']

	if gmm is None:
		gmm = GaussianMixture(n_components = n_comp, max_iter = 1, init_params = 'random').fit(hsi_data.T)

	n_pixel = hsi_data.shape[1]

	# make a matched filter for each background class
	means = gmm.means_
	covariances = gmm.covariances_
	sig_inv = np.zeros(covariances.shape)
	filt = np.zeros(means.shape)

	for i in range(n_comp):
		sig_inv[i,:,:] = np.linalg.pinv(covariances[i,:,:])

		s = tgt_sig - means[i,:][:,np.newaxis]
		filt[i,:] = s.T @ sig_inv[i,:,:] / np.sqrt(s.T @ sig_inv[i,:,:] @ s)

	# run the appropriate filter for class of each pixels
	idx = gmm.predict(hsi_data.T)

	ccmf_data = np.zeros(n_pixel)
	for i in range(n_pixel):
		ccmf_data[i] = filt[idx[i],:] @ (hsi_data[:,i] - means[idx[i],:])

	return ccmf_data, {'gmm': gmm}
