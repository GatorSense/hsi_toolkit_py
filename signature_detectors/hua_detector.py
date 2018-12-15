import sys
sys.path.append('../util/')
from img_det import img_det
from unmix import unmix
import numpy as np
from sklearn.mixture import GaussianMixture

def hua_detector(hsi_img, tgt_sig, ems, mask = None, n_comp = 2, sig_inv = None):
	"""
	Hybrid Unstructured Abundance Detector

	 Ref:
	 Hybrid Detectors for Subpixel Targets
	 Broadwater, J. and Chellappa, R.
	 Pattern Analysis and Machine Intelligence, IEEE Transactions on
	 2007 Volume 29 Number 11 Pages 1891 -1903 Month nov.

	Inputs:
	 hsi_image - n_row x n_col x n_band hyperspectral image
	 tgt_sig - target signature (n_band x 1 - column vector)
	 mask - binary image limiting detector operation to pixels where mask is true
	        if not present or empty, no mask restrictions are used
	 ems - background endmembers
	 siginv - background inverse covariance (n_band x n_band matrix)

	Outputs:
	 hua_out - detector image

	8/19/2012 - Taylor C. Glenn
	6/3/2018 - Edited by Alina Zare
	12/2018 - Python Implementation by Yutai Zhou
	"""
	hua_out, kwargsout = img_det(hua_helper, hsi_img, tgt_sig, mask, ems = ems, n_comp = n_comp, sig_inv = sig_inv)
	return hua_out

def hua_helper(hsi_data, tgt_sig, kwargs):
	ems = kwargs['ems']
	n_comp = kwargs['n_comp']
	sig_inv = np.linalg.pinv(np.cov(hsi_data.T, rowvar = False)) if kwargs['sig_inv'] is None else kwargs['sig_inv']

	n_pixel = hsi_data.shape[1]

	# unmix data with only background endmembers
	P = unmix(hsi_data, ems)

	# unmix data with target signature as well
	targ_P = unmix(hsi_data, np.hstack((tgt_sig, ems)))

	gmm_bg = GaussianMixture(n_components = n_comp, max_iter = 1, init_params = 'random').fit(P)

	# compute mixture likelihood ratio of each pixel
	n_endmeber = ems.shape[1]
	ll_bg = np.log(np.max(gmm_bg.predict_proba(P),1)) / (n_endmeber * n_comp)
	ll_tgt = targ_P[:,0] > 0.05

	hua_data = np.zeros(n_pixel)

	for i in range(n_pixel):
		s = tgt_sig * targ_P[i,0]
		x = hsi_data[:,i]
		hua_data[i] = x[np.newaxis,:] @ sig_inv @ s / (x[np.newaxis,:] @ sig_inv @ x[:,np.newaxis])

	hud_rg = np.max(hua_data) - np.min(hua_data)
	bg_rg = np.max(ll_bg) - np.min(ll_bg)
	hua_data = hua_data + ll_tgt * (hud_rg/3) -  ll_bg * (hud_rg/(3*bg_rg))

	return hua_data, {'None': None}
