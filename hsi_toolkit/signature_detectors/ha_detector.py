from hsi_toolkit.util.img_det import img_det
from hsi_toolkit.util.unmix import unmix
import numpy as np
from sklearn.mixture import GaussianMixture

def ha_detector(hsi_img, tgt_sig, ems, mask = None, n_comp = 2):
	"""
	Hybrid Abundance Detector

	Inputs:
	 hsi_image - n_row x n_col x n_band hyperspectral image
	 tgt_sig - target signature (n_band x 1 - column vector)
	 mask - binary image limiting detector operation to pixels where mask is true
	        if not present or empty, no mask restrictions are used
	 ems - background endmembers
	 n_comp - number of mixture components for abundance mixtures

	Outputs:
	 ha_out - detector image

	8/27/2012 - Taylor C. Glenn
	6/2/2018 - Edited by Alina Zare
	12/2018 - Python Implementation by Yutai Zhou
	"""
	if tgt_sig.ndim == 1:
		tgt_sig = tgt_sig[:, np.newaxis]

	ha_out, kwargsout = img_det(ha_helper, hsi_img, tgt_sig, mask, ems = ems, n_comp = n_comp)
	return ha_out

def ha_helper(hsi_data, tgt_sig, kwargs):
	ems = kwargs['ems']
	n_comp = kwargs['n_comp']

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

	hs_data = np.zeros(n_pixel)

	for i in range(n_pixel):
		z = hsi_data[:,i] - ems @ P[i,:].T
		w = hsi_data[:,i] - np.hstack((tgt_sig, ems)) @ targ_P[i,:].T

		hs_data[i] = z[np.newaxis,:] @ z[:,np.newaxis] / (w[np.newaxis,:] @ w[:,np.newaxis])

	ha_data = hs_data + ll_tgt - ll_bg
	return ha_data, {}
