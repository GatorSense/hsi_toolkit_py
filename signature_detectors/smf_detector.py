import sys
sys.path.append('../util/')
from img_det import img_det
import numpy as np

def smf_detector(hsi_img, tgt_sig, mask = None, mu = None, sig_inv = None):
	"""
	Spectral Matched Filter

	Inputs:
	 hsi_image - n_row x n_col x n_band hyperspectral image
	 tgt_sig - target signature (n_band x 1 - column vector)
	 mask - binary image limiting detector operation to pixels where mask is true
	        if not present or empty, no mask restrictions are used
	 mu - (optional) mean for filter (if not provided, computed from image)
	 siginv - (optional) inverse covariance for filter (if not provided, computed from image)

	Outputs:
	 smf_out - detector image
	 mu - mean of background
	 sig_inv - inverse covariance of background

	8/8/2012 - Taylor C. Glenn
	6/2/2018 - Edited by Alina Zare
	10/2018 - Python Implementation by Yutai Zhou
	"""
	smf_out, kwargsout = img_det(smf_det_array_helper, hsi_img, tgt_sig, mask, mu = mu, sig_inv = sig_inv)
	return smf_out, kwargsout['mu'], kwargsout['sig_inv']

def smf_det_array_helper(hsi_data, tgt_sig, kwargs):
	"""
	Spectral Matched Filter for array (not image) data

	Inputs:
	 hsi_data - n_spectra x n_band array of hyperspectral data
	 tgt_sig - target signature (n_band x 1 - column vector)
	 mu - (optional) mean for filter
	 sig_inv - (optional) inverse covariance for filter

	Outputs:
	 smf_data - detector output per spectra
	 mu - mean of background
	 sig_inv - inverse covariance of background

	8/19/2012 - Taylor C. Glenn
	6/2/2018 - Edited by Alina Zare
	10/2018 - Python Implementation by Yutai Zhou

	alternative formulation from Eismann's book, pp 653
	 subtract mean from target to reduce effect of additive model
	 on hyperspectral (non additive) data
	 also take positive square root of filter
	"""
	mu = np.mean(hsi_data, axis = 1) if kwargs['mu'] is None else kwargs['mu']
	sig_inv = np.linalg.pinv(np.cov(hsi_data.T, rowvar = False)) if kwargs['sig_inv'] is None else kwargs['sig_inv']

	s = tgt_sig - np.reshape(mu, (-1,1))
	z = hsi_data - np.reshape(mu, (-1,1))
	f = (s.T @ sig_inv) / np.sqrt(s.T @ sig_inv @ s)

	smf_data = f @ z
	return smf_data, {'mu':mu, 'sig_inv': sig_inv}
