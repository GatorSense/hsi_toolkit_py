import sys
sys.path.append('../util/')
from img_det import img_det
import numpy as np

def ace_detector(hsi_img, tgt_sig, mask = None, mu = None, sig_inv = None):
	"""
	Squared Adaptive Cosine/Coherence Estimator

	Inputs:
		hsi_image - n_row x n_col x n_band hyperspectral image
		tgt_sig - target signature (n_band x 1 - column vector)
		mask - binary image limiting detector operation to pixels where mask is true
		       if not present or empty, no mask restrictions are used
		mu - background mean (n_band x 1 column vector)
		sig_inv - background inverse covariance (n_band x n_band matrix)

	Outputs:
		ace_out - detector image
		mu - mean of input data
		sig_inv - inverse covariance of input data

	 8/8/2012 - Taylor C. Glenn
	 6/2/2018 - Edited by Alina Zare
	 10/2018 - Python Implementation by Yutai Zhou
	 """
	ace_out, kwargsout = img_det(ace_det_helper, hsi_img, tgt_sig, mask, mu = mu, sig_inv = sig_inv)
	return ace_out, kwargsout['mu'], kwargsout['sig_inv']

def ace_det_helper(hsi_data, tgt_sig, kwargs):
	mu = np.mean(hsi_data, axis = 1) if kwargs['mu'] is None else kwargs['mu']
	sig_inv = np.linalg.pinv(np.cov(hsi_data.T, rowvar = False)) if kwargs['sig_inv'] is None else kwargs['sig_inv']

	mu = np.reshape(mu, (len(mu), 1), order='F')
	s = tgt_sig - mu
	z = hsi_data - mu

	st_sig_inv = s.T @ sig_inv
	st_sig_inv_s = s.T @ sig_inv @ s

	A = np.sum(st_sig_inv @ z, 0)
	B = st_sig_inv_s
	C = np.sum(z * (sig_inv @ z), 0)

	ace_data = A * A / (B * C)

	return ace_data, {'mu':mu, 'sig_inv': sig_inv}
