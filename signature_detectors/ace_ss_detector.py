import sys
sys.path.append('../util/')
from img_det import img_det
import numpy as np

def ace_ss_detector(hsi_img, tgt_sig, mask = None, mu = None, sig_inv = None):
	"""
	Adaptive Cosine/Coherence Estimator - Subspace Formulation

	Inputs:
	 hsi_image - n_row x n_col x n_band hyperspectral image
	 tgt_sigs - target signatures (n_band x M - column vector)
	 mask - binary image limiting detector operation to pixels where mask is true
	        if not present or empty, no mask restrictions are used

	Outputs:
	 ace_ss_out - detector image

	8/8/2012 - Taylor C. Glenn
	6/2/2018 - Edited by Alina Zare
	11/2018 - Python Implementation by Yutai Zhou
	"""
	if tgt_sig.ndim == 1:
		tgt_sig = tgt_sig[:, np.newaxis]

	ace_ss_out, kwargsout = img_det(ace_ss_helper, hsi_img, tgt_sig, mask, mu = mu, sig_inv = sig_inv)
	return ace_ss_out

def ace_ss_helper(hsi_data, tgt_sig, kwargs):
	mu = np.mean(hsi_data, axis = 1) if kwargs['mu'] is None else kwargs['mu']
	mu = mu[:, np.newaxis]
	sig_inv = np.linalg.pinv(np.cov(hsi_data.T, rowvar = False)) if kwargs['sig_inv'] is None else kwargs['sig_inv']
	S = tgt_sig - mu
	z = hsi_data - mu

	G = sig_inv @ S @ np.linalg.pinv(S.T @ sig_inv @ S) @ S.T @ sig_inv

	A = np.sum(z * (G @ z),0)
	B = np.sum(z * (sig_inv @ z),0)

	out = A / B

	return out, {}
