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
	  siginv - background inverse covariance (n_band x n_band matrix)

	 Outputs:
	  ace_out - detector image
	  mu - mean of input data
	  siginv - inverse covariance of input data

	 8/8/2012 - Taylor C. Glenn
	 6/2/2018 - Edited by Alina Zare
	 """
	 ace_out, mu, sig_inv = img_det(ace_det_helper, hsi_img, tgt_sig, mask, mu, sig_inv)
	 return ace_out, mu, sig_inv

def ace_det_helper(hsi_data, tgt_sig, mu = None, sig_inv = None):
	if mu is None:
		mu = np.mean(hsi_data, axis = 1)
	if sig_inv is None:
		sig_inv = np.linalg.pinv(np.linalg.cov(hsi_data.T, rowvar = False))

	s = tgt_sig - mu
	z = hsi_data - mu

	st_sig_inv = s.T @ sig_inv
	st_sig_inv_s = s.T @ sig_inv @ s

	A = np.sum(st_sig_inv @ z, 0)
	B = st_sig_inv_s
	C = np.sum(z * (sig_inv @ z), 0)

	ace_data = A * A / (B * C)

	return ace_data, mu, sig_inv
