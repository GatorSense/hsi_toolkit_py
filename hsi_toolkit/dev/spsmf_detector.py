from hsi_toolkit.util.img_det import img_det
import numpy as np

def spsmf_detector(hsi_img, tgt_sig, mask = None, mu = None, sig_inv = None):
	"""
	Subpixel Spectral Matched Filter
	 matched filter derived from a subpixel mixing model
	 H0: x = b
	 H1: x = alpha*s + beta*b
	 Ref: formulation from Eismann's book, pp 664

	Inputs:
	 hsi_image - n_row x n_col x n_band hyperspectral image
	 tgt_sig - target signature (n_band x 1 - column vector)
	 mask - binary image limiting detector operation to pixels where mask is true
	        if not present or empty, no mask restrictions are used
	 mu - background mean (n_band x 1 column vector)
	 siginv - background inverse covariance (n_band x n_band matrix)

	Outputs:
	 spsmf_out - detector image

	8/8/2012 - Taylor C. Glenn
	6/2/2018 - Edited by Alina Zare
	12/2018 - Python Implementation by Yutai Zhou
	"""
	if tgt_sig.ndim == 1:
		tgt_sig = tgt_sig[:, np.newaxis]

	spsmf_out, kwargsout = img_det(spsmf_helper, hsi_img, tgt_sig, mask, mu = mu, sig_inv = sig_inv)

	return spsmf_out

def spsmf_helper(hsi_data, tgt_sig, kwargs):
	mu = np.mean(hsi_data, axis = 1) if kwargs['mu'] is None else kwargs['mu']
	mu = mu[:, np.newaxis]
	sig_inv = np.linalg.pinv(np.cov(hsi_data.T, rowvar = False)) if kwargs['sig_inv'] is None else kwargs['sig_inv']

	n_band, n_pixel = hsi_data.shape
	s = tgt_sig # 72 x 1
	st_sig_inv = s.T @ sig_inv # 1 x 72
	st_sig_inv_s = s.T @ sig_inv @ s # 1 x 1
	K = n_band

	spsmf_data = np.zeros(n_pixel)

	for i in range(n_pixel):
		x = hsi_data[:,i][:, np.newaxis] # 72x1
		st_sig_inv_x = st_sig_inv @ x # 1 x 1
		a0 = (x.T @ sig_inv @ x) * st_sig_inv_s - st_sig_inv_x ** 2
		a1 = st_sig_inv_x * (s.T @ sig_inv @ mu) - st_sig_inv_s * (mu.T @ sig_inv @ x)
		a2 = -K * st_sig_inv_s

		beta = (-a1 + np.sqrt(a1 ** 2 - 4 * a2 * a0)) / (2 * a2)
		alpha = st_sig_inv @ (x - beta * mu) / st_sig_inv_s
		z1 = x - mu
		z2 = x - alpha * s - beta * mu

		spsmf_data[i] = z1.T @ sig_inv @ z1 - (z2.T @ sig_inv @ z2) / (beta ** 2) - 2 * K * np.log(np.abs(beta))

	return spsmf_data, {}
