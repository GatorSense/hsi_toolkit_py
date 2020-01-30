from hsi_toolkit.util.rx_det import rx_det
import numpy as np

def ace_local_detector(hsi_img, tgt_sig, mask = None, guard_win = 2, bg_win = 4, beta = 0):
	"""
	Adaptive Cosine/Coherence Estimator with RX style local background estimation

	Inputs:
		hsi_image - n_row x n_col x n_band hyperspectral image
		tgt_sig - target signature (n_band x 1 - column vector)
		mask - binary image limiting detector operation to pixels where mask is true
	           if not present or empty, no mask restrictions are used
		guard_win - guard window radius (square,symmetric about pixel of interest)
		bg_win - background window radius
		beta - scalar value used to diagonal load covariance

	Outputs:
		out - detector image

	10/25/2012 - Taylor C. Glenn
	6/2/2018 - Edited by Alina Zare
	10/2018 - Python Implementation by Yutai Zhou
	"""
	n_row, n_col, n_band = hsi_img.shape
	mask = np.ones([n_row, n_col]) if mask is None else mask
	reg = beta * np.eye(n_band)

	if tgt_sig.ndim == 1:
		tgt_sig = tgt_sig[:, np.newaxis]


	out, kwargsout = rx_det(ace_local_helper, hsi_img, tgt_sig, mask = mask, guard_win = guard_win, bg_win = bg_win, reg = reg)
	return out, kwargsout

def ace_local_helper(x, ind, bg, b_mask_list, args, kwargs):
	if bg is None:
		sig_inv = args['global_sig_inv']
		mu = args['mu']
	else:
		sig_inv = np.linalg.pinv(np.cov(bg.T, rowvar = False) + kwargs['reg'])
		mu = np.mean(bg, 1)

	z = x - mu
	s = args['tgt_sig'] - np.tile(mu, (1, args['n_sig'])).T

	sig_out = np.zeros(args['n_sig'])

	for k in range(args['n_sig']):
		st_sig_inv = s[:,k].T @ sig_inv
		st_sig_inv_s = s[:,k].T @ sig_inv @ s[:,k]
		sig_out[k] = ((st_sig_inv @ z) ** 2) / (st_sig_inv_s * (z.T @ sig_inv @ z))

	sig_index = np.argmax(sig_out, 0)

	return sig_out[sig_index], {'sig_index': sig_index}
