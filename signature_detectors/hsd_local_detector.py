import sys
sys.path.append('../util/')
from rx_det import rx_det
from unmix import unmix
import numpy as np

def hsd_local_detector(hsi_img, tgt_sig, ems, mask = None, guard_win = 2, bg_win = 4, beta = 0):
	"""
	Hybrid Subpixel Detector with RX style local background estimation

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

	1/25/2013 - Taylor C. Glenn
	6/3/2018 - Edited by Alina Zare
	12/2018 - Python Implementation by Yutai Zhou
	"""
	n_row, n_col, n_band = hsi_img.shape
	hsi_data = hsi_img.reshape((n_row * n_col, n_band), order='F').T

	reg = beta * np.eye(n_band)

	# unmix data with only background endmembers
	P = unmix(hsi_data, ems)

	# unmix data with target signature as well
	targ_P = unmix(hsi_data, np.hstack((tgt_sig, ems)))

	out, kwargsout = rx_det(hsd_local_helper, hsi_img, tgt_sig, mask, guard_win, bg_win, ems = ems, reg = reg, P = P, targ_P = targ_P)
	return out

def hsd_local_helper(x, ind, bg, b_mask_list, args, kwargs):
	if bg is not None:
		sigma = np.cov(bg.T, rowvar=False)
		sig_inv = np.linalg.pinv(sigma + kwargs['reg'])
	else:
		sig_inv = args['global_siginv']

	z = x - kwargs['ems'] @ kwargs['P'][ind,:]
	w = x - np.hstack((args['tgt_sig'], kwargs['ems'])) @ kwargs['targ_P'][ind,:]
	r = (z[np.newaxis,:] @ sig_inv @ z[:,np.newaxis]) / (w[np.newaxis,:] @ sig_inv @ w[:,np.newaxis])

	return r, {}
