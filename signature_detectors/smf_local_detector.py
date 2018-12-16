import sys
sys.path.append('../util/')
from rx_det import rx_det
import numpy as np

def smf_local_detector(hsi_img, tgt_sig, mask = None, guard_win = 2, bg_win = 4):
	"""
	Spectral Matched Filter with RX style local background estimation

	Inputs:
	 hsi_image - n_row x n_col x n_band hyperspectral image
	 tgt_sig - target signature (n_band x 1 - column vector)
	 mask - binary image limiting detector operation to pixels where mask is true
	        if not present or empty, no mask restrictions are used
	 guard_win - guard window radius (square,symmetric about pixel of interest)
	 bg_win - background window radius

	Outputs:
	 out - detector image

	10/25/2012 - Taylor C. Glenn
	10/2018 - Python Implementation by Yutai Zhou
	"""
	out, kwargsout = rx_det(smf_local_helper, hsi_img, tgt_sig, mask = mask, guard_win = guard_win, bg_win = bg_win)
	return out

def smf_local_helper(x, ind, bg, b_mask_list, args, kwargs):
	if bg is None:
		sig_inv = args['global_sig_inv']
		mu = args['mu']
	else:
		sig_inv = np.linalg.pinv(np.cov(bg.T, rowvar = False))
		mu = np.mean(bg, 1)

	s = args['tgt_sig'] - np.reshape(mu, (-1,1))
	z = np.reshape(x, (-1,1)) - np.reshape(mu, (-1,1))
	f = (s.T @ sig_inv) / np.sqrt(s.T @ sig_inv @ s)

	smf_data = f @ z
	return smf_data, {}
