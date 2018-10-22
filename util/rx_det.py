import numpy as np

def rx_det(det_func, hsi_img, tgt_sigs, mask = None, guard_win = None, bg_win = None, **kwargs):
	"""
	Wrapper to make an RX style sliding window detector given the local detection function

	Inputs:
		det_fun - detection function
		hsi_image - n_row x n_col x n_band hyperspectral image
		tgt_sig - target signature (n_band x 1 - column vector)
		mask - binary image limiting detector operation to pixels where mask is true
	           if not present or empty, no mask restrictions are used
		guard_win - guard window radius (square,symmetric about pixel of interest)
		bg_win - background window radius

	Outputs:
		det_out - detector image

	1/27/2013 - Taylor C. Glenn
	10/2018 - Python Implementation by Yutai Zhou
	"""
