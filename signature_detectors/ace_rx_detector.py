import sys
sys.path.append('../util/')
from rx_det import rx_det
import numpy as np

def ace_rx_detector(hsi_img, tgt_sig, mask = None, guard_win = None, bg_win = None, beta = 0):
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
