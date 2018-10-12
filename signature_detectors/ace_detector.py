import numpy as np
def ace_detector(hsi_img, tgt_sig, mask, mu, siginv):
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
	 return ace_out, mu, siginv
