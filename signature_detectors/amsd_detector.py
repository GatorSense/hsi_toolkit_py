import sys
sys.path.append('../util/')
from img_det import img_det
import numpy as np
def amsd_detector(hsi_img, tgt_sig, mask = None, n_dim_tgt = 1, n_dim_bg = 5):
	"""
	Adaptive Matched Subspace Detector

	 Reference:
	 Hyperspectral subpixel target detection using the linear mixing model (article)
	 Manolakis, D. and Siracusa, C. and Shaw, G.
	 Geoscience and Remote Sensing, IEEE Transactions on
	 2001 Volume 39 Number 7 Pages 1392 -1409 Month jul

	Inputs:
	 hsi_image - n_row x n_col x n_band hyperspectral image
	 tgt_sigs - target signature(s) (n_band x n_sig - column vectors)
	 mask - binary image limiting detector operation to pixels where mask is true
	        if not present or empty, no mask restrictions are used
	 n_dim_tgt - number of dimensions to use for target subspace,
	             if argument is 0, use the target sigs themselves
	 n_dim_bg - number of dimensions to use for background subspace

	Outputs:
	 amsd_out - detector image

	8/22/2012 - Taylor C. Glenn
	6/02/2018 - Edited by Alina Zare
	12/2018 - Python Implementation by Yutai Zhou
	"""
	amsd_out, kwargsout = img_det(amsd_helper, hsi_img, tgt_sig, mask, n_dim_tgt = n_dim_tgt, n_dim_bg = n_dim_bg)
	return amsd_out

def amsd_helper(hsi_data, tgt_sig, kwargs):
	n_dim_tgt = kwargs['n_dim_tgt']
	n_dim_bg = kwargs['n_dim_bg']

	n_band, n_pixel = hsi_data.shape
	n_sigs = tgt_sig.shape[1]

	# find target and background subspace
	corr_bg = np.zeros((n_band, n_band))
	for i in range(n_pixel):
		corr_bg = corr_bg + hsi_data[:,i] @ hsi_data[:,i]

	corr_bg = corr_bg / n_pixel
