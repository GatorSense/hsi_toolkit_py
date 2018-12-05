import sys
sys.path.append('../util/')
from pca import pca
from img_det import img_det
import numpy as np

def osp_detector(hsi_img, tgt_sig, mask = None, n_dim_ss = 2):
	"""
	Orthogonal Subspace Projection Detector

	Inputs:
	 hsi_image - n_row x n_col x n_band hyperspectral image
	 tgt_sig - target signature (n_band x 1 - column vector)
	 mask - binary image limiting detector operation to pixels where mask is true
	        if not present or empty, no mask restrictions are used
	 n_dim_ss - number of dimensions to use in the background subspace

	Outputs:
	 osp_out - detector image

	8/8/2012 - Taylor C. Glenn
	6/2/2018 - Edited by Alina Zare
	11/2018 - Python Implementation by Yutai Zhou
	"""
	osp_out, kwargsout = img_det(osp_helper, hsi_img, tgt_sig, mask, n_dim_ss = n_dim_ss)

	return osp_out

def osp_helper(hsi_data, tgt_sig, kwargs):
	n_dim_ss = kwargs['n_dim_ss']
	# see Eismann, pp670
	n_band, n_pixel = hsi_data.shape
	mu = np.mean(hsi_data, 1)
	mu = mu[:, np.newaxis]
	x = hsi_data - mu

	# get PCA rotation, no dim reduction
	_, _, evecs, _, _ = pca(hsi_data, 1)
	s = tgt_sig - mu

	# get a subspace that theoretically encompasses the background
	B = evecs[:, :n_dim_ss]

	PB = B @ np.linalg.pinv(B.T @ B) @ B.T
	PperpB = np.eye(n_band) - PB

	f = s.T @ PperpB

	osp_data = np.zeros(n_pixel)

	for i in range(n_pixel):
		osp_data[i] = f @ x[:,i]
	return osp_data, {'None', None}
