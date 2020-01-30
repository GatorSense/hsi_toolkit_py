import sys
from pca import pca
import numpy as np

def csd_anomaly(hsi_img, n_dim_bg, n_dim_tgt, tgt_orth):
	"""
	Complementary Subspace Detector
	 assumes background and target are complementary subspaces
	 of PCA variance ranked space
	Ref: A. Schaum, "Joint subspace detection of hyperspectral targets," 2004 IEEE Aerospace Conference Proceedings (IEEE Cat. No.04TH8720), 2004, pp. 1824 Vol.3. doi: 10.1109/AERO.2004.1367963

	inputs:
	  hsi_image - n_row x n_col x n_band
	  n_dim_bg - number of leading dimensions to assign to background subspace
	  n_dim_tgt - number of dimensions to assign to target subspace
	              use empty matrix, [], to use all remaining after background assignment
	  tgt_orth - True/False, set target subspace orthogonal to background subspace

	8/7/2012 - Taylor C. Glenn
	5/5/2018 - Edited by Alina Zare
	11/2018 - Python Implementation by Yutai Zhou
	"""
	n_row, n_col, n_band = hsi_img.shape
	n_pixel = n_row * n_col

	hsi_data = hsi_img.reshape((n_pixel, n_band), order='F').T

	# PCA rotation, no reduction
	pca_data, _, evecs, evals, _ = pca(hsi_data, 1)

	# whiten the data so that later steps are equivalent to Mahalanobis distance
	z = np.diag(1 / np.sqrt(evals)) @ pca_data

	# figure out background and target subspaces
	bg_rg = np.array(range(0,n_dim_bg))

	if tgt_orth:
		# set target to orthogonal complement of background
		if n_dim_tgt is None:
			n_dim_tgt = n_band - n_dim_bg
		tgt_rg = np.array(range(n_dim_bg, n_dim_tgt))
	else:
		# target and background overlap
		if n_dim_tgt is None:
			n_dim_tgt = n_band
		tgt_rg = np.array(range(0, n_dim_tgt))

	# set background and target subspaces
	B = evecs[:, bg_rg]
	S = evecs[:, tgt_rg]

	# run the detector
	csd_data = np.zeros(n_pixel)

	for i in range(n_pixel):
		Sz = S.T @ z[:,i]
		Bz = B.T @ z[:,i]

		csd_data[i] = Sz.T @ Sz - Bz.T @ Bz

	csd_out = csd_data.reshape(n_row, n_col, order = 'F')

	return csd_out
