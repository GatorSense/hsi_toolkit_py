import sys
from pca import pca
import numpy as np

def ssrx_anomaly(hsi_img, n_dim_ss, guard_win, bg_win):
	"""
	function ssrx_img = ssrx_anomaly(hsi_img,n_dim_ss,guard_win,bg_win)

	Subspace Reed-Xiaoli anomaly detector
	 eliminate leading subspace as background, then
	 use local mean and covariance to determine pixel to background distance

	Inputs:
	  hsi_image - n_row x n_col x n_band
	  n_dim_ss - number of leading dimensions to use in the background subspace
	  guard_win - guard window radius (square,symmetric about pixel of interest)
	  bg_win - background window radius

	8/7/2012 - Taylor C. Glenn
	5/5/2018 - Edited by Alina Zare
	11/2018 - Python Implementation by Yutai Zhou
	"""
	n_row, n_col, n_band = hsi_img.shape
	n_pixels = n_row * n_col
	hsi_data = np.reshape(hsi_img, (n_pixels, n_band), order='F').T

	# PCA with no dim deduction
	pca_data, _, evecs, evals, _ = pca(hsi_data, 1)

	pca_img = np.reshape(pca_data.T, (n_row, n_col, n_band), order='F')
	proj = np.eye(n_band) - evecs[:, :n_dim_ss] @ evecs[:, :n_dim_ss].T
	# Create the mask
	mask_width = 1 + 2 * guard_win + 2 * bg_win
	half_width = guard_win + bg_win
	mask_rg = np.array(range(mask_width)) - 1

	b_mask = np.ones((mask_width, mask_width), dtype=bool)
	b_mask[bg_win:b_mask.shape[0] - bg_win, bg_win:b_mask.shape[0] - bg_win] = 0

	# run the detector (only on fully valid points)
	ssrx_img = np.zeros((n_row, n_col))

	for i in range(n_col - mask_width + 1):
		for j in range(n_row - mask_width + 1):
			row = j + half_width
			col = i + half_width

			b_mask_img = np.zeros((n_row, n_col))
			b_mask_img[j:mask_width + j, i:mask_width + i] = b_mask
			b_mask_list = np.reshape(b_mask_img, -1, order='F')
			# pull out background points
			bg = pca_data[:, b_mask_list == 1]

			# Mahalanobis distance
			covariance = np.cov(bg.T, rowvar=False)

			s = np.float32(np.linalg.svd(covariance, compute_uv=False))
			rcond = np.max(covariance.shape)*np.spacing(np.float32(np.linalg.norm(s, ord=np.inf)))
			# pinv differs from MATLAB
			sig_inv = np.linalg.pinv(covariance)
			mu = np.mean(bg, 1)
			z = proj @ pca_img[row, col, :].squeeze() - proj @ mu
			ssrx_img[row, col] = z.T @ sig_inv @ z

	return ssrx_img
