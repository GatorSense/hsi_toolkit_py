import numpy as np
def rx_anomaly(hsi_img, guard_win, bg_win, mask = None):
	"""
	Widowed Reed-Xiaoli anomaly detector
		use local mean and covariance to determine pixel to background distance

	Inputs:
		hsi_image - n_row x n_col x n_band
		mask - binary image limiting detector operation to pixels where mask is true
	           if not present or empty, no mask restrictions are used
		guard_win - guard window radius (square,symmetric about pixel of interest)
		bg_win - background window radius

	8/7/2012 - Taylor C. Glenn - tcg@cise.ufl.edu
	5/5/2018 - Edited by Alina Zare
	10/1/2018 - Python Implementation by Yutai Zhou
	"""
	n_row, n_col, n_band = hsi_img.shape
	n_pixels = n_row * n_col
	hsi_data = np.reshape(hsi_img, (n_pixels, n_band), order='F').T

	# Create the mask
	mask_width = 1 + 2 * guard_win + 2 * bg_win
	b_mask = np.ones((mask_width, mask_width))
	b_mask[bg_win:b_mask.shape[0] - bg_win, bg_win:b_mask.shape[0] - bg_win] = 0

	# run the detector (only on fully valid points)
	mask = np.ones((n_row, n_col)) if mask is None else mask
	rx_img = np.zeros((n_row, n_col))
	half_width = guard_win + bg_win

	for i in range(n_col - mask_width + 1):
		for j in range(n_row - mask_width + 1):
			row = j + half_width
			col = i + half_width

			if mask[row, col] == 0: continue

			b_mask_img = np.zeros((n_row, n_col))
			b_mask_img[j:mask_width + j, i:mask_width + i] = b_mask
			b_mask_list = np.reshape(b_mask_img, (b_mask_img.size), order='F')

			# pull out background points
			bg = hsi_data[:, [int(m) for m in np.argwhere(b_mask_list == 1)]]

			# Mahalanobis distance
			covariance = np.cov(bg.T, rowvar=False)
			s = np.float32(np.linalg.svd(covariance, compute_uv=False))
			rcond = np.max(covariance.shape)*np.spacing(np.float32(np.linalg.norm(s, ord=np.inf)))
			# pinv differs from MATLAB
			sig_inv = np.linalg.pinv(covariance, rcond=rcond)

			mu = np.mean(bg, 1)
			z = hsi_img[row, col, :] - mu
			z = np.reshape(z, (len(z), 1), order='F')

			rx_img[row, col] = z.T @ sig_inv @ z

	return rx_img
