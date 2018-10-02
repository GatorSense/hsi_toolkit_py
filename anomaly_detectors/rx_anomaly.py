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

	mask = np.ones((n_row, n_col)) if mask is None else mask

	# Create the mask
	mask_width = 1 + 2 * guard_win + 2 * bg_win
	half_width = guard_win + bg_win
	# mask_rg = np.array(range(1, mask_width + 1)) - 2
	mask_rg = mask_width - 1# 13 - 1 = 12

	b_mask = np.ones((mask_width, mask_width))
	b_mask[bg_win:b_mask.shape[0] - bg_win, bg_win:b_mask.shape[0] - bg_win] = False

	hsi_data = np.reshape(hsi_img, (n_pixels, n_band)).T

	# run the detector (only on fully valid points)
	rx_img = np.zeros((n_row, n_col))

	for i in range(1, n_col - mask_width + 2):
		for j in range(1, n_row - mask_width + 2):
			row = j + half_width
			col = i + half_width

			if mask[row, col] is None: continue

			b_mask_img = np.zeros((n_row, n_col))
			b_mask_img[-1 + j:mask_rg + j, -1 + i:mask_rg + i] = b_mask
			b_mask_list = np.reshape(b_mask_img, -1)

			bg = hsi_data[:, [int(m) for m in b_mask_list]]
			sig_inv = np.linalg.pinv(np.cov(bg.T))
			mu = np.mean(bg, 1)
			z = np.squeeze(hsi_img[row, col,:]) - mu
			rx_img[row, col] = z.T @ siginv @ z
	return rx_img
