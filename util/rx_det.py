import numpy as np

def rx_det(det_func, hsi_img, tgt_sig, mask = None, guard_win = 2, bg_win = 4, **kwargs):
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
	n_row, n_col, n_band = hsi_img.shape
	n_pixel = n_row * n_col

	mask = np.ones([n_row, n_col]) if mask is None else mask

	# create the mask
	mask_width = 1 + 2 * guard_win + 2 * bg_win
	half_width = guard_win + bg_win

	b_mask = np.ones((mask_width, mask_width))
	b_mask[bg_win:b_mask.shape[0] - bg_win, bg_win:b_mask.shape[0] - bg_win] = 0

	hsi_data = np.reshape(hsi_img, (n_pixel, n_band), order='F').T

	# get global image/segment statistics in case we need to fall back on them
	global_mu = np.mean(hsi_data[:, [int(m) for m in np.argwhere(np.reshape(mask,-1,order='F') == 1)]], 1)
	global_cov = np.cov(hsi_data[:, [int(m) for m in np.argwhere(np.reshape(mask,-1,order='F') == 1)]].T, rowvar = False)
	# s = np.float32(np.linalg.svd(global_cov, compute_uv=False))
	# rcond = np.max(global_cov.shape)*np.spacing(np.float32(np.linalg.norm(s, ord=np.inf)))
	global_sig_inv = np.linalg.pinv(global_cov)

	args = {
	'hsi_data': hsi_data,
	'global_mu': global_mu,
	'global_sig_inv': global_sig_inv,
	'tgt_sig': tgt_sig,
	'n_sig': tgt_sig.shape[1]}

	# run the detector (only on fully valid points)
	ind_img = np.reshape(np.array([i for i in range(n_pixel)]), (n_row, n_col), order='F')
	out = np.empty((n_row, n_col))
	out[:] = np.nan
	det_stat = np.empty((n_row, n_col))
	det_stat[:] = np.nan


	for i in range(n_col - mask_width + 1):
		if i % 10 == 0:
			print('.')

		for j in range(n_row - mask_width + 1):
			row = j + half_width
			col = i + half_width

			if mask[row, col] == 0: continue

			b_mask_img = np.zeros((n_row, n_col))
			b_mask_img[j:mask_width + j, i:mask_width + i] = b_mask
			b_mask_img = np.logical_and(b_mask_img, mask)
			b_mask_list = np.reshape(b_mask_img, (b_mask_img.size), order='F')

			# pull out background and foreground points
			bg = hsi_data[:, [int(m) for m in np.argwhere(b_mask_list == 1)]]
			ind = ind_img[row, col]
			x = hsi_data[:, ind]

			# compute detection statistic
			out[row, col], kwargout = det_func(x, ind, bg, b_mask_list, args, kwargs)

			if 'sig_index' in kwargout:
				det_stat[row, col] = kwargout['sig_index']

	print('\n')
	return out, det_stat
