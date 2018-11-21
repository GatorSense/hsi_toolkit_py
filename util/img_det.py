import numpy as np

def img_det(det_func, hsi_img, tgt_sig, mask = None, **kwargs):
	"""
	Wrapper to use array based detector as a image based detector with the given mask

	Taylor C. Glenn
	5/5/2018 - Edited by Alina Zare
	10/2018 - Python Implementation by Yutai Zhou
	"""
	n_row, n_col, n_band = hsi_img.shape
	n_pixels = n_row * n_col
	if mask is None:
		mask = np.ones(n_row * n_col, dtype=bool)
	else:
		mask_rows, mask_cols = mask.shape
		mask = mask.reshape(mask_rows * mask_cols, order ='F').astype(bool)
	hsi_data = np.reshape(hsi_img, (n_pixels, n_band), order='F').T

	# Linearize image-like inputs
	# Mask linearized (n x n) pixel arguments
	for key, val in kwargs.items():
		if type(val) == np.ndarray: # CHECK ME
			sz = val.shape
			if sz.size == 2 and sz == (n_row, n_col):
				val = np.reshape(val,(1, n_pixels), order='F')
				val = val[mask == 1]
				kwargs[key] = val
				print('image input to img_det!1')

			elif sz.size == 3 and sz[:2] == (n_row, n_col):
				val = np.reshape(val,(n_pix, sz[-1]), order='F').T
				val = val[:, mask == 1]
				kwargs[key] = val
				print('image input to img_det!2')

	det_data = np.empty(n_pixels)
	det_data[:] = np.nan
	det_data[mask == 1], kwargsout = det_func(hsi_data[:, mask == 1], tgt_sig, kwargs)

	if 'None' not in kwargsout:
		# Reshape image-like flattened outputs back into images
		for key, val in kwargsout.items():
			if type(val) is np.ndarray:
				if val.squeeze().ndim == 1 and val.size == np.sum(mask):
					tmp = np.empty(n_pixels)
					tmp[:] = np.nan
					tmp[mask == 1] = val
					kwargsout[key] = np.reshape(tmp, (n_row, n_col), order='F')

	return np.reshape(det_data, (n_row, n_col), order='F'), kwargsout
