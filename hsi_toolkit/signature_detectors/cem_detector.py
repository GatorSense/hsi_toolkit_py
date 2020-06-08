from hsi_toolkit.util import img_det
import numpy as np

def cem_detector(hsi_img, tgt_sig, mask = None):
	"""
	Constrained Energy Minimization Detector
	 solution to filter with minimum energy projected into background space

	Ref: J. C. Harsanyi, ?Detection and classification of subpixel spectral signatures in hyperspectral image sequences,? Ph.D. dissertation, University of Maryland Baltimore County, 1993.

	Inputs:
	 hsi_image - n_row x n_col x n_band hyperspectral image
	 tgt_sigs - target signatures (n_band x n_sigs)
	 mask - binary image limiting detector operation to pixels where mask is true
	        if not present or empty, no mask restrictions are used

	Outputs:
	 cem_out - detector image
	 w - cem filter

	8/8/2012 - Taylor C. Glenn
	6/2/2018 - Edited by Alina Zare
	12/2018 - Python Implementation by Yutai Zhou
	"""
	if tgt_sig.ndim == 1:
		tgt_sig = tgt_sig[:, np.newaxis]

	cem_out, kwargsout = img_det(cem_helper, hsi_img, tgt_sig, mask)

	return cem_out, kwargsout['w']

def cem_helper(hsi_data, tgt_sig, kwarg):
	n_pixel = hsi_data.shape[1]
	n_sigs = tgt_sig.shape[1]

	R = np.cov(hsi_data.T, rowvar=False)
	mu = np.mean(hsi_data,1)
	mu = mu[:,np.newaxis]

	z = hsi_data - mu
	M = tgt_sig - mu
	R_inv = np.linalg.pinv(R)

	w = R_inv @ M @ np.linalg.pinv(M.T @ R_inv @ M) * np.ones((n_sigs,1))

	cem_data = w.T @ z
	return cem_data.squeeze(), {'w':w}
