import numpy as np
def fam_statistic(hsi_img, tgt_sig, mu = None, sig_inv = None):
	"""
	False Alarm Mitigation Statistic from Subpixel Replacement Model

	Inputs:
	 hsi_image - n_row x n_col x n_band hyperspectral image
	 tgt_sig - target signature (n_band x 1 - column vector)
	 mu - background mean (n_band x 1 column vector)
	 siginv - background inverse covariance (n_band x n_band matrix)

	Outputs:
	 fam_out - false alarm mitigation statistic

	8/8/2012 - Taylor C. Glenn
	6/2/2018 - Edited by Alina Zare
	11/2018 - Python Implementation by Yutai Zhou
	"""
	# assume target variance same as background variance
	n_row, n_col, n_band = hsi_img.shape
	n_pixel = n_row * n_col

	hsi_data = hsi_img.reshape((n_pixel, n_band), order='F').T
	if mu is None:
		mu = np.mean(hsi_data, axis = 1)
	if sig_inv is None:
		sig_inv = np.linalg.pinv(np.cov(hsi_data.T, rowvar = False))

	mu = mu[:, np.newaxis]
	s = tgt_sig
	sts = s.T @ s
	s_mu = s - mu

	z = hsi_data - mu
	fam_data = np.zeros(n_pixel)

	for i in range(n_pixel):
		alpha = s.T @ hsi_data[:,i] / sts
		w = z[:,i][:, np.newaxis] - alpha * s_mu
		fam_data[i] = w.T @ sig_inv @ w

	fam_out = fam_data.reshape((n_row, n_col), order = 'F')
	return fam_out
