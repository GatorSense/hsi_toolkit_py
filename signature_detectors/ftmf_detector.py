import numpy as np

def ftmf_detector(hsi_img, tgt_sig, gamma = 1):
	"""
	Finite Target Matched Filter

	Inputs:
	 hsi_image - n_row x n_col x n_band hyperspectral image
	 tgt_sig - target signature (n_band x 1 - column vector)
	 gamma - scale factor of background variance to model target variance (V_t = gamma^2*V_bg)

	Outputs:
	 ftmf_out - detector image

	8/12/2012 - Taylor C. Glenn
	12/2018 - Python Implementation by Yutai Zhou


	makes the simplifying assumption that target variance
	 is a scaled version of bg variance
	Eismann pp 681
	"""
	n_row, n_col, n_band = hsi_img.shape
	n_pixel = n_row * n_col

	hsi_data = hsi_img.reshape((n_pixel,n_band), order='F').T

	mu = np.mean(hsi_data,1)
	mu = mu[:,np.newaxis]
	sigma = np.cov(hsi_data.T, rowvar=False)
	sig_inv = np.linalg.pinv(sigma)

	s = tgt_sig - mu
	z = hsi_data - mu
	f = s.T @ sig_inv

	#signal to cluster ratio
	scr = s.T @ sig_inv @ s

	ftmf_data = np.zeros(n_pixel)
	g2p1 = gamma ** 2 + 1
	for i in range(n_pixel):
		md = z[:,i].T @ sig_inv @ z[:,i] # mahalanobis distance
		mf = f @ z[:,i] # matched filter

		A = n_band * g2p1 ** 2
		B = (mf - 3 * n_band) * g2p1 ** 2 - scr
		C = -md * g2p1 + n_band * gamma ** 2 + 3 * n_band + scr
		D = -n_band - mf + md

		r = np.roots([A, B, C, D])

		r_ind = np.where(np.bitwise_and(r>=0, r<=1))
		if r_ind[0].shape[0] == 0:
			alpha = 1
		else:
			alpha = r[r_ind[0][0]]
		mu_a = alpha * s + (1 - alpha) * mu
		sigma_a = (alpha ** 2 * gamma ** 2 + (1 - alpha)**2) * sigma
		sig_inv_a = np.linalg.pinv(sigma_a)

		x_mu_a = hsi_data[:,i][:,np.newaxis] - mu_a

		ftmf_data[i] = (md - x_mu_a.T @ sig_inv_a @ x_mu_a - np.log(np.linalg.det(sigma_a))).squeeze()

	return ftmf_data.reshape((n_row, n_col), order='F')
