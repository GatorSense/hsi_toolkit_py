import numpy as np
def qmf_detector(hsi_img, tgt_sig, tgt_cov):
	"""
	Quadratic Spectral Matched Filter

	Inputs:
	 hsi_image - n_row x n_col x n_band hyperspectral image
	 tgt_sig - mean target signature (n_band x 1 - column vector)
	 tgt_cov - covariance matrix for target

	Outputs:
	 qmf_out - detector image

	8/9/2012 - Taylor C. Glenn
	10/2018 - Python Implementation by Yutai Zhou
	"""
	if tgt_sig.ndim == 1:
		tgt_sig = tgt_sig[:, np.newaxis]

	n_row, n_col, n_band = hsi_img.shape
	n_pixel = n_row * n_col

	hsi_data = hsi_img.reshape((n_pixel, n_band), order ='F').T

	#get an estimate of the noise covariance of the image
	running_cov = np.zeros((n_band, n_band))
	for i in range(n_col - 1):
		for j in range(n_row - 1):

			diff1 = (hsi_img[j,i+1,:] - hsi_img[j,i,:])[:,np.newaxis]
			diff2 = (hsi_img[j+1,i,:] - hsi_img[j,i,:])[np.newaxis,:]
			running_cov = running_cov + diff1 @ diff1.T + diff2 @ diff2.T

	noise_cov = running_cov / (2 * n_pixel - 1)
	# precompute other stats
	mu = np.mean(hsi_data,1)
	sigma = np.cov(hsi_data.T, rowvar=False)

	sig_inv_bn = np.linalg.pinv(sigma + noise_cov)
	sig_inv_sn = np.linalg.pinv(tgt_cov + noise_cov)

	z = hsi_data - mu[:, np.newaxis]
	w = hsi_data - tgt_sig

	# run the filter
	qmf_data = np.zeros(n_pixel)

	for i in range(n_pixel):
		qmf_data[i] = z[:,i].T @ sig_inv_bn @ z[:,i] - w[:,i].T @ sig_inv_sn @ w[:,i] + np.log(np.linalg.det(sigma + noise_cov)/np.linalg.det(tgt_cov + noise_cov))

	qmf_out = qmf_data.reshape((n_row, n_col), order = 'F')

	return qmf_out
