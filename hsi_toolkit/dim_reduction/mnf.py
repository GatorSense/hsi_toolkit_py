import numpy as np

def mnf(in_img, eigval_retain = 1):
	"""
	Maximum Noise Fraction code

	Ref: Green, A. A., Berman, M., Switzer, P., & Craig, M. D. (1988). A transformation for ordering multispectral data in terms of image quality with implications for noise removal. IEEE Transactions on geoscience and remote sensing, 26(1), 65-74.

	Inputs:
	  in_img: hyperspectral data cube (n_row x n_cols x n_bands)
	  eigval_retain: percentage of eigenvalue to retain durig dimensionality reduction step. If 1, no reduction is done.

	Outputs:
	  out_img: noise ordered and dimensionality reduced data

	Author: Alina Zare
	Email Address: azare@ufl.edu
	Created: 2008
	Latest Revision: June 3, 2018
	Python Implementation by Yutai Zhou on 12/2018
	"""
	# get the noise covariance
	# assumes neighbor pixels are essentially the same except for noise
	# use a simple mask of neighbor pixels to the right and below
	n_row, n_col, n_band = in_img.shape
	n_pixel = n_row * n_col

	hsi_data = in_img.reshape((n_pixel, n_band), order='F').T

	mu = np.mean(hsi_data,1)[:,np.newaxis]
	running_cov = np.zeros((n_band, n_band))

	for i in range(n_col - 1):
		for j in range(n_row - 1):

			diff1 = (in_img[j,i+1,:] - in_img[j,i,:])[:,np.newaxis]
			diff2 = (in_img[j+1,i,:] - in_img[j,i,:])[np.newaxis,:]
			running_cov = running_cov + diff1 @ diff1.T + diff2 @ diff2.T

	noise_cov = 1 / (2 * (n_row - 1) * (n_col - 1) - 1) * running_cov
	U_noise, S_noise, _ = np.linalg.svd(noise_cov)
	S_noise = np.diag(S_noise)

	# align and whiten noise
	hsi_prime = np.linalg.pinv(np.sqrt(S_noise)) @ U_noise @ (hsi_data - mu)

	# PCA the noise whitened data
	U, S, _ = np.linalg.svd(np.cov(hsi_prime.T, rowvar=False))

	hsi_mnf = U @ hsi_prime

	out_img = hsi_mnf.T.reshape((n_row, n_col, n_band), order = 'F')

	A = U @ np.linalg.pinv(np.sqrt(S_noise)) @ U_noise

	n_dim = n_band
	if eigval_retain < 1:
		pcts = np.cumsum(S) / np.sum(S)
		cut_ind = np.where(pcts >= eigval_retain)[0]

		out_img = out_img[:,:,:cut_ind[0]+1]
		n_dim = cut_ind[0]+1
		# mu = mu[:cut_ind[0]+1,0][:,np.newaxis]
		# A = A[:,:cut_ind[0]+1]
		A = A[:cut_ind[0]+1,:]

	eig_vals = np.diag(S[:n_dim])

	# print(n_dim,mu.shape, A.shape)
	return out_img, n_dim, A, eig_vals, mu
