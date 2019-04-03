import numpy as np
from sklearn.cluster import KMeans

def ctmf_detector(hsi_img, tgt_sig, n_cluster = 2):
	"""
	Cluster Tuned Matched Filter
	 k-means cluster all spectra, make a matched filter for each cluster

	Inputs:
	 hsi_image - n_row x n_col x n_band hyperspectral image
	 tgt_sig - target signature (n_band x 1 - column vector)
	 n_cluster - number of clusters to use

	Outputs:
	 ctmf_out - detector output image
	 cluster_img - cluster label image

	8/15/2012 - Taylor C. Glenn
	6/02/2018 - Edited by Alina Zare
	12/2018 - Python Implementation by Yutai Zhou
	"""
	if tgt_sig.ndim == 1:
		tgt_sig = tgt_sig[:, np.newaxis]

	n_row, n_col, n_band = hsi_img.shape
	n_pixel = n_row * n_col

	hsi_data = hsi_img.reshape((n_pixel,n_band), order='F').T

	# cluster the data
	idx = KMeans(n_clusters = n_cluster, n_init = 1, max_iter=100).fit(hsi_data.T).labels_

	cluster_img = idx.reshape((n_row, n_col), order = 'F')

	# get cluster stats, create match filters
	mu = np.zeros((n_band,n_cluster))
	sig_inv = np.zeros((n_band, n_band, n_cluster))
	f = np.zeros((n_band,n_cluster))

	for i in range(n_cluster):
		z = hsi_data[:, idx == i]

		mu[:,i] = np.mean(z,1)
		sig_inv[:,:,i] = np.linalg.pinv(np.cov(z.T, rowvar=False))

		s = tgt_sig - mu[:,i][:,np.newaxis]
		f[:,i] = s.T @ sig_inv[:,:,i] / np.sqrt(s.T @ sig_inv[:,:,i] @ s)

	# compute matched filter output of each point
	ctmf_data = np.zeros(n_pixel)

	for i in range(n_pixel):
		z = hsi_data[:,i] - mu[:,idx[i]]
		ctmf_data[i] = f[:,idx[i]] @ z

	return ctmf_data.reshape([n_row, n_col], order='F'), cluster_img
