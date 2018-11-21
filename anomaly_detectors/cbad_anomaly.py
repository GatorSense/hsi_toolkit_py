import sys
sys.path.append('../util/')
from img_det import img_det
import numpy as np
from sklearn.cluster import KMeans

def cbad_anomaly(hsi_img, n_cluster, mask = None):
	"""
	Cluster Based Anomaly Detection (CBAD)
	Ref: Carlotto, Mark J. "A cluster-based approach for detecting man-made objects and changes in imagery." IEEE Transactions on Geoscience and Remote Sensing 43.2 (2005): 374-387.

	Inputs:
	 hsi_image - n_row x n_col x n_band hyperspectral image
	 mask - binary image limiting detector operation to pixels where mask is true
	        if not present or empty, no mask restrictions are used
	 n_cluster - number of clusters to use

	Outputs:
	 cbad_out - detector output image
	 cluster_img - cluster label image

	8/7/2012 - Taylor C. Glenn
	5/5/2018 - Edited by Alina Zare
	11/2018 - Python Implementation by Yutai Zhou
	"""
	cbad_out, kwargsout = img_det(cbad_out_helper, hsi_img, None, mask, n_cluster = n_cluster)
	cluster_img = kwargsout['idx']
	return cbad_out, cluster_img

def cbad_out_helper(hsi_data, tgt_sig, kwargs):
	n_cluster = kwargs['n_cluster']
	n_bands, n_pixel = hsi_data.shape
	# cluster the data
	kmeans = KMeans(n_clusters = n_cluster, n_init = 1).fit(hsi_data.T)
	# get cluster stats
	mu = np.zeros((n_bands, n_cluster))
	sig_inv = np.zeros((n_bands, n_bands, n_cluster))

	for i in range(n_cluster):
		z = hsi_data[:, kmeans.labels_ == i]
		mu[:,i] = np.mean(z, 1)
		sig_inv[:,:,i] = np.linalg.pinv(np.cov(z.T, rowvar=False))

	cbad_data = np.zeros(n_pixel)
	for i in range(n_pixel):
		z = hsi_data[:,i] - mu[:,kmeans.labels_[i]]
		cbad_data[i] = z.T @ sig_inv[:,:,kmeans.labels_[i]] @ z
	return cbad_data, {'idx': kmeans.labels_}
