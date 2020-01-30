import sys
from hsi_toolkit.util.img_det import img_det
import numpy as np
from sklearn.cluster import KMeans
import skfuzzy as fuzz

def fcbad_anomaly(hsi_img, n_cluster, mask = None):
	"""
	Fuzzy Cluster Based Anomaly Detection (FCBAD)
	Ref: Hytla, Patrick C., et al. "Anomaly detection in hyperspectral imagery: comparison of methods using diurnal and seasonal data." Journal of Applied Remote Sensing 3.1 (2009): 033546

	This algorithm requires skfuzz! https://pythonhosted.org/scikit-fuzzy/install.html

	Inputs:
	 hsi_image - n_row x n_col x n_band hyperspectral image
	 mask - binary image limiting detector operation to pixels where mask is true
	        if not present or empty, no mask restrictions are used
	 n_cluster - number of clusters to use

	Outputs:
	 fcbad_out - detector output image
	 cluster_img - cluster label image

	8/8/2012 - Taylor C. Glenn
	5/5/2018 - Edited by Alina Zare
	11/2018 - Python Implementation by Yutai Zhou
	"""
	fcbad_out, kwargsout = img_det(fcbad_out_helper, hsi_img, None, mask, n_cluster = n_cluster)
	cluster_img = kwargsout['idx']
	return fcbad_out, cluster_img

def fcbad_out_helper(hsi_data, tgt_sig, kwargs):
	n_cluster = kwargs['n_cluster']
	n_band, n_pixel = hsi_data.shape

	options = {'c': n_cluster, # number of clusters
			   'm': 2.0,	       # exponent for the partition matrix
			   'error': 1e-6,  # minimum amount of improvement
			   'maxiter': 500} # max number of iterations

	C, U, _, _, _, _, _ = fuzz.cluster.cmeans(hsi_data, **options)
	idx = np.max(U,0)

	# Cluster stats
	mu = np.zeros((n_band, n_cluster))
	sig_inv = np.zeros((n_band, n_band, n_cluster))

	for i in range(n_cluster):
		mu[:,i] = C[i,:].T

		#computer membership weighted covariance for the cluster
		sigma = np.zeros((n_band, n_band))
		for j in range(n_pixel):
			z = hsi_data[:,j] - mu[:,i]
			sigma += U[i,j] * (z @ z.T)

		sigma /= np.sum(U[i,:])
		s = np.float32(np.linalg.svd(sigma, compute_uv=False))
		rcond = np.max(sigma.shape)*np.spacing(np.float64(np.linalg.norm(s, ord=np.inf)))
		sig_inv[:,:,i] = np.linalg.pinv(sigma, rcond=rcond)

	# compute total membership weighted Mahalanobis Distance
	fcbad_data = np.zeros(n_pixel)

	for j in range(n_pixel):
		m_dists = np.zeros(n_cluster)

		for i in range(n_cluster):
			z = hsi_data[:,j] - mu[:,i]
			m_dists[i] = z.T @ sig_inv[:,:,i] @ z

		fcbad_data[j] = U[:,j].T @ m_dists
	return fcbad_data, {'idx': idx}
