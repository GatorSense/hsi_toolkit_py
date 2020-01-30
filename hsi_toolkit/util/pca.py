import numpy as np
import scipy as scp

def pca(X, frac, mask = None):
	"""
	function [y,n_dim,vecs,vals,mu] = pca(x,frac,mask)

	Principal Components Analysis
	 transforms input data x using PCA, reduces dimensionality
	 to capture fraction of magnitude of eigenvalues

	inputs:
	 x - input data M dimensions by N samples
	 frac - [0-1] fractional amount of total eigenvalue magnitude to retain, 1 = no dimensionality reduction
	 mask - binary mask of samples to include in covariance computation (use to mask off invalid pixels)
	         leave off the argument or use [] for no mask

	outputs:
	 y - PCA transformed dimensionality reduced data
	 n_dim - number of dimensions in the output data
	 vecs - full set of eigenvectors of covariance matrix (column vectors)
	 vals - eigenvalues of covariance matrix
	 mu - mean of input data, subtracted before rotating

	8/7/2012 - Taylor C. Glenn - tcg@cise.ufl.edu
	11/2018 - Python Implementation by Yutai Zhou
	"""
	n_dim, n_sample = X.shape
	if n_dim > 210:
		input('That is a lot of dimensions. It may crash. Press ctrl + c to stop execution, or enter to continue')

	mask = np.ones(n_sample, dtype=bool) if mask is None else mask
	mu = np.mean(X[:, mask == 1], 1)
	sigma = np.cov(X[:, mask == 1].T,  rowvar=False)

	z = X - mu[:, np.newaxis]
	U, S, V = scp.linalg.svd(sigma, lapack_driver = 'gesdd')

	evecs = U

	mag = np.sum(S)
	ind = np.where(np.cumsum(S) / mag >= frac)[0][0]
	n_dim = ind
	y = evecs[:, :ind + 1].T @ z # d x 72 * 72 x 4488
	return y, n_dim, evecs, S, mu
