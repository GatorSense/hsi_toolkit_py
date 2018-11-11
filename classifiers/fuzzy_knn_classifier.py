import sys
sys.path.append('../util/')
from img_det import img_det
import numpy as np
from sklearn.neighbors import NearestNeighbors

def fuzzy_knn_classifier(hsi_img, train_data, K, m = 2):
	"""
	Fuzzy K nearest nieghbors classifier

	Ref: Keller, J. M., Gray, M. R., & Givens, J. A. (1985). A fuzzy k-nearest neighbor algorithm. IEEE transactions on systems, man, and cybernetics, (4), 580-585.

	Inputs:
	  hsi_img: hyperspectral data cube (n_rows x n_cols x n_bands)
	  train_data:  train_data - numpy void structure containing training data
 	      		   train_data[0,i].Spectra: matrix containing training data from class i
	  K:  number of neighbors to use during classification
	  m:  fuzzifier (usually = 2)

	Outputs:
	  fknn_img: class membership matrix (n_row x n_col x n_class)

	6/3/2018 - Alina Zare
	10/2018 - Python Implementation by Yutai Zhou
	"""
	n_row, n_col, n_band = hsi_img.shape
	n_pixels = n_row * n_col
	hsi_data = np.reshape(hsi_img, (n_pixels, n_band), order='F').T

	# concatenate the training data
	train_data = train_data.squeeze()
	train = np.hstack([class_data for class_data in train_data['Spectra']])
	n_train = train.shape[1]

	labels = np.zeros((n_train, 1))
	n_class = train_data.size

	last = -1
	for i in range(n_class):
		nt = train_data[i]['Spectra'].shape[1]
		labels[(last + 1):(last + nt + 1)] = i
		last += nt

	n_pix = hsi_data.shape[1]

	#compute mu weights for training data
	knn_mu = NearestNeighbors(n_neighbors=K)
	knn_mu.fit(train.T)
	idx_train = knn_mu.kneighbors(train.T)[1]
	idx_labels = labels[idx_train].squeeze()
	mu = np.zeros((n_train, n_class))

	for i in range(n_train):
		unique_labels = np.unique(idx_labels[i, :])
		for j in range(len(unique_labels)):
			count = np.sum(idx_labels[i,:] == unique_labels[j])
			if unique_labels[j] == labels[i,0]:
				mu[i, int(unique_labels[j])] = 0.51 + count * 0.49 / K
			else:
				mu[i, int(unique_labels[j])] = count * 0.49 / K

	# classify with weighted KNN
	fknn_out = np.zeros((n_pix, n_class))
	knn = NearestNeighbors(n_neighbors=K)
	knn.fit(train.T)
	distance, idx = knn.kneighbors(hsi_data.T)
	weights = 1 / (distance ** (2 / (m - 1)) + np.finfo(float).eps)
	weights = weights / (np.tile(np.sum(weights,1)[:,np.newaxis],(1,K)))

	for i in range(n_pix):
		fknn_out[i,:] = np.sum(np.tile(weights[i,:][np.newaxis,:].T, (1,n_class)) * mu[idx[i,:],:],0)

	fknn_img = np.zeros((n_row, n_col, n_class))
	for i in range(n_class):
		fknn_img[:,:,i] = np.reshape(fknn_out[:,i], (n_row, n_col), order='F')
	return fknn_img
