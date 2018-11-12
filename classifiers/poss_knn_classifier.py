import numpy as np
from sklearn.neighbors import NearestNeighbors

def poss_knn_classifier(hsi_img, train_data, K, eta, m = 2):
	"""
	Possibilistic K nearest neighbors classifier

	Ref: Frigui, Hichem, and Paul Gader. "Detection and discrimination of land mines in ground-penetrating radar based on edge histogram descriptors and a possibilistic $ k $-nearest neighbor classifier." IEEE Transactions on Fuzzy Systems 17.1 (2009): 185-199.

	Inputs:
	  hsi_img: hyperspectral data cube (n_rows x n_cols x n_bands)
	  train_data - numpy void structure containing training data
	      		   train_data['Spectra'][0, i]: matrix containing training data from class i
				   train_data['name'][0, i]: matrix containing name of class i
	  K:  number of neighbors to use during classification
	  m:  fuzzifier (usually = 2)
	  eta: eta parameter to determine what is an outlier

	Outputs:
	  pknn_img: class membership matrix (n_row x n_col x n_class)

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
	pknn_out = np.zeros((n_pix, n_class))
	knn = NearestNeighbors(n_neighbors=K)
	knn.fit(train.T)
	distance, idx = knn.kneighbors(hsi_data.T)
	weights = distance - eta
	weights[weights < 0] = 0
	weights = 1 / (1 + (weights ** (2/(m-1))) + np.finfo(float).eps)

	for i in range(n_pix):
		pknn_out[i,:] = np.sum(np.tile(weights[i,:][np.newaxis,:].T, (1,n_class)) * mu[idx[i,:],:],0) / K

	pknn_img = np.zeros((n_row, n_col, n_class))
	for i in range(n_class):
		pknn_img[:,:,i] = np.reshape(pknn_out[:,i], (n_row, n_col), order='F')

	return pknn_img
