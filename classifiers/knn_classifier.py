import sys
sys.path.append('../util/')
from img_det import img_det
import numpy as np
from sklearn.neighbors import NearestNeighbors

def knn_classifier(hsi_img, train_data, K, mask = None):
	"""
	 A simple K nearest neighbors classifier

	Inputs:
	  hsi_img - hyperspectral data cube (n_rows x n_cols x n_bands)
	  train_data - numpy void structure containing training data
	      		   train_data['Spectra'][0, i]: matrix containing training data from class i
				   train_data['name'][0, i]: matrix containing name of class i
	  mask - binary image indicating where to apply classifier
	  K - number of neighbors to use during classification

	10/31/2012 - Taylor C. Glenn
	05/12/2018 - Edited by Alina Zare
	10/2018 - Python Implementation by Yutai Zhou
	"""
	knn_out, kwargsout = img_det(knn_cfr, hsi_img, train_data, mask = mask, K = K);
	return knn_out

def knn_cfr(hsi_data, train_data, kwargs):
	K = kwargs['K']
	train_data = train_data.squeeze()
	# concatenate the training data
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

	# classify by majority of K nearest neighbors
	knn_out = np.zeros((n_pix,1))
	knn = NearestNeighbors(n_neighbors=K)
	knn.fit(train.T)
	idx = knn.kneighbors(hsi_data.T)[1]

	for i in range(n_pix):
		counts = np.zeros((n_class, 1))
		for j in range(K):
			counts[int(labels[idx[i,j]])] = counts[int(labels[idx[i,j]])] + 1
		max_i = np.argmax(counts)
		knn_out[i,0] = max_i
	return knn_out.squeeze(), {'None': None}
