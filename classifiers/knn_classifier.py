import sys
sys.path.append('../util/')
from img_det import img_det
import numpy as np
def knn_classifier(hsi_img, train_data, k, mask = None):
	"""
	 A simple K nearest neigbors classifier

	Inputs:
	  hsi_img - hyperspectral data cube (n_rows x n_cols x n_bands)
	  train_data - numpy void structure containing training data
	      		   train_data[i].Spectra: matrix containing training data from class i
	  mask - binary image indicating where to apply classifier
	  k - number of neighbors to use during classification

	10/31/2012 - Taylor C. Glenn
	05/12/2018 - Edited by Alina Zare
	10/2018 - Python Implementation by Yutai Zhou
	"""
	print(train_data[0][0][0])

	knn_out = 0

	# knn_out = img_det(@knn_cfr, hsi_img, train_data, mask, k);
	return knn_out

def knn_cfr(hsi_data, train_data, k):
	knn_out = 0
	return knn_out
