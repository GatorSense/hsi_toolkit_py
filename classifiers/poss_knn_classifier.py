def poss_knn_classifier(hsi_img,train_data,k,m,eta, mask = None):
	"""
	Possibilistic K nearest nieghbors classifier

	Ref: Frigui, Hichem, and Paul Gader. "Detection and discrimination of land mines in ground-penetrating radar based on edge histogram descriptors and a possibilistic $ k $-nearest neighbor classifier." IEEE Transactions on Fuzzy Systems 17.1 (2009): 185-199.

	Inputs:
	  hsi_img: hyperspectral data cube (n_rows x n_cols x n_bands)
	  train_data: structure containing training data
	      		  train_data(i).Spectra: matrix containing training data from class i
	  mask: binary image indicating where to apply classifier
	  K:  number of neighbors to use during classification
	  m:  fuzzifier (usually = 2)
	  eta: eta parameter to determine what is an outlier

	Outputs:
	  pknn_img: class membership matrix (n_row x n_col x n_class)

	6/3/2018 - Alina Zare
	10/2018 - Python Implementation by Yutai Zhou
	"""
	return true
