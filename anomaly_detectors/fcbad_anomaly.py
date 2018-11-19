import sys
sys.path.append('../util/')
from img_det import img_det
import numpy as np
from sklearn.cluster import KMeans

def fcbad_anomaly(hsi_img, n_cluster, mask = None):
	"""
	Fuzzy Cluster Based Anomaly Detection (FCBAD)
	Ref: Hytla, Patrick C., et al. "Anomaly detection in hyperspectral imagery: comparison of methods using diurnal and seasonal data." Journal of Applied Remote Sensing 3.1 (2009): 033546

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
