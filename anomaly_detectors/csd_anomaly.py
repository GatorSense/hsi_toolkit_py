import sys
sys.path.append('../util/')
from pca import pca
import numpy as np

def csd_anomaly(hsi_img, n_dim_bg, n_dim_tgt, tgt_orth):
	"""
	Complementary Subspace Detector
	 assumes background and target are complementary subspaces
	 of PCA variance ranked space
	Ref: A. Schaum, "Joint subspace detection of hyperspectral targets," 2004 IEEE Aerospace Conference Proceedings (IEEE Cat. No.04TH8720), 2004, pp. 1824 Vol.3. doi: 10.1109/AERO.2004.1367963

	inputs:
	  hsi_image - n_row x n_col x n_band
	  n_dim_bg - number of leading dimensions to assign to background subspace
	  n_dim_tgt - number of dimensions to assign to target subspace
	              use empty matrix, [], to use all remaining after background assignment
	  tgt_orth - True/False, set target subspace orthogonal to background subspace

	8/7/2012 - Taylor C. Glenn
	5/5/2018 - Edited by Alina Zare
	11/2018 - Python Implementation by Yutai Zhou
	"""
