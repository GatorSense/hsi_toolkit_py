import sys
sys.path.append('../util/')
from img_det import img_det
from unmix import unmix
import numpy as np

def hsd_detector(hsi_img, tgt_sig, ems, mask = None, sig_inv = None):
	"""
	Hybrid Structured Detector

	Ref:
	Hybrid Detectors for Subpixel Targets
	Broadwater, J. and Chellappa, R.
	Pattern Analysis and Machine Intelligence, IEEE Transactions on
	2007 Volume 29 Number 11 Pages 1891 -1903 Month nov.

	Inputs:
	 hsi_image - n_row x n_col x n_band hyperspectral image
	 tgt_sig - target signature (n_band x 1 - column vector)
	 mask - binary image limiting detector operation to pixels where mask is true
	        if not present or empty, no mask restrictions are used
	 ems - background endmembers
	 siginv - background inverse covariance (n_band x n_band matrix)

	Outputs:
	 hsd_out - detector image
	 tgt_p - target proportion in unmixing

	8/19/2012 - Taylor C. Glenn
	6/2/2018 - Edited by Alina Zare
	12/2018 - Python Implementation by Yutai Zhou
	"""
	hsd_out, kwargsout = img_det(hsd_helper, hsi_img, tgt_sig, mask, ems = ems, sig_inv = sig_inv)
	return hsd_out, kwargsout['tgt_p']

def hsd_helper(hsi_data, tgt_sig, kwargs):
	ems = kwargs['ems']
	sig_inv = np.linalg.pinv(np.cov(hsi_data.T, rowvar = False)) if kwargs['sig_inv'] is None else kwargs['sig_inv']

	n_pixel = hsi_data.shape[1]

	# unmix data with only background endmembers
	P = unmix(hsi_data, ems)

	# unmix data with target signature as well
	targ_P = unmix(hsi_data, np.hstack((tgt_sig, ems)))

	hsd_data = np.zeros(n_pixel)

	for i in range(n_pixel):
		z = hsi_data[:,i] - ems @ P[i,:].T
		w = hsi_data[:,i] - np.hstack((tgt_sig, ems)) @ targ_P[i,:].T
		hsd_data[i] = z[np.newaxis,:] @ sig_inv @ z[:,np.newaxis] / (w[np.newaxis,:] @ sig_inv @ w[:,np.newaxis])

	tgt_p = targ_P[:,:tgt_sig.shape[1]]
	return hsd_data, {'tgt_p': tgt_p.squeeze()}
