from hsi_toolkit.util.img_det import img_det
import numpy as np

def ace_rt_max_detector(hsi_img, tgt_sig, mask = None, mu = None, sig_inv = None):
	"""
	Adaptive Cosine/Coherence Estimator given Multiple Target Signatures.
	Confidence value is the max ace score over all target signatures.

	Inputs:
	 hsi_image - n_row x n_col x n_band hyperspectral image
	 tgt_sig - target signature (n_band x n_sig - column vector)
	 mask - binary image limiting detector operation to pixels where mask is true
	        if not present or empty, no mask restrictions are used
	 mu - background mean (n_band x 1 column vector)
	 siginv - background inverse covariance (n_band x n_band matrix)

	Outputs:
	 ace_out - detector image
	 mu - mean of input data
	 siginv - inverse covariance of input data

	8/8/2012 - Taylor C. Glenn
	6/2/2018 - Edited by Alina Zare
	11/2018 - Python Implementation by Yutai Zhou
	"""
	if tgt_sig.ndim == 1:
		tgt_sig = tgt_sig[:, np.newaxis]

	ace_rt_max_out, kwargsout = img_det(ace_rt_max_helper, hsi_img, tgt_sig, mask, mu = mu, sig_inv = sig_inv)
	return ace_rt_max_out, kwargsout['mu'], kwargsout['sig_inv']

def ace_rt_max_helper(hsi_data, tgt_sig, kwargs):
	mu = np.mean(hsi_data, axis = 1) if kwargs['mu'] is None else kwargs['mu']
	sig_inv = np.linalg.pinv(np.cov(hsi_data.T, rowvar = False)) if kwargs['sig_inv'] is None else kwargs['sig_inv']
	mu = mu[:, np.newaxis]

	n_sigs = tgt_sig.shape[1]
	n_pixel = hsi_data.shape[1]

	S = tgt_sig - mu
	z = hsi_data - mu
	det = np.zeros((n_sigs, n_pixel))

	for i in range(n_sigs):
		s = S[:,i][:,np.newaxis]
		st_sig_inv = s.T @ sig_inv
		st_sig_inv_s = s.T @ sig_inv @ s

		A = np.sum(st_sig_inv @ z, 0)
		B = np.sqrt(st_sig_inv_s)
		C = np.sqrt(np.sum(z * (sig_inv @ z), 0))

		det[i,:] = A / (B * C)

	ace_rt_data = np.max(det, 0)
	return ace_rt_data, {'mu':mu, 'sig_inv': sig_inv}
