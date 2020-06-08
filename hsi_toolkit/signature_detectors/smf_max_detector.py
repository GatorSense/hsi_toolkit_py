from hsi_toolkit.util import img_det
from smf_detector import smf_det_array_helper
import numpy as np

def smf_max_detector(hsi_img, tgt_sig, mask = None, mu = None, sig_inv = None):
	"""
	Spectral Matched Filter, Max over targets

	Inputs:
	 hsi_image - n_row x n_col x n_band hyperspectral image
	 tgt_sigs - target signatures (n_band x n_signatures)
	 mask - binary image limiting detector operation to pixels where mask is true
	        if not present or empty, no mask restrictions are used

	Outputs:
	 smf_out - detector image

	8/8/2012 - Taylor C. Glenn
	12/2018 - Python Implementation by Yutai Zhou
	"""
	if tgt_sig.ndim == 1:
		tgt_sig = tgt_sig[:, np.newaxis]

	n_sig = tgt_sig.shape[1]
	n_row, n_col, n_band = hsi_img.shape

	sig_out = np.zeros((n_row, n_col, n_sig))

	for i in range(n_sig):
		sig_out[:,:,i], kwargsout = img_det(smf_det_array_helper, hsi_img, tgt_sig[:,i][:,np.newaxis], mask, mu = mu, sig_inv = mu)

	smf_out = np.max(sig_out, 2)
	return smf_out
