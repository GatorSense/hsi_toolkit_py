import sys
sys.path.append('../util/')
from img_det import img_det
sys.path.append('../dim_reduction/')
from mnf import mnf
import numpy as np

def mtmf_statistic(hsi_img,tgt_sig, mask = None):
	"""
	Mixture Tuned Matched Filter Infeasibility Statistic

	Inputs:
	 hsi_image - n_row x n_col x n_band hyperspectral image
	 tgt_sig - target signature (n_band x 1 - column vector)
	 mask - binary image limiting detector operation to pixels where mask is true
	        if not present or empty, no mask restrictions are used

	Outputs:
	 mtmf_out - MTMF infeasibility statistic
	 alpha - matched filter output

	8/12/2012 - Taylor C. Glenn - tcg@cise.ufl.edu
	12/2018 - Python Implementation by Yutai Zhou
	"""
	mnf_img, n_dim, mnf_vecs, mnf_eigvals, mnf_mu = mnf(hsi_img,1);
	# tgt_sig = tgt_sig[:n_dim,0][:,np.newaxis]
	s = mnf_vecs @ (tgt_sig - mnf_mu)
	# print(mnf_img.shape, mnf_vecs.shape, mnf_mu.shape, s.shape)

	mtmf_out, kwargsout = img_det(mtmf_helper, mnf_img, s, mnf_eigvals = mnf_eigvals)

	return mtmf_out, kwargsout['alpha']

def mtmf_helper(hsi_data, tgt_sig, kwargs):
	mnf_eigvals = kwargs['mnf_eigvals']
	n_band, n_pixel = hsi_data.shape

	z = hsi_data
	s = tgt_sig
	sts = s.T @ s

	alpha = np.zeros(n_pixel)
	mtmf_data = np.zeros(n_pixel)
	ev = np.sqrt(mnf_eigvals)
	one = np.ones(n_band)

	for i in range(n_pixel):
		# print(s.shape,z.shape, z[:,i].shape)
		a = (s.T @ z[:,i][:,np.newaxis] / sts).squeeze()
		alpha[i] = np.max((0, np.min((1, a))))
		# print(alpha[i])
		sig_inv = 1 / ((ev * (1 - alpha[i]) - one) ** 2)

		mtmf_data[i] = z[:,i][np.newaxis,:] @ sig_inv @ z[:,i][:,np.newaxis]

	return mtmf_data, {'alpha': alpha}
