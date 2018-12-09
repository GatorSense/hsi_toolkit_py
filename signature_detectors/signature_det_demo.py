import sys
from scipy.io import loadmat
import matplotlib.pyplot as plt
sys.path.append('../')
sys.path.append('../util/')
from signature_detectors import *
from get_RGB import get_RGB
"""
Demo script that runs all signature detectors in hsi_toolkit_py

Inputs:
	hsi_sub - n_row x n_col x n_band hyperspectral image
	tgt_spectra - n_band x 1 target signature vector
	wavelengths - n_band x 1 vector listing wavelength values for hsi_sub in nm
	gt_img_sub - n_row x n_col ground truths
	mask - binary image limiting detector operation to pixels where mask is true
	 	if not present or empty, no mask restrictions are used
Outputs:
	det_out - dictionary of RGB image, ground truth image, and detector outputs

6/2/2018 - Alina Zare
10/12/2018 - Python Implementation by Yutai Zhou
"""
# Load data
an_hsi_img_for_tgt_det_demo = loadmat('an_hsi_img_for_tgt_det_demo.mat')
hsi_sub = an_hsi_img_for_tgt_det_demo['hsi_sub']
tgt_spectra = an_hsi_img_for_tgt_det_demo['tgt_spectra']
wavelengths = an_hsi_img_for_tgt_det_demo['wavelengths']
gt_img_sub = an_hsi_img_for_tgt_det_demo['gtImg_sub']

det_out = {}
det_out['RGB'] = get_RGB(hsi_sub, wavelengths)
det_out['Ground Truth'] = gt_img_sub

# init detector args
guard_win = 2; bg_win = 4; beta = 0.001; n_dim_ss = 10

# call detectors
# ace_out, _, _ = ace_detector(hsi_sub, tgt_spectra)
# det_out['ACE Squared'] = ace_out
# ace_local_out, _ = ace_local_detector(hsi_sub, tgt_spectra, guard_win = guard_win, bg_win = bg_win, beta = beta)
# det_out['ACE Local Squared'] = ace_local_out
# ace_ss_out = ace_ss_detector(hsi_sub, tgt_spectra)
# det_out['ACE SS'] = ace_ss_out
# ace_rt_out, _, _ = ace_rt_detector(hsi_sub, tgt_spectra)
# det_out['ACE RT'] = ace_rt_out
# ace_rt_max_out, _, _ = ace_rt_max_detector(hsi_sub, tgt_spectra)
# det_out['ACE RT Max'] = ace_rt_max_out
# smf_out, _, _ = smf_detector(hsi_sub, tgt_spectra)
# det_out['SMF'] = smf_out
# smf_local_out = smf_local_detector(hsi_sub, tgt_spectra, guard_win = guard_win, bg_win = bg_win)
# det_out['SMF Local'] = smf_local_out
# smf_max_out = smf_max_detector(hsi_sub, tgt_spectra)
# det_out['SMF Max'] = smf_max_out
# fam_statistic_out = fam_statistic(hsi_sub, tgt_spectra)
# det_out['FAM Statistic'] = fam_statistic_out
# osp_out = osp_detector(hsi_sub, tgt_spectra, n_dim_ss = 10)
# det_out['OSP'] = osp_out
# qmf_out = qmf_detector(hsi_sub, tgt_spectra, 0.1 * np.eye(hsi_sub.shape[2]))
# det_out['QMF'] = qmf_out
# sam_out = sam_detector(hsi_sub, tgt_spectra)
# det_out['SAM'] = sam_out
spsmf_out = spsmf_detector(hsi_sub, tgt_spectra)
det_out['SPSMF'] = spsmf_out
# visualization
# plt.figure(figsize=(10, 15))
# plt.subplots_adjust(hspace=.5)
# n_row = 4; n_col = 3
#
# i = 1
# for key, value in det_out.items():
# 	plt.subplot(n_row, n_col, i);
# 	plt.imshow(value); plt.title(key)
# 	i += 1
plt.imshow(spsmf_out)
plt.show()
