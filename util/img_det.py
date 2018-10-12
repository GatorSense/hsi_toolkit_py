import numpy as np
def img_det(det_func, hsi_img, tgt_sig, mask = None, *args):
"""
function [det_out,varargout] = img_det(detector_fn,hsi_img,tgt_sig,mask,varargin)

 wrapper to use array based detector as a image based detector with the given mask

 Taylor C. Glenn
 5/5/2018 - Edited by Alina Zare
"""
n_row, n_col, n_band = hsi_img.shape
n_pixels = n_row * n_col

if mask is None:
	mask = np.ones((n_row, n_col))


return det_out
