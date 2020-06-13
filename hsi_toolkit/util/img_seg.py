import numpy as np
def img_seg(det_func, hsi_img, tgt_sig, segments, **kwargs):
	"""
	Segmented Detector Wrapper
	 uses any detector with the signature detector(img,tgt_sig,mask,args...)
	 as a segmented detector over the given segments

	Inputs:
	 det_func - function handle for wrapped detector
	 hsi_img - n_row x n_col x n_band hyperspectral image
	 tgt_sig - target signature (n_band x 1 - column vector)
	 segments - cell array of segment masks, n_row x n_col binary images
	 varargin - variable array of arguments passed to the detector function

	Outputs:
	 det_out - detector output image, concatenation of outputs from each segment
	           NaN valued in pixels not contained by a segment
	 varargout - other outputs, assumed to be images

	8/20/2012 - Taylor C. Glenn
	12/2018 - Python Implementation by Yutai Zhou
	"""
	n_seg, n_row, n_col = segments.shape
	det_out = np.empty((n_row, n_col))
	segments = segments.astype(bool)

	for i in range(n_seg):
		out = det_func(hsi_img, tgt_sig, segments[i,:,:], **kwargs)
		if type(out) is tuple:
			out = list(out)
			seg_out = out[0]
			det_out[segments[i,:,:]] = seg_out[segments[i,:,:]]
			out[0] = det_out
		else:
			det_out[segments[i,:,:]] = out[segments[i,:,:]]
			out = det_out
	return out
