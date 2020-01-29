import numpy as np
def get_hsi_bands(hsi_img, wavelengths, wavelengths_to_get):
	"""
	Return hsi stack with bands that are closest to desired wavelengths

	inputs:
		hsi_img - n_row x n_col x n_band hyperspectral image
		wavelengths - 1 x n_band vector listing wavelength values for hsi_img
		wavengths_to_get - 1 x n_band_new vector listing desired wavelengths
	outputs:
		hsi_out - n_row x n_col x n_band_new  image

	5/5/2018 - Alina Zare
	10/5/2018 - Yutai Zhou
	"""
	waves = []
	for i in range(len(wavelengths_to_get)):
		min_index = np.argmin(np.abs(wavelengths - wavelengths_to_get[i]))
		waves.append(min_index)
	waves=np.array(waves)
	hsi_out = hsi_img[:,:,[int(m) for m in waves]];
	return hsi_out
