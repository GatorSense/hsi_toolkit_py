import numpy as np
from get_hsi_bands import get_hsi_bands

def get_RGB(hsi_img, wavelengths):
	"""
	 Creates an RGB image from a hyperspectral image

	 inputs:
	  hsi_img - n_row x n_col x n_band hyperspectral image
	  wavelengths - 1 x n_band vector listing wavelength values for hsi_img in nm
	 outputs:
	  RGB_img - n_row x n_col x 3 RGB image

	 5/5/2018 - Alina Zare
	 10/5/2018 - Yutai Zhou
	"""
	n_row, n_col, n_band = hsi_img.shape
	red_wavelengths = list(range(620,660))
	green_wavelengths = list(range(550,571))
	blue_wavelengths = list(range(450,496))

	RGB_img = np.zeros((n_row, n_col, 3))
	RGB_img[:,:,0] = np.mean(get_hsi_bands(hsi_img, wavelengths, red_wavelengths), axis=2);
	RGB_img[:,:,1] = np.mean(get_hsi_bands(hsi_img, wavelengths, green_wavelengths), axis=2);
	RGB_img[:,:,2] = np.mean(get_hsi_bands(hsi_img, wavelengths, blue_wavelengths), axis=2);

	RGB_img = (RGB_img - np.min(RGB_img.flatten())) / (np.max(RGB_img.flatten()) - np.min(RGB_img.flatten())) ** (1/1.5)
	# print(RGB_img[:,0,0])
	return RGB_img

# RGB_img = ((RGB_img - min(RGB_img(:)))/(max(RGB_img(:))-min(RGB_img(:)))).^(1/1.5);
