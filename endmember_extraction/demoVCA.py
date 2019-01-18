"""
Demo script that runs the VCA algorithm using example sub MUUFL Gulfport data

Inputs:
	hsi_img_sub - n_row x n_col x n_band hyperspectral image
	wavelengths - n_band x 1 vector listing wavelength values for hsi_img in nm
	mask_sub - n_row x n_col binary image limiting detector operation to pixels where mask is true
		       if not present or empty, no mask restrictions are used
Outputs:
	det_out - dictionary of detector output images

1/17/2019 - Ronald Fick
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from VCA import VCA
import scipy.io as sio

an_hsi_image_sub_for_demo = sio.loadmat('an_hsi_image_sub_for_demo.mat')
hsi_img_sub = an_hsi_image_sub_for_demo['hsi_img_sub']
wavelengths = an_hsi_image_sub_for_demo['wavelengths']
mask_sub = an_hsi_image_sub_for_demo['mask_sub']

x_dims, y_dims, band_dims = hsi_img_sub.shape

mat_data = np.reshape(hsi_img_sub, (x_dims*y_dims, band_dims))

mask_reshaped = np.reshape(mask_sub, (x_dims*y_dims))

masked_data = mat_data[mask_reshaped == 1]

M = 3

E, IdxOfE, Xpca = VCA(np.transpose(masked_data), M=M)

print(Xpca.shape)

fig = plt.figure()
ax = fig.gca(projection='3d')

nonendmembers = np.delete(np.arange(Xpca.shape[1]), IdxOfE)
ax.scatter(Xpca[0,nonendmembers], Xpca[1,nonendmembers], Xpca[2,nonendmembers], s=5, c='b')
ax.scatter(Xpca[0,IdxOfE], Xpca[1,IdxOfE], Xpca[2,IdxOfE], s=40, c='r')
plt.title('Gulfport Data Projected to 3D - Endmembers in Red')

plt.figure()
plt.plot(wavelengths, E)
plt.title('Estimated Endmembers from Gulfport Data')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.show()