from scipy.io import loadmat
from hsi_toolkit.dim_reduction import *
from hsi_toolkit.dev.dim_reduction.mnf import *
from hsi_toolkit.util.get_RGB import get_RGB
# Demo script that runs all dimensionality reduction methods in hsi_toolkit_py
#
# Inputs:
#  img: hyperspectral data cube (n_row x n_col x n_bands)
#  wavelengths: vector containing wavelengths of each HSI band (n_bands x  1)
#
# Outputs:
#   im_reduced: cell array containing reduced data results
#
# Author: Alina Zare
# Email Address: azare@ufl.edu
# Created: June 3, 2018

# load data
an_hsi_image_sub_for_demo = loadmat('an_hsi_image_sub_for_demo.mat')
img = an_hsi_image_sub_for_demo['hsi_img_sub']
wavelengths = an_hsi_image_sub_for_demo['wavelengths']

# run hdr
hdr_out = dimReduction(img)
# run mnf
mnf_out = mnf(img, 0.999)[0]
mnf_out = (mnf_out - np.min(mnf_out)) / np.max(mnf_out)

# visualize
plt.subplot(1,3,1)
plt.title("RGB Image")
plt.imshow(get_RGB(img, wavelengths))
plt.subplot(1,3,2)
plt.title("HDR Image")
plt.imshow(hdr_out[:,:,:3])
plt.subplot(1,3,3)
plt.title("MNF Image")
plt.imshow(mnf_out[:,:,:3])
plt.show()
