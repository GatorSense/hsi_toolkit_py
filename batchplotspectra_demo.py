import scipy.io as sio
import numpy as np
from hsi_toolkit.util.BatchPlotSpectra.batchplotspectra import PlotSpectraDistribution

an_hsi_image_sub_for_demo = sio.loadmat('an_hsi_image_sub_for_demo.mat')
hsi_img_sub = an_hsi_image_sub_for_demo['hsi_img_sub']
wavelengths = an_hsi_image_sub_for_demo['wavelengths']
mask_sub = an_hsi_image_sub_for_demo['mask_sub']

x_dims, y_dims, band_dims = hsi_img_sub.shape

mat_data = np.reshape(hsi_img_sub, (x_dims*y_dims, band_dims))

mask_reshaped = np.reshape(mask_sub, (x_dims*y_dims))

masked_data = mat_data[mask_reshaped == 1]

PlotSpectraDistribution(masked_data, wavelengths, [1,1,100], 1)