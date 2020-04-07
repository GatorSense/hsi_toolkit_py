from scipy.io import loadmat
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
sys.path.append('../util/')
from get_RGB import get_RGB
from utilities_VI import *
import os

"""
Demo script that runs all spectral indices in hsi_toolkit_py and visualizes the results. 

Inputs:
	hsi_sub - n_row x n_col x n_band hyperspectral image
	wavelengths - n_band x 1 vector listing wavelength values for hsi_sub in nm
	mask - binary image limiting detector operation to pixels where mask is true
	 	if not present or empty, no mask restrictions are used
Outputs:
	det_out - dictionary of RGB image and spectral indices outputs

04/2020 - Susan Meerdink
"""

# Load data
an_hsi_img_for_VI_demo = loadmat('a_hsi_img_for_VI_demo.mat')
hsi_sub = an_hsi_img_for_VI_demo['hsi_img']
wavelengths = an_hsi_img_for_VI_demo['wavelengths']

# Set up True Color or Red/Green/Blue Image
det_out = {}
det_out['RGB'] = get_RGB(hsi_sub, wavelengths)

# Call Spectral Indices
det_out['ACI'] = aci_vi(hsi_sub, wavelengths)
det_out['ARI'] = ari_vi(hsi_sub, wavelengths)
det_out['ARVI'] = arvi_vi(hsi_sub, wavelengths)
det_out['CAI'] = cai_vi(hsi_sub, wavelengths)
det_out['CARI'] = cari_vi(hsi_sub, wavelengths)
det_out['CIrededge'] = cirededge_vi(hsi_sub, wavelengths)
det_out['CRI1'] = cri1_vi(hsi_sub, wavelengths)
det_out['CRI2'] = cri2_vi(hsi_sub, wavelengths)
det_out['EVI'] = evi_vi(hsi_sub, wavelengths)
det_out['MARI'] = mari_vi(hsi_sub, wavelengths)
det_out['MCARI'] = mcari_vi(hsi_sub, wavelengths)
det_out['MTCI'] = mtci_vi(hsi_sub, wavelengths)
det_out['NDII'] = ndii_vi(hsi_sub, wavelengths)
det_out['NDLI'] = ndli_vi(hsi_sub, wavelengths)
det_out['NDNI'] = ndni_vi(hsi_sub, wavelengths)
det_out['NDRE'] = ndre_vi(hsi_sub, wavelengths)
det_out['NDVI'] = ndvi_vi(hsi_sub, wavelengths)
det_out['NDWI'] = ndwi_vi(hsi_sub, wavelengths)
det_out['PRI'] = pri_vi(hsi_sub, wavelengths)
det_out['PSND ChlA'] = psnd_chlA_vi(hsi_sub, wavelengths)
det_out['PSND ChlB'] = psnd_chlB_vi(hsi_sub, wavelengths)
det_out['PSND Car'] = psnd_car_vi(hsi_sub, wavelengths)
det_out['PSRI'] = psri_vi(hsi_sub, wavelengths)
det_out['PSSR1'] = pssr1_vi(hsi_sub, wavelengths)
det_out['PSSR2'] = pssr2_vi(hsi_sub, wavelengths)
det_out['PSSR3'] = pssr3_vi(hsi_sub, wavelengths)
det_out['REP'] = rep_vi(hsi_sub, wavelengths)
det_out['RVSI'] = rvsi_vi(hsi_sub, wavelengths)
det_out['SIPI'] = sipi_vi(hsi_sub, wavelengths)
det_out['SR'] = sr_vi(hsi_sub, wavelengths)
det_out['VARI'] = vari_vi(hsi_sub, wavelengths)
det_out['VIgreen'] = vigreen_vi(hsi_sub, wavelengths)
det_out['WDVI'] = wdvi_vi(hsi_sub, wavelengths)
det_out['WBI'] = wbi_vi(hsi_sub, wavelengths)

# Visualization with ALL indices
plt.figure(figsize=(10, 15))
plt.subplots_adjust(hspace=0.5)
n_row = 6; n_col = 6
i = 1
for key, value in det_out.items():
    plt.subplot(n_row, n_col, i)
    plt.imshow(value)
    plt.yticks([])
    plt.xticks([])
    plt.title(key)
    i += 1
plt.show()

# Visualization with individual indices
# Figures are saved in /hsi_toolkit_py/spectral_indices/Results/
value_rgb = det_out['RGB']
dirout = os.getcwd()
for key, value in det_out.items():
    if key != 'RGB':
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(value_rgb)
        plt.colorbar(shrink=0.75) # only adding to make figures the same size
        plt.yticks([])
        plt.xticks([])
        plt.title('RGB')
        plt.subplot(1,2,2)
        plt.imshow(value)
        plt.colorbar(shrink=0.75)
        plt.yticks([])
        plt.xticks([])
        plt.title(key)
        plt.savefig((dirout + '/Results/'+key + '.png'), format='png',dpi=200)
        plt.close()
