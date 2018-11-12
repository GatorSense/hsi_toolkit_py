import sys
from scipy.io import loadmat
import matplotlib.pyplot as plt
sys.path.append('../')
sys.path.append('../util/')
from classifiers import *
from get_RGB import get_RGB
"""
%function class_out = classification_demo(hsi_img, train_data, mask, wavelength)

Demo script that runs all classifiers in hsi_toolkit_py

Inputs:
	hsi_img - n_row x n_col x n_band hyperspectral image
	train_data - structure containing training data, train_data(i).Spectra
	  is matrix containing spectra from class i, train_data(i).name is the
	  name of the ith class label
	mask - binary image limiting detector operation to pixels where mask is true
	    if not present or empty, no mask restrictions are used
	wavelength - 1 x n_band vector listing wavelength values for hsi_img in nm

Outputs:
  class_out - dictionary of rgb image and classifier outputs

6/3/2018 - Alina Zare
10/2018 - Python Implementation by Yutai Zhou
"""
an_hsi_img_for_class_demo = loadmat('an_hsi_img_for_class_demo.mat')
hsi_img = an_hsi_img_for_class_demo['hsi_sub']
train_data = an_hsi_img_for_class_demo['train_data']
wavelength = an_hsi_img_for_class_demo['wavlength']

class_out = {}
class_out['RGB'] = get_RGB(hsi_img, wavelength)
class_out['KNN'] = knn_classifier(hsi_img, train_data, K = 3)
class_out['FKNN'] = fuzzy_knn_classifier(hsi_img, train_data, K = 3, m = 2.1)
class_out['PKNN'] = poss_knn_classifier(hsi_img, train_data, K = 3, eta = 0.01, m = 2.1)

# KNN Results
fig, ax = plt.subplots(1,2,figsize=(15, 8))
plt.subplots_adjust(hspace=.5)
plt.suptitle('KNN Results')

ax[0].imshow(class_out['RGB']); ax[0].set_title('RGB')
cax = ax[1].imshow(class_out['KNN']); ax[1].set_title('KNN')
cbar = fig.colorbar(cax, ticks=[i for i in range(len(train_data[0]))], orientation='vertical')
cbar.ax.set_yticklabels([i[0] for i in train_data[0][:]['name']])

# Fuzzy KNN Results
plt.figure(figsize=(10, 15))
plt.subplots_adjust(hspace=.5)
n_row = 2; n_col = 3
plt.suptitle('Fuzzy KNN Results')
plt.subplot(n_row, n_col, 1)
plt.imshow(class_out['RGB']); plt.title('RGB')
for i in range(0, len(train_data[0])):
	plt.subplot(n_row, n_col, i + 2)
	plt.imshow(class_out['FKNN'][:,:,i]);
	plt.title(train_data['name'][0][i][0])

# Possibilistic KNN Results
plt.figure(figsize=(10, 15))
plt.subplots_adjust(hspace=.5)
n_row = 2; n_col = 3
plt.suptitle('Possibilistic KNN Results')
plt.subplot(n_row, n_col, 1)
plt.imshow(class_out['RGB']); plt.title('RGB')
for i in range(0, len(train_data[0])):
	plt.subplot(n_row, n_col, i + 2)
	plt.imshow(class_out['PKNN'][:,:,i]);
	plt.title(train_data['name'][0][i][0])
plt.show()
