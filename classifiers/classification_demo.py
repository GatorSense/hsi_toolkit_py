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
class_out['KNN'] = knn_classifier(hsi_img,train_data, k = 3)
