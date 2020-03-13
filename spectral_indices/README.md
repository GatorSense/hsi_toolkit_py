# Gatorsense HSI Toolkit - Spectral Indices

Location: hsi_toolkit_py/spectral_indices

This code base contains common spectral indices for vegetation analysis. Also, referred to as Vegetation Indices (VIs) in the literature. These indices have been designed for hyperspectral imagery, but can be used on multi-spectral imagery also. Some indices were originally designed for multi-spectral data and narrow bands have been selected to calculate them. Those indices are noted below with *. Each function contains citation for original spectral index calculation. A demo (DEMO_spectral_indices.py) has been created that will calculate all VIs listed below. 

More information about VIs can be found in this book chapter: 
Roberts, D.A., Roth, K.L, Wetherley, E.B., Meerdink, S.K., & Perroy, R.L. (2019). Chapter 1: Hyperspectral Vegetation Indices, in: Hyperspectral Remote Sensing of Vegetation (second edition), CRC Press, New York.

## Current suite of VIs:
  * ari_vi: calculates the Anthocyanin Reflectance Index
  * cai_vi: calculates the Cellulose Absorption Index **
  * cari_vi: calculates the Chlorophyll Absorption in Reflectance Index
  * cri1_vi: calculates the Carotenoid Reflectance Index 1
  * cri2_vi: calculates the Carotenoid Reflectance Index 2
  * evi_vi: calculates the Enhanced Vegetation Index *
  * mcari_vi: calculates the Modified Anthocyanin Reflectance Index
  * mtci_vi: calculates the MERIS Terrestrial Chlorophyll Index
  * ndli_vi: calculates the Normalized Difference Lignin Index **
  * ndvi_vi: calculates the Normalized Difference Vegetation Index *
  * ndwi_ vi: calculates the Normalized Difference Water Index **
  * pri_vi: calculates the Photochemical Reflectance Index
  * psnd_chlA_vi: calculates the Pigment Sensitive Normalized Difference for Chlorophyll A
  * psnd_chlB_vi: calculates the Pigment Sensitive Normalized Difference for Chlorophyll B
  * psnd_car_vi: calculates the Pigment Sensitive Normalized Difference for Carotenoids 
  * psri_vi: calculates the Plant Senescence Reflectance Index
  * pssr1_vi: calculates the Pigment-Specific Spectral Ratio 1
  * pssr2_vi: calculates the Pigment-Specific Spectral Ratio 2
  * pssr3_vi: calculates the Pigment-Specific Spectral Ratio 3
  * rvsi_vi: calculates the Red-edge Vegetation Stress Index
  * sipi_vi: calculates the Structure Insensitive Pigment Index
  * sr_vi: calculates the Simple Ratio
  * wbi_vi: calculates the Water Band Index

* Spectral index original designed for multi-spectral data. We have selected narrow bands to calculate index based on hyperspectral literature. The citations are noted in each function description.

** This spectral index requires the shortwave infrared. 
	
Questions? Contact: Alina Zare, azare@ufl.edu
