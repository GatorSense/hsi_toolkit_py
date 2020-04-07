# Gatorsense HSI Toolkit - Spectral Indices

Location: hsi_toolkit_py/spectral_indices

This code base contains common spectral indices for vegetation analysis. Also, referred to as Vegetation Indices (VIs) in the literature. These indices have been designed for hyperspectral imagery, but can be used on multi-spectral imagery also. Some indices were originally designed for multi-spectral data and narrow bands have been selected to calculate them. Those indices are noted below with *. Each function contains citation for original spectral index calculation. A demo (DEMO_spectral_indices.py) has been created that will calculate all VIs listed below. 

More information about VIs can be found in this book chapter: 
Roberts, D.A., Roth, K.L, Wetherley, E.B., Meerdink, S.K., & Perroy, R.L. (2019). Chapter 1: Hyperspectral Vegetation Indices, in: Hyperspectral Remote Sensing of Vegetation (second edition), CRC Press, New York.

## Inputs:
Each function takes similar inputs starting with hyperspectral image, wavelengths (in nanometers or nm), an optional mask (if you want to exclude pixels from analysis), and an optional band index. All functions will calculate the index based on literature suggested bands, but IF the user wants to input their own bands that is possible by specifying which band index should be used with the optional band variable. 

## Demo:
This repo contains a demo to run all spectral indices that are available. The demo hyperspectral image used is from the AVIRIS sensor (https://aviris.jpl.nasa.gov/) collected on April 16, 2014 over the Santa Barbara area. Specifically the image contains the La Cumbre Country Club and features a golf course with lake and residental areas surrounding. This should provide a mix of vegetation - trees and irrigated grasses.The demo will calculate all the indices, display them together for the user, and will save individual png files of each index with the RGB image in the Results folder. Keep in mind, the result images that are generated use the default value range and may need to be adjust to emphasize the range of values for vegetation. 

## Current suite of VIs:
  * aci_vi: calculates the Anthocyanin Content Index **
  * ari_vi: calculates the Anthocyanin Reflectance Index
  * arvi_vi: calculates the Atmospherically Resistant Vegetation Index **
  * cai_vi: calculates the Cellulose Absorption Index ***
  * cari_vi: calculates the Chlorophyll Absorption in Reflectance Index
  * cirededge_vi: calculates the Chlorophyll Index Red Edge **
  * cri1_vi: calculates the Carotenoid Reflectance Index 1
  * cri2_vi: calculates the Carotenoid Reflectance Index 2
  * evi_vi: calculates the Enhanced Vegetation Index **
  * mari_vi: calculates the Modified Anthocyanin Reflectance Index
  * mcari_vi: calculates the Modified Anthocyanin Reflectance Index
  * msi_vi: calculates the Moisture Stress Index ** & ***
  * mtci_vi: calculates the MERIS Terrestrial Chlorophyll Index
  * ndii_vi: calculates the Normalized Difference Infrared Index ** & ***
  * ndli_vi: calculates the Normalized Difference Lignin Index ***
  * ndni_vi: calculates the Normalized Difference Nitrogen Index ***
  * ndre_vi: calculates the Normalized Difference Red Edge Index 
  * ndvi_vi: calculates the Normalized Difference Vegetation Index **
  * ndwi_vi: calculates the Normalized Difference Water Index ***
  * pri_vi: calculates the Photochemical Reflectance Index
  * psnd_chlA_vi: calculates the Pigment Sensitive Normalized Difference for Chlorophyll A
  * psnd_chlB_vi: calculates the Pigment Sensitive Normalized Difference for Chlorophyll B
  * psnd_car_vi: calculates the Pigment Sensitive Normalized Difference for Carotenoids 
  * psri_vi: calculates the Plant Senescence Reflectance Index
  * pssr1_vi: calculates the Pigment-Specific Spectral Ratio 1
  * pssr2_vi: calculates the Pigment-Specific Spectral Ratio 2
  * pssr3_vi: calculates the Pigment-Specific Spectral Ratio 3
  * rep_vi: calculates the Red-edge Position
  * rgri_vi: calculates the Red/Green Ratio Index **
  * rvsi_vi: calculates the Red-edge Vegetation Stress Index
  * savi_vi: calculates the Soil Adjusted Vegetation Index
  * sipi_vi: calculates the Structure Insensitive Pigment Index
  * sr_vi: calculates the Simple Ratio
  * vari_vi: calculates the Visible Atmospherically Resistant Index **
  * vigreen_vi: calculates the Vegetation Index using green band **
  * wdvi_vi: calculates the Weighted Difference Vegetation Index **
  * wbi_vi: calculates the Water Band Index

** Spectral index original designed for multi-spectral data. We have selected narrow bands to calculate index based on hyperspectral literature. The citations with used hyperspectral bands are noted in each function description.

*** This spectral index requires the shortwave infrared (1200 - 2500 nm). Not all hyperspectral systems measure the entire visible shortwave infrared spectrum. 
	
Questions? Contact: Alina Zare, azare@ufl.edu
