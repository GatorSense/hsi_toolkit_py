# Information for using files in hsi_toolkit/util
## hsi_gui_mask
Use this file if you have binary masks for your hyperspectral image and have aligned the VNIR-E and SWIR data.
You will need to change some code from lines 50 to 95:
* Change file paths for SWIR data, VNIR-E data, and binary mask
* You may need to change how the data cube is uploaded
* You may need to change the RGB bands to construct the RGB image
Note: In this file, SWIR data is the aligned data and VNIR-E data is normal hsi data. You can edit the code to make it vice versa. 
It will take a minute for the window to show the plots and the RGB image. Please click on the 'Help!' button and read all of the functionalities before interacting with the GUI. 

## hsi_gui
Use this file if you want to visualize VNIR-E or SWIR hyperspectral data. 
You will need to change some of the code from lines 50-68:
* Change file paths for hyperspectral data and RGB image
* You may need to change how the data cube is uploaded
It will take a minute for the window to show the plots and the RGB image. Please click on the 'Help!' button and read all of the functionalities before interacting with the GUI.