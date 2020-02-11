import numpy as np

def evi_VI(imgData, wave, mask = None):
    """
    Function that calculates the Enhanced Vegetation Index. 
    This functions uses the 470, 650, and 860 nm bands. The closest bands to these values will be used.
    Citation: Huete, A.; Didan, K.; Miura, T.; Rodriguez, E.P.; Gao, X.; Ferreira, L.G. Overview of the radiometric and biophysical performance of the MODIS vegetation indices. Remote Sens. Environ. 2002, 83, 195–213.
    INPUTS:
    1) imgData: an array of hyperspectral data either as 3D [n_row x n_col x n_band] or 2D [n_row x n_band]
    2) wave: an array of wavelengths in nanometers that correspond to the n_bands in imgData
    3) mask: OPTIONAL - a binary array (same size as imgData) that designates which pixels should be included in analysis. Pixels with 1 are used, while pixels with 0 are not.
    OUTPUTS:
    1) vi: the calculated spectral index value for each pixel either returned as [n_row x n_col x 1] or [n_row x 1]

    02/2020 - Susan Meerdink
    """
    # Find band indexes
    idx_1 = (np.abs(wave - 470)).argmin()
    idx_2 = (np.abs(wave - 650)).argmin()
    idx_3 = (np.abs(wave - 860)).argmin()
    print('Using bands ' + str(wave[idx_1]) +', '+ str(wave[idx_2])+', '+str(wave[idx_3]))
    
    # 3D data, hyperspectral image, [n_row x n_col x n_band]
    if imgData.ndim > 2:
        data_1 = np.reshape(imgData[:,:,idx_1],[-1,1])
        data_2 = np.reshape(imgData[:,:,idx_2],[-1,1])
        data_3 = np.reshape(imgData[:,:,idx_3],[-1,1])
    
    # 2D data, flattened hyperspectral data, [n_row x n_band]
    else:
        data_1 = imgData[:,idx_1]
        data_2 = imgData[:,idx_2]
        data_3 = imgData[:,idx_3]
    
    # Calculate NDVI
    index = 2.5* ((data_3 - data_2)/(data_3 + 6*data_2 - 7.5*data1 + 1))
    
    # If data was 3D, reshape the index value back into 3D shape
    if imgData.ndim > 2:
        index = np.reshape(index,[imgData.shape[0],imgData.shape[1],1])
    
    if kwargs['mask'] is not None:
        idx_x, idx_y = np.where(mask==0)
        index[idx_x,idx_y] = 0
        
    return index
