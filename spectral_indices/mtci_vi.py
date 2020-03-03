import numpy as np

def mtci_VI(imgData, wave, mask=0):
    """
    Function that calculates the MERIS Terrestrial Chlorophyll Index. 
    This functions uses wavelengths 753.75, 708.75, and 681.25 nm. The closest bands to these values will be used.
    Citation: Dash, J. and Curran, P.J. 2004. The MERIS terrestrial chlorophyll index, International Journal of Remote Sensing, 25(23), 5403–5413.
    INPUTS:
    1) imgData: an array of hyperspectral data either as 3D [n_row x n_col x n_band] or 2D [n_row x n_band]
    2) wave: an array of wavelengths in nanometers that correspond to the n_bands in imgData
    3) mask: OPTIONAL - a binary array (same size as imgData) that designates which pixels should be included in analysis. Pixels with 1 are used, while pixels with 0 are not.
    OUTPUTS:
    1) vi: the calculated spectral index value for each pixel either returned as [n_row x n_col x 1] or [n_row x 1]

    02/2020 - Susan Meerdink
    """
    # Find band indexes
    idx_753 = (np.abs(wave - 753.75)).argmin()
    idx_708 = (np.abs(wave - 708.75)).argmin()
    idx_681 = (np.abs(wave - 681.25)).argmin()
    print('Using bands ' + str(wave[idx_753]) +', '+ str(wave[idx_708])+', '+ str(wave[idx_681]))
    
    # 3D data, hyperspectral image, [n_row x n_col x n_band]
    if imgData.ndim > 2:
        data_753 = np.reshape(imgData[:,:,idx_753],[-1,1])
        data_708 = np.reshape(imgData[:,:,idx_708],[-1,1])
        data_681 = np.reshape(imgData[:,:,idx_681],[-1,1])
    
    # 2D data, flattened hyperspectral data, [n_row x n_band]
    else:
        data_753 = imgData[:,idx_753]
        data_708 = imgData[:,idx_708]
        data_681 = imgData[:,idx_681]
    
    # Calculate MTCI
    index = (data_753 - data_708)/(data_708 - data_681)
    
    # If data was 3D, reshape the index value back into 3D shape
    if imgData.ndim > 2:
        index = np.reshape(index,[imgData.shape[0],imgData.shape[1],1])
    
    if kwargs['mask'] is not None:
        idx_x, idx_y = np.where(mask==0)
        index[idx_x,idx_y] = 0
        
    return index
