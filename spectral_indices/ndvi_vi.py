import numpy as np

def ndvi_VI(imgData, wave, mask=0):
    """
    Function that calculates the Normalized Difference Vegetation Index. 
    This functions uses red band 670 nm and Near Infrared Band 860 nm. The closest bands to these values will be used.
    Citation: Rouse, J., Jr.; Haas, R.H.; Schell, J.A.; Deering, D.W. Monitoring Vegetation Systems in the Great Plains with ERTS; Third ERTS Symposium, NASA SP-351; NASA: Washington, DC, USA, 1974.
    INPUTS:
    1) imgData: an array of hyperspectral data either as 3D [n_row x n_col x n_band] or 2D [n_row x n_band]
    2) wave: an array of wavelengths in nanometers that correspond to the n_bands in imgData
    3) mask: OPTIONAL - a binary array (same size as imgData) that designates which pixels should be included in analysis. Pixels with 1 are used, while pixels with 0 are not.
    OUTPUTS:
    1) vi: the calculated spectral index value for each pixel either returned as [n_row x n_col x 1] or [n_row x 1]

    02/2020 - Susan Meerdink
    """
    # Find band indexes
    idx_red = (np.abs(wave - 670)).argmin()
    idx_nir = (np.abs(wave - 860)).argmin()
    print('Using bands ' + str(wave[idx_red]) +', '+ str(wave[idx_nir]))
    
    # 3D data, hyperspectral image, [n_row x n_col x n_band]
    if imgData.ndim > 2:
        data_red = np.reshape(imgData[:,:,idx_red],[-1,1])
        data_nir = np.reshape(imgData[:,:,idx_nir],[-1,1])
    
    # 2D data, flattened hyperspectral data, [n_row x n_band]
    else:
        data_red = imgData[:,idx_red]
        data_nir = imgData[:,idx_nir]
    
    # Calculate NDVI
    index = (data_nir - data_red)/(data_nir + data_red)
    
    # If data was 3D, reshape the index value back into 3D shape
    if imgData.ndim > 2:
        index = np.reshape(index,[imgData.shape[0],imgData.shape[1],1])
    
    if kwargs['mask'] is not None:
        idx_x, idx_y = np.where(mask==0)
        index[idx_x,idx_y] = 0
        
    return index
