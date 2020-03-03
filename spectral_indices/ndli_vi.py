import numpy as np

def ndli_VI(imgData, wave, mask=0):
    """
    Function that calculates the Normalized Difference Lignin Index. 
    This functions uses Shortwave Infrared wavelengths 1754 and 1680 nm. The closest bands to these values will be used.
    Citation: Serrano, L., Penuelas, J. and Ustin, S.L. 2002. Remote sensing of nitrogen and lignin in Mediterranean vegetation from AVIRIS data: Decomposing biochemical from structural signals, Remote Sensing of Environment, 81, 355–364
    INPUTS:
    1) imgData: an array of hyperspectral data either as 3D [n_row x n_col x n_band] or 2D [n_row x n_band]
    2) wave: an array of wavelengths in nanometers that correspond to the n_bands in imgData
    3) mask: OPTIONAL - a binary array (same size as imgData) that designates which pixels should be included in analysis. Pixels with 1 are used, while pixels with 0 are not.
    OUTPUTS:
    1) vi: the calculated spectral index value for each pixel either returned as [n_row x n_col x 1] or [n_row x 1]

    03/2020 - Susan Meerdink
    """
    # Find band indexes
    idx_1754 = (np.abs(wave - 1754)).argmin()
    idx_1680 = (np.abs(wave - 1680)).argmin()
    print('Using bands ' + str(wave[idx_1754]) +', '+ str(wave[idx_1680]))
    
    # 3D data, hyperspectral image, [n_row x n_col x n_band]
    if imgData.ndim > 2:
        data_1754 = np.reshape(imgData[:,:,idx_1754],[-1,1])
        data_1680 = np.reshape(imgData[:,:,idx_1680],[-1,1])
    
    # 2D data, flattened hyperspectral data, [n_row x n_band]
    else:
        data_1754 = imgData[:,idx_1754]
        data_1680 = imgData[:,idx_1680]
    
    # Calculate NDLI
    index = (np.log(1/data_1754) - np.log(1/data_1680))/(np.log(1/data_1754) + np.log(1/data_1680))
    
    # If data was 3D, reshape the index value back into 3D shape
    if imgData.ndim > 2:
        index = np.reshape(index,[imgData.shape[0],imgData.shape[1],1])
    
    if kwargs['mask'] is not None:
        idx_x, idx_y = np.where(mask==0)
        index[idx_x,idx_y] = 0
        
    return index
