import numpy as np

def ari_VI(imgData, wave, mask=0):
    """
    Function that calculates the Anthocyanin Reflectance Index. 
    This functions uses red band 550 nm and Near Infrared Band 700 nm. The closest bands to these values will be used.
    Citation: Gitelson, A.A., Merzlyak, M.N. and Chivkunova, O.B. 2001. Optical properties and non-destructive estimationof anthocyanin content in plant leaves, Photochemistry and Photobiology, 74(1), 38–45
    INPUTS:
    1) imgData: an array of hyperspectral data either as 3D [n_row x n_col x n_band] or 2D [n_row x n_band]
    2) wave: an array of wavelengths in nanometers that correspond to the n_bands in imgData
    3) mask: OPTIONAL - a binary array (same size as imgData) that designates which pixels should be included in analysis. Pixels with 1 are used, while pixels with 0 are not.
    OUTPUTS:
    1) vi: the calculated spectral index value for each pixel either returned as [n_row x n_col x 1] or [n_row x 1]

    03/2020 - Susan Meerdink
    """
    # Find band indexes
    idx_550 = (np.abs(wave - 550)).argmin()
    idx_700 = (np.abs(wave - 700)).argmin()
    print('Using bands ' + str(wave[idx_550]) +', '+ str(wave[idx_700]))
    
    # 3D data, hyperspectral image, [n_row x n_col x n_band]
    if imgData.ndim > 2:
        data_550 = np.reshape(imgData[:,:,idx_550],[-1,1])
        data_700 = np.reshape(imgData[:,:,idx_700],[-1,1])
    
    # 2D data, flattened hyperspectral data, [n_row x n_band]
    else:
        data_550 = imgData[:,idx_550]
        data_700 = imgData[:,idx_700]
    
    # Calculate ARI
    index = (1/data_550) - (1/data_700)
    
    # If data was 3D, reshape the index value back into 3D shape
    if imgData.ndim > 2:
        index = np.reshape(index,[imgData.shape[0],imgData.shape[1],1])
    
    if kwargs['mask'] is not None:
        idx_x, idx_y = np.where(mask==0)
        index[idx_x,idx_y] = 0
        
    return index
