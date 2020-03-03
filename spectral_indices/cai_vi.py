import numpy as np

def cai_VI(imgData, wave, mask=0):
    """
    Function that calculates the Cellulose Absorption Index. 
    This functions uses the shortwave infrared (SWIR) bands 2019, 2206, and 2109 nm. The closest bands to these values will be used.
    Citation: Daughtry, C.S.T. 2001. Discriminating crop residues from soil by shortwave infrared reflectance, Agronomy Journal, 93, 125–131.
    INPUTS:
    1) imgData: an array of hyperspectral data either as 3D [n_row x n_col x n_band] or 2D [n_row x n_band]
    2) wave: an array of wavelengths in nanometers that correspond to the n_bands in imgData
    3) mask: OPTIONAL - a binary array (same size as imgData) that designates which pixels should be included in analysis. Pixels with 1 are used, while pixels with 0 are not.
    OUTPUTS:
    1) vi: the calculated spectral index value for each pixel either returned as [n_row x n_col x 1] or [n_row x 1]

    03/2020 - Susan Meerdink
    """
    # Find band indexes
    idx_2019 = (np.abs(wave - 2019)).argmin()
    idx_2206 = (np.abs(wave - 2206)).argmin()
    idx_2109 = (np.abs(wave - 2109)).argmin()
    print('Using bands ' + str(wave[idx_2019]) +', '+ str(wave[idx_2206])+', '+ str(wave[idx_2109]))
    
    # 3D data, hyperspectral image, [n_row x n_col x n_band]
    if imgData.ndim > 2:
        data_2019 = np.reshape(imgData[:,:,idx_2019],[-1,1])
        data_2206 = np.reshape(imgData[:,:,idx_2206],[-1,1])
        data_2109 = np.reshape(imgData[:,:,idx_2109],[-1,1])
    
    # 2D data, flattened hyperspectral data, [n_row x n_band]
    else:
        data_2019 = imgData[:,idx_2019]
        data_2206 = imgData[:,idx_2206]
        data_2109 = imgData[:,idx_2109]
    
    # Calculate CAI
    index = 0.5*(data_2019 + data_2206) - data_2109
    
    # If data was 3D, reshape the index value back into 3D shape
    if imgData.ndim > 2:
        index = np.reshape(index,[imgData.shape[0],imgData.shape[1],1])
    
    if kwargs['mask'] is not None:
        idx_x, idx_y = np.where(mask==0)
        index[idx_x,idx_y] = 0
        
    return index
