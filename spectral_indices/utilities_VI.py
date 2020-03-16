import numpy as np

def ari_vi(imgData, wave, mask=0):
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
    print('Calls for bands 550 and 700 nm. Using bands ' + str(wave[idx_550]) +', '+ str(wave[idx_700]))
    
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
        index = np.reshape(index,[imgData.shape[0],imgData.shape[1]])
    
    if isinstance(mask, int) is False:
        idx_x, idx_y = np.where(mask==0)
        index[idx_x,idx_y] = 0
        
    return index

def cai_vi(imgData, wave, mask=0):
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
    # Check that data has shortwave infrared data
    if wave[-1] < 2200:
        raise Exception('Data does not have Shortwave Infrared Bands and CAI cannot be calculated.')
    
    # Find band indexes
    idx_2019 = (np.abs(wave - 2019)).argmin()
    idx_2206 = (np.abs(wave - 2206)).argmin()
    idx_2109 = (np.abs(wave - 2109)).argmin()
    print('Calls for bands 2019, 2206, and 2109 nm. Using bands ' + str(wave[idx_2019]) +', '+ str(wave[idx_2206])+', '+ str(wave[idx_2109]))
    
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
        index = np.reshape(index,[imgData.shape[0],imgData.shape[1]])
    
    if isinstance(mask, int) is False:
        idx_x, idx_y = np.where(mask==0)
        index[idx_x,idx_y] = 0
        
    return index

def cari_vi(imgData, wave, mask=0):
    """
    Function that calculates the Chlorophyll Absorption in Reflectance Index. 
    This functions uses bands 550, 670, and 700 nm. The closest bands to these values will be used.
    Citation: Daughtry, C.S.T., Walthall, C.L. Kim, M.S., de Colstoun, E.B. and McMurtrey, J.E. 2000. Estimating corn leaf chlorophyll concentration from leaf and canopy reflectance. Remote Sensing of Environment, 74,229–239.
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
    idx_670 = (np.abs(wave - 670)).argmin()
    idx_700 = (np.abs(wave - 700)).argmin()
    print('Calls for bands 550, 670, and 700 nm. Using bands ' + str(wave[idx_550]) +', '+ str(wave[idx_670])+', '+ str(wave[idx_700]))
    
    # 3D data, hyperspectral image, [n_row x n_col x n_band]
    if imgData.ndim > 2:
        data_550 = np.reshape(imgData[:,:,idx_550],[-1,1])
        data_670 = np.reshape(imgData[:,:,idx_670],[-1,1])
        data_700 = np.reshape(imgData[:,:,idx_700],[-1,1])
    
    # 2D data, flattened hyperspectral data, [n_row x n_band]
    else:
        data_550 = imgData[:,idx_550]
        data_670 = imgData[:,idx_670]
        data_700 = imgData[:,idx_700]
    
    # Calculate CARI
    index = ((data_700 - data_670) - 0.2*(data_700 - data_550))
    
    # If data was 3D, reshape the index value back into 3D shape
    if imgData.ndim > 2:
        index = np.reshape(index,[imgData.shape[0],imgData.shape[1]])
    
    if isinstance(mask, int) is False:
        idx_x, idx_y = np.where(mask==0)
        index[idx_x,idx_y] = 0
        
    return index

def cri1_vi(imgData, wave, mask=0):
    """
    Function that calculates the Carotenoid Reflectance Index 1. There is a Carotenoid Reflectance Index 2.  
    This functions uses bands 510 and 550 nm. The closest bands to these values will be used.
    Citation: Gitelson, A.A., Zur, Y., Chivkunova, O.B. and Merzlyak, M.N. 2002. Assessing Carotenoid Content in Plant Leaves with Reflectance Spectroscopy, Photochemistry and Photobiology, 75(3), 272–281
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
    idx_510 = (np.abs(wave - 510)).argmin()
    print('Calls for bands 510 and 550 nm. Using bands ' + str(wave[idx_550]) +', '+ str(wave[idx_510]))
    
    # 3D data, hyperspectral image, [n_row x n_col x n_band]
    if imgData.ndim > 2:
        data_550 = np.reshape(imgData[:,:,idx_550],[-1,1])
        data_510 = np.reshape(imgData[:,:,idx_510],[-1,1])
    
    # 2D data, flattened hyperspectral data, [n_row x n_band]
    else:
        data_550 = imgData[:,idx_550]
        data_510 = imgData[:,idx_510]
    
    # Calculate CRI 1
    index = (1/data_510) - (1/data_550)
    
    # If data was 3D, reshape the index value back into 3D shape
    if imgData.ndim > 2:
        index = np.reshape(index,[imgData.shape[0],imgData.shape[1]])
    
    if isinstance(mask, int) is False:
        idx_x, idx_y = np.where(mask==0)
        index[idx_x,idx_y] = 0
        
    return index

def cri2_vi(imgData, wave, mask=0):
    """
    Function that calculates the Carotenoid Reflectance Index 2. There is a Carotenoid Reflectance Index 1.  
    This functions uses bands 700 and 700 nm. The closest bands to these values will be used.
    Citation: Gitelson, A.A., Zur, Y., Chivkunova, O.B. and Merzlyak, M.N. 2002. Assessing Carotenoid Content in Plant Leaves with Reflectance Spectroscopy, Photochemistry and Photobiology, 75(3), 272–281
    INPUTS:
    1) imgData: an array of hyperspectral data either as 3D [n_row x n_col x n_band] or 2D [n_row x n_band]
    2) wave: an array of wavelengths in nanometers that correspond to the n_bands in imgData
    3) mask: OPTIONAL - a binary array (same size as imgData) that designates which pixels should be included in analysis. Pixels with 1 are used, while pixels with 0 are not.
    OUTPUTS:
    1) vi: the calculated spectral index value for each pixel either returned as [n_row x n_col x 1] or [n_row x 1]

    03/2020 - Susan Meerdink
    """
    # Find band indexes
    idx_510 = (np.abs(wave - 510)).argmin()
    idx_700 = (np.abs(wave - 700)).argmin()
    print('Calls for bands 700 and 700 nm. Using bands ' + str(wave[idx_550]) +', '+ str(wave[idx_700]))
    
    # 3D data, hyperspectral image, [n_row x n_col x n_band]
    if imgData.ndim > 2:
        data_510 = np.reshape(imgData[:,:,idx_510],[-1,1])
        data_700 = np.reshape(imgData[:,:,idx_700],[-1,1])
    
    # 2D data, flattened hyperspectral data, [n_row x n_band]
    else:
        data_510 = imgData[:,idx_510]
        data_700 = imgData[:,idx_700]
    
    # Calculate CRI 2
    index = (1/data_510) - (1/data_700)
    
    # If data was 3D, reshape the index value back into 3D shape
    if imgData.ndim > 2:
        index = np.reshape(index,[imgData.shape[0],imgData.shape[1]])
    
    if isinstance(mask, int) is False:
        idx_x, idx_y = np.where(mask==0)
        index[idx_x,idx_y] = 0
        
    return index

def evi_vi(imgData, wave, mask=0):
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
    print('Calls for bands 470, 650, and 860 nm. Using bands ' + str(wave[idx_1]) +', '+ str(wave[idx_2])+', '+str(wave[idx_3]))
    
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
    index = 2.5* ((data_3 - data_2)/(data_3 + 6*data_2 - 7.5*data_1 + 1))
    
    # If data was 3D, reshape the index value back into 3D shape
    if imgData.ndim > 2:
        index = np.reshape(index,[imgData.shape[0],imgData.shape[1]])
    
    if isinstance(mask, int) is False:
        idx_x, idx_y = np.where(mask==0)
        index[idx_x,idx_y] = 0
        
    return index

def mcari_vi(imgData, wave, mask=0):
    """
    Function that calculates the Modified Anthocyanin Reflectance Index.  
    This functions uses bands 550, 670, and 700 nm. The closest bands to these values will be used.
    Citation: Daughtry, C.S.T., Walthall, C.L. Kim, M.S., de Colstoun, E.B. and McMurtrey, J.E. 2000. Estimating corn leaf chlorophyll concentration from leaf and canopy reflectance. Remote Sensing of Environment, 74,229–239.
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
    idx_670 = (np.abs(wave - 670)).argmin()
    idx_700 = (np.abs(wave - 700)).argmin()
    print('Calls for bands 550, 670, and 700 nm. Using bands ' + str(wave[idx_550]) +', '+ str(wave[idx_670])+', '+ str(wave[idx_700]))
    
    # 3D data, hyperspectral image, [n_row x n_col x n_band]
    if imgData.ndim > 2:
        data_550 = np.reshape(imgData[:,:,idx_550],[-1,1])
        data_670 = np.reshape(imgData[:,:,idx_670],[-1,1])
        data_700 = np.reshape(imgData[:,:,idx_700],[-1,1])
    
    # 2D data, flattened hyperspectral data, [n_row x n_band]
    else:
        data_550 = imgData[:,idx_550]
        data_670 = imgData[:,idx_670]
        data_700 = imgData[:,idx_700]
    
    # Calculate MCARI
    index = ((data_700 - data_670) - 0.2*(data_700 - data_550))*(data_700/data_670)
    
    # If data was 3D, reshape the index value back into 3D shape
    if imgData.ndim > 2:
        index = np.reshape(index,[imgData.shape[0],imgData.shape[1]])
    
    if isinstance(mask, int) is False:
        idx_x, idx_y = np.where(mask==0)
        index[idx_x,idx_y] = 0
        
    return index

def mtci_vi(imgData, wave, mask=0):
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
    print('Calls for bands 753.75, 708.75, and 681.25 nm. Using bands ' + str(wave[idx_753]) +', '+ str(wave[idx_708])+', '+ str(wave[idx_681]))
    
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
        index = np.reshape(index,[imgData.shape[0],imgData.shape[1]])
    
    if isinstance(mask, int) is False:
        idx_x, idx_y = np.where(mask==0)
        index[idx_x,idx_y] = 0
        
    return index

def ndli_vi(imgData, wave, mask=0):
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
    # Check that data has shortwave infrared data
    if wave[-1] < 1800:
        raise Exception('Data does not have Shortwave Infrared Bands and CAI cannot be calculated.')
    
    # Find band indexes
    idx_1754 = (np.abs(wave - 1754)).argmin()
    idx_1680 = (np.abs(wave - 1680)).argmin()
    print('Calls for bands 1754 and 1680 nm. Using bands ' + str(wave[idx_1754]) +', '+ str(wave[idx_1680]))
    
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
        index = np.reshape(index,[imgData.shape[0],imgData.shape[1]])
    
    if isinstance(mask, int) is False:
        idx_x, idx_y = np.where(mask==0)
        index[idx_x,idx_y] = 0
        
    return index

def ndvi_vi(imgData, wave, mask=0):
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
    print('Calls for bands 670 and 860 nm. Using bands ' + str(wave[idx_red]) +', '+ str(wave[idx_nir]))
    
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
        index = np.reshape(index,[imgData.shape[0],imgData.shape[1]])
    
    if isinstance(mask, int) is False:
        idx_x, idx_y = np.where(mask==0)
        index[idx_x,idx_y] = 0
        
    return index

def ndwi_vi(imgData, wave, mask=0):
    """
    Function that calculates the Normalized Difference Water Index. 
    This functions uses shortwave infrared bands - 860 and 1240 nm. The closest bands to these values will be used.
    Citation: Gao, B. 1996. NDWI: A normalized difference water index for remote sensing of vegetation liquid water from space, Remote Sensing of Environment, 58, 257–266.
    INPUTS:
    1) imgData: an array of hyperspectral data either as 3D [n_row x n_col x n_band] or 2D [n_row x n_band]
    2) wave: an array of wavelengths in nanometers that correspond to the n_bands in imgData
    3) mask: OPTIONAL - a binary array (same size as imgData) that designates which pixels should be included in analysis. Pixels with 1 are used, while pixels with 0 are not.
    OUTPUTS:
    1) vi: the calculated spectral index value for each pixel either returned as [n_row x n_col x 1] or [n_row x 1]

    02/2020 - Susan Meerdink
    """
    # Check that data has shortwave infrared data
    if wave[-1] < 1300:
        raise Exception('Data does not have Shortwave Infrared Bands and CAI cannot be calculated.')
    
    # Find band indexes
    idx_860 = (np.abs(wave - 860)).argmin()
    idx_1240 = (np.abs(wave - 1240)).argmin()
    print('Calls for bands 860 and 1240 nm. Using bands ' + str(wave[idx_860]) +', '+ str(wave[idx_1240]))
    
    # 3D data, hyperspectral image, [n_row x n_col x n_band]
    if imgData.ndim > 2:
        data_860 = np.reshape(imgData[:,:,idx_860],[-1,1])
        data_1240 = np.reshape(imgData[:,:,idx_1240],[-1,1])
    
    # 2D data, flattened hyperspectral data, [n_row x n_band]
    else:
        data_860 = imgData[:,idx_860]
        data_1240 = imgData[:,idx_1240]
    
    # Calculate NDWI
    index = (data_860 - data_1240)/(data_860 + data_1240)
    
    # If data was 3D, reshape the index value back into 3D shape
    if imgData.ndim > 2:
        index = np.reshape(index,[imgData.shape[0],imgData.shape[1]])
    
    if isinstance(mask, int) is False:
        idx_x, idx_y = np.where(mask==0)
        index[idx_x,idx_y] = 0
        
    return index

def pri_vi(imgData, wave, mask=0):
    """
    Function that calculates the Photochemical Reflectance Index. 
    This functions uses bands 531 and 570 nm. The closest bands to these values will be used.
    Citation: Gamon J.A., Serrano, L. and Surfus, J.S. 1997. The photochemical reflectance index: An optical indicator of photosynthetic radiation-use efficiency across species, functional types, and nutrient levels, Oecologia, 112, 492–501.
    INPUTS:
    1) imgData: an array of hyperspectral data either as 3D [n_row x n_col x n_band] or 2D [n_row x n_band]
    2) wave: an array of wavelengths in nanometers that correspond to the n_bands in imgData
    3) mask: OPTIONAL - a binary array (same size as imgData) that designates which pixels should be included in analysis. Pixels with 1 are used, while pixels with 0 are not.
    OUTPUTS:
    1) vi: the calculated spectral index value for each pixel either returned as [n_row x n_col x 1] or [n_row x 1]

    03/2020 - Susan Meerdink
    """
    
    # Find band indexes
    idx_531 = (np.abs(wave - 531)).argmin()
    idx_570 = (np.abs(wave - 570)).argmin()
    print('Calls for bands 531 and 570 nm. Using bands ' + str(wave[idx_531]) +', '+ str(wave[idx_570]))
    
    # 3D data, hyperspectral image, [n_row x n_col x n_band]
    if imgData.ndim > 2:
        data_531 = np.reshape(imgData[:,:,idx_531],[-1,1])
        data_570 = np.reshape(imgData[:,:,idx_570],[-1,1])
    
    # 2D data, flattened hyperspectral data, [n_row x n_band]
    else:
        data_531 = imgData[:,idx_531]
        data_570 = imgData[:,idx_570]
    
    # Calculate PRI
    index = (data_531 - data_570)/(data_531 + data_570)
    
    # If data was 3D, reshape the index value back into 3D shape
    if imgData.ndim > 2:
        index = np.reshape(index,[imgData.shape[0],imgData.shape[1]])
    
    if isinstance(mask, int) is False:
        idx_x, idx_y = np.where(mask==0)
        index[idx_x,idx_y] = 0
        
    return index

def psnd_chlA_vi(imgData, wave, mask=0):
    """
    Function that calculates the Pigment Sensitive Normalized Difference for Chlorophyll A. 
    This functions uses bands 675 and 800 nm. The closest bands to these values will be used.
    Citation: Blackburn, G.A. 1998. Spectral indices for estimating photosynthetic pigment concentrations: A test using senescent tree leaves, International Journal of Remote Sensing, 19, 657–675.
    INPUTS:
    1) imgData: an array of hyperspectral data either as 3D [n_row x n_col x n_band] or 2D [n_row x n_band]
    2) wave: an array of wavelengths in nanometers that correspond to the n_bands in imgData
    3) mask: OPTIONAL - a binary array (same size as imgData) that designates which pixels should be included in analysis. Pixels with 1 are used, while pixels with 0 are not.
    OUTPUTS:
    1) vi: the calculated spectral index value for each pixel either returned as [n_row x n_col x 1] or [n_row x 1]

    03/2020 - Susan Meerdink
    """
    
    # Find band indexes
    idx_675 = (np.abs(wave - 675)).argmin()
    idx_800 = (np.abs(wave - 800)).argmin()
    print('Calls for bands 675 and 800 nm. Using bands ' + str(wave[idx_675]) +', '+ str(wave[idx_800]))
    
    # 3D data, hyperspectral image, [n_row x n_col x n_band]
    if imgData.ndim > 2:
        data_675 = np.reshape(imgData[:,:,idx_675],[-1,1])
        data_800 = np.reshape(imgData[:,:,idx_800],[-1,1])
    
    # 2D data, flattened hyperspectral data, [n_row x n_band]
    else:
        data_675 = imgData[:,idx_675]
        data_800 = imgData[:,idx_800]
    
    # Calculate PSND ChlA
    index = (data_800 - data_675)/(data_800 + data_675)
    
    # If data was 3D, reshape the index value back into 3D shape
    if imgData.ndim > 2:
        index = np.reshape(index,[imgData.shape[0],imgData.shape[1]])
    
    if isinstance(mask, int) is False:
        idx_x, idx_y = np.where(mask==0)
        index[idx_x,idx_y] = 0
        
    return index

def psnd_chlB_vi(imgData, wave, mask=0):
    """
    Function that calculates the Pigment Sensitive Normalized Difference for Chlorophyll B. 
    This functions uses bands 650 and 800 nm. The closest bands to these values will be used.
    Citation: Blackburn, G.A. 1998. Spectral indices for estimating photosynthetic pigment concentrations: A test using senescent tree leaves, International Journal of Remote Sensing, 19, 657–675.
    INPUTS:
    1) imgData: an array of hyperspectral data either as 3D [n_row x n_col x n_band] or 2D [n_row x n_band]
    2) wave: an array of wavelengths in nanometers that correspond to the n_bands in imgData
    3) mask: OPTIONAL - a binary array (same size as imgData) that designates which pixels should be included in analysis. Pixels with 1 are used, while pixels with 0 are not.
    OUTPUTS:
    1) vi: the calculated spectral index value for each pixel either returned as [n_row x n_col x 1] or [n_row x 1]

    03/2020 - Susan Meerdink
    """
    
    # Find band indexes
    idx_650 = (np.abs(wave - 650)).argmin()
    idx_800 = (np.abs(wave - 800)).argmin()
    print('Calls for bands 650 and 800 nm. Using bands ' + str(wave[idx_650]) +', '+ str(wave[idx_800]))
    
    # 3D data, hyperspectral image, [n_row x n_col x n_band]
    if imgData.ndim > 2:
        data_650 = np.reshape(imgData[:,:,idx_650],[-1,1])
        data_800 = np.reshape(imgData[:,:,idx_800],[-1,1])
    
    # 2D data, flattened hyperspectral data, [n_row x n_band]
    else:
        data_650 = imgData[:,idx_650]
        data_800 = imgData[:,idx_800]
    
    # Calculate PSND ChlB
    index = (data_800 - data_650)/(data_800 + data_650)
    
    # If data was 3D, reshape the index value back into 3D shape
    if imgData.ndim > 2:
        index = np.reshape(index,[imgData.shape[0],imgData.shape[1]])
    
    if isinstance(mask, int) is False:
        idx_x, idx_y = np.where(mask==0)
        index[idx_x,idx_y] = 0
        
    return index                              

def psnd_car_vi(imgData, wave, mask=0):
    """
    Function that calculates the Pigment Sensitive Normalized Difference for Carotenoids 
    This functions uses bands 500 and 800 nm. The closest bands to these values will be used.
    Citation: Blackburn, G.A. 1998. Spectral indices for estimating photosynthetic pigment concentrations: A test using senescent tree leaves, International Journal of Remote Sensing, 19, 657–675.
    INPUTS:
    1) imgData: an array of hyperspectral data either as 3D [n_row x n_col x n_band] or 2D [n_row x n_band]
    2) wave: an array of wavelengths in nanometers that correspond to the n_bands in imgData
    3) mask: OPTIONAL - a binary array (same size as imgData) that designates which pixels should be included in analysis. Pixels with 1 are used, while pixels with 0 are not.
    OUTPUTS:
    1) vi: the calculated spectral index value for each pixel either returned as [n_row x n_col x 1] or [n_row x 1]

    03/2020 - Susan Meerdink
    """
    
    # Find band indexes
    idx_500 = (np.abs(wave - 500)).argmin()
    idx_800 = (np.abs(wave - 800)).argmin()
    print('Calls for bands 500 and 800 nm. Using bands ' + str(wave[idx_500]) +', '+ str(wave[idx_800]))
    
    # 3D data, hyperspectral image, [n_row x n_col x n_band]
    if imgData.ndim > 2:
        data_500 = np.reshape(imgData[:,:,idx_500],[-1,1])
        data_800 = np.reshape(imgData[:,:,idx_800],[-1,1])
    
    # 2D data, flattened hyperspectral data, [n_row x n_band]
    else:
        data_500 = imgData[:,idx_500]
        data_800 = imgData[:,idx_800]
    
    # Calculate PSND Car
    index = (data_800 - data_500)/(data_800 + data_500)
    
    # If data was 3D, reshape the index value back into 3D shape
    if imgData.ndim > 2:
        index = np.reshape(index,[imgData.shape[0],imgData.shape[1]])
    
    if isinstance(mask, int) is False:
        idx_x, idx_y = np.where(mask==0)
        index[idx_x,idx_y] = 0
        
    return index  

def psri_vi(imgData, wave, mask=0):
    """
    Function that calculates the Plant Senescence Reflectance Index.
    This functions uses bands 500, 678, and 750 nm. The closest bands to these values will be used.
    Citation: Merzlyak, M.N., Gitelson, A.A., Chivkunova, O.B. and Rakitin, Y. 1999. Non-destructive optical detection of pigment changes during leaf senescence and fruit ripening, Physiologia Plantarum, 105, 135–141.
    INPUTS:
    1) imgData: an array of hyperspectral data either as 3D [n_row x n_col x n_band] or 2D [n_row x n_band]
    2) wave: an array of wavelengths in nanometers that correspond to the n_bands in imgData
    3) mask: OPTIONAL - a binary array (same size as imgData) that designates which pixels should be included in analysis. Pixels with 1 are used, while pixels with 0 are not.
    OUTPUTS:
    1) vi: the calculated spectral index value for each pixel either returned as [n_row x n_col x 1] or [n_row x 1]

    03/2020 - Susan Meerdink
    """
    
    # Find band indexes
    idx_500 = (np.abs(wave - 500)).argmin()
    idx_678 = (np.abs(wave - 678)).argmin()                          
    idx_750 = (np.abs(wave - 750)).argmin()
    print('Calls for bands 500, 678, and 750 nm. Using bands ' + str(wave[idx_500])+', '+ str(wave[idx_678]) +', '+ str(wave[idx_750]))
    
    # 3D data, hyperspectral image, [n_row x n_col x n_band]
    if imgData.ndim > 2:
        data_500 = np.reshape(imgData[:,:,idx_500],[-1,1])
        data_678 = np.reshape(imgData[:,:,idx_678],[-1,1])                      
        data_750 = np.reshape(imgData[:,:,idx_750],[-1,1])
    
    # 2D data, flattened hyperspectral data, [n_row x n_band]
    else:
        data_500 = imgData[:,idx_500]
        data_678 = imgData[:,idx_678]
        data_750 = imgData[:,idx_750]                      
    
    # Calculate PSRI
    index = (data_678 - data_500)/(data_750)
    
    # If data was 3D, reshape the index value back into 3D shape
    if imgData.ndim > 2:
        index = np.reshape(index,[imgData.shape[0],imgData.shape[1]])
    
    if isinstance(mask, int) is False:
        idx_x, idx_y = np.where(mask==0)
        index[idx_x,idx_y] = 0
        
    return index

def pssr1_vi(imgData, wave, mask=0):
    """
    Function that calculates the Pigment-Specific Spectral Ratio 1. There is PSSR 1, 2, and 3. 
    This functions uses bands 675 and 800 nm. The closest bands to these values will be used.
    Citation: Blackburn, G.A. 1998. Spectral indices for estimating photosynthetic pigment concentrations: A test using senescent tree leaves, International Journal of Remote Sensing, 19, 657–675.
    INPUTS:
    1) imgData: an array of hyperspectral data either as 3D [n_row x n_col x n_band] or 2D [n_row x n_band]
    2) wave: an array of wavelengths in nanometers that correspond to the n_bands in imgData
    3) mask: OPTIONAL - a binary array (same size as imgData) that designates which pixels should be included in analysis. Pixels with 1 are used, while pixels with 0 are not.
    OUTPUTS:
    1) vi: the calculated spectral index value for each pixel either returned as [n_row x n_col x 1] or [n_row x 1]

    03/2020 - Susan Meerdink
    """
    
    # Find band indexes
    idx_675 = (np.abs(wave - 675)).argmin()                          
    idx_800 = (np.abs(wave - 800)).argmin()
    print('Calls for bands 675 and 800 nm. Using bands ' + str(wave[idx_675])+', '+ str(wave[idx_800]))
    
    # 3D data, hyperspectral image, [n_row x n_col x n_band]
    if imgData.ndim > 2:
        data_675 = np.reshape(imgData[:,:,idx_675],[-1,1])
        data_800 = np.reshape(imgData[:,:,idx_800],[-1,1])                      
    
    # 2D data, flattened hyperspectral data, [n_row x n_band]
    else:
        data_675 = imgData[:,idx_675]
        data_800 = imgData[:,idx_800]                      
    
    # Calculate PSSR 1
    index = (data_800/data_675)
    
    # If data was 3D, reshape the index value back into 3D shape
    if imgData.ndim > 2:
        index = np.reshape(index,[imgData.shape[0],imgData.shape[1]])
    
    if isinstance(mask, int) is False:
        idx_x, idx_y = np.where(mask==0)
        index[idx_x,idx_y] = 0
        
    return index

def pssr2_vi(imgData, wave, mask=0):
    """
    Function that calculates the Pigment-Specific Spectral Ratio 2. There is PSSR 1, 2, and 3. 
    This functions uses bands 650 and 800 nm. The closest bands to these values will be used.
    Citation: Blackburn, G.A. 1998. Spectral indices for estimating photosynthetic pigment concentrations: A test using senescent tree leaves, International Journal of Remote Sensing, 19, 657–675.
    INPUTS:
    1) imgData: an array of hyperspectral data either as 3D [n_row x n_col x n_band] or 2D [n_row x n_band]
    2) wave: an array of wavelengths in nanometers that correspond to the n_bands in imgData
    3) mask: OPTIONAL - a binary array (same size as imgData) that designates which pixels should be included in analysis. Pixels with 1 are used, while pixels with 0 are not.
    OUTPUTS:
    1) vi: the calculated spectral index value for each pixel either returned as [n_row x n_col x 1] or [n_row x 1]

    03/2020 - Susan Meerdink
    """
    
    # Find band indexes
    idx_650 = (np.abs(wave - 650)).argmin()                          
    idx_800 = (np.abs(wave - 800)).argmin()
    print('Calls for bands 650 and 800 nm. Using bands ' + str(wave[idx_650])+', '+ str(wave[idx_800]))
    
    # 3D data, hyperspectral image, [n_row x n_col x n_band]
    if imgData.ndim > 2:
        data_650 = np.reshape(imgData[:,:,idx_650],[-1,1])
        data_800 = np.reshape(imgData[:,:,idx_800],[-1,1])                      
    
    # 2D data, flattened hyperspectral data, [n_row x n_band]
    else:
        data_650 = imgData[:,idx_650]
        data_800 = imgData[:,idx_800]                      
    
    # Calculate PSSR 2
    index = (data_800/data_650)
    
    # If data was 3D, reshape the index value back into 3D shape
    if imgData.ndim > 2:
        index = np.reshape(index,[imgData.shape[0],imgData.shape[1]])
    
    if isinstance(mask, int) is False:
        idx_x, idx_y = np.where(mask==0)
        index[idx_x,idx_y] = 0
        
    return index                              

def pssr3_vi(imgData, wave, mask=0):
    """
    Function that calculates the Pigment-Specific Spectral Ratio 3. There is PSSR 1, 2, and 3. 
    This functions uses bands 500 and 800 nm. The closest bands to these values will be used.
    Citation: Blackburn, G.A. 1998. Spectral indices for estimating photosynthetic pigment concentrations: A test using senescent tree leaves, International Journal of Remote Sensing, 19, 657–675.
    INPUTS:
    1) imgData: an array of hyperspectral data either as 3D [n_row x n_col x n_band] or 2D [n_row x n_band]
    2) wave: an array of wavelengths in nanometers that correspond to the n_bands in imgData
    3) mask: OPTIONAL - a binary array (same size as imgData) that designates which pixels should be included in analysis. Pixels with 1 are used, while pixels with 0 are not.
    OUTPUTS:
    1) vi: the calculated spectral index value for each pixel either returned as [n_row x n_col x 1] or [n_row x 1]

    03/2020 - Susan Meerdink
    """
    
    # Find band indexes
    idx_500 = (np.abs(wave - 500)).argmin()                          
    idx_800 = (np.abs(wave - 800)).argmin()
    print('Calls for bands 500 and 800 nm. Using bands ' + str(wave[idx_500])+', '+ str(wave[idx_800]))
    
    # 3D data, hyperspectral image, [n_row x n_col x n_band]
    if imgData.ndim > 2:
        data_500 = np.reshape(imgData[:,:,idx_500],[-1,1])
        data_800 = np.reshape(imgData[:,:,idx_800],[-1,1])                      
    
    # 2D data, flattened hyperspectral data, [n_row x n_band]
    else:
        data_500 = imgData[:,idx_500]
        data_800 = imgData[:,idx_800]                      
    
    # Calculate PSSR 3
    index = (data_800/data_500)
    
    # If data was 3D, reshape the index value back into 3D shape
    if imgData.ndim > 2:
        index = np.reshape(index,[imgData.shape[0],imgData.shape[1]])
    
    if isinstance(mask, int) is False:
        idx_x, idx_y = np.where(mask==0)
        index[idx_x,idx_y] = 0
        
    return index  

def rvsi_vi(imgData, wave, mask=0):
    """
    Function that calculates the Red-edge Vegetation Stress Index. 
    This functions uses bands 714, 733, and 752 nm. The closest bands to these values will be used.
    Citation: Merton, R. and Huntington, J. 1999. Early simulation results of the ARIES-1 satellite sensor for multi-temporal vegetation research derived from AVIRIS. Available at ftp://popo.jpl.nasa.gov/pub/docs/workshops/99_docs/41.pdf, NASA Jet Propulsion Lab., Pasadena, CA.
    INPUTS:
    1) imgData: an array of hyperspectral data either as 3D [n_row x n_col x n_band] or 2D [n_row x n_band]
    2) wave: an array of wavelengths in nanometers that correspond to the n_bands in imgData
    3) mask: OPTIONAL - a binary array (same size as imgData) that designates which pixels should be included in analysis. Pixels with 1 are used, while pixels with 0 are not.
    OUTPUTS:
    1) vi: the calculated spectral index value for each pixel either returned as [n_row x n_col x 1] or [n_row x 1]

    03/2020 - Susan Meerdink
    """
    
    # Find band indexes
    idx_714 = (np.abs(wave - 714)).argmin()                          
    idx_733 = (np.abs(wave - 733)).argmin()
    idx_752 = (np.abs(wave - 752)).argmin()                          
    print('Calls for bands 714, 733, and 752 nm. Using bands ' + str(wave[idx_714])+', '+ str(wave[idx_733])+', '+ str(wave[idx_752]))
    
    # 3D data, hyperspectral image, [n_row x n_col x n_band]
    if imgData.ndim > 2:
        data_714 = np.reshape(imgData[:,:,idx_714],[-1,1])
        data_733 = np.reshape(imgData[:,:,idx_733],[-1,1])                      
        data_752 = np.reshape(imgData[:,:,idx_752],[-1,1])
          
    # 2D data, flattened hyperspectral data, [n_row x n_band]
    else:
        data_714 = imgData[:,idx_714]
        data_733 = imgData[:,idx_733]                      
        data_752 = imgData[:,idx_752]
          
    # Calculate RVSI
    index = (data_714 + data_752)/ 2 - data_733
    
    # If data was 3D, reshape the index value back into 3D shape
    if imgData.ndim > 2:
        index = np.reshape(index,[imgData.shape[0],imgData.shape[1]])
    
    if isinstance(mask, int) is False:
        idx_x, idx_y = np.where(mask==0)
        index[idx_x,idx_y] = 0
        
    return index

def sipi_vi(imgData, wave, mask=0):
    """
    Function that calculates the Structure Insensitive Pigment Index. 
    This functions uses bands 445, 680, and 800 nm. The closest bands to these values will be used.
    Citation: Peñuelas, J., Baret, F. and Filella, I. 1995. Semi-empirical indices to assess carotenoids/chlorophyll a ratio from leaf spectral reflectance. Photosynthetica, 31, 221–230.
    INPUTS:
    1) imgData: an array of hyperspectral data either as 3D [n_row x n_col x n_band] or 2D [n_row x n_band]
    2) wave: an array of wavelengths in nanometers that correspond to the n_bands in imgData
    3) mask: OPTIONAL - a binary array (same size as imgData) that designates which pixels should be included in analysis. Pixels with 1 are used, while pixels with 0 are not.
    OUTPUTS:
    1) vi: the calculated spectral index value for each pixel either returned as [n_row x n_col x 1] or [n_row x 1]

    03/2020 - Susan Meerdink
    """
    
    # Find band indexes
    idx_445 = (np.abs(wave - 445)).argmin()                          
    idx_680 = (np.abs(wave - 680)).argmin()
    idx_800 = (np.abs(wave - 800)).argmin()                          
    print('Calls for bands 445, 680, and 800 nm. Using bands ' + str(wave[idx_445])+', '+ str(wave[idx_680])+', '+ str(wave[idx_800]))
    
    # 3D data, hyperspectral image, [n_row x n_col x n_band]
    if imgData.ndim > 2:
        data_445 = np.reshape(imgData[:,:,idx_445],[-1,1])
        data_680 = np.reshape(imgData[:,:,idx_680],[-1,1])                      
        data_800 = np.reshape(imgData[:,:,idx_800],[-1,1])
          
    # 2D data, flattened hyperspectral data, [n_row x n_band]
    else:
        data_445 = imgData[:,idx_445]
        data_680 = imgData[:,idx_680]                      
        data_800 = imgData[:,idx_800]
          
    # Calculate RVSI
    index = (data_800 + data_445)/(data_800 - data_680)
    
    # If data was 3D, reshape the index value back into 3D shape
    if imgData.ndim > 2:
        index = np.reshape(index,[imgData.shape[0],imgData.shape[1]])
    
    if isinstance(mask, int) is False:
        idx_x, idx_y = np.where(mask==0)
        index[idx_x,idx_y] = 0
        
    return index 

def sr_vi(imgData, wave, mask=0):
    """
    Function that calculates the Simple Ratio. 
    This functions uses bands 675 and 800 nm. The closest bands to these values will be used.
    Citation: Jordan, C.F. 1969. Leaf-area index from quality of light on the forest floor, Ecology, 50(4), 663–666.
    INPUTS:
    1) imgData: an array of hyperspectral data either as 3D [n_row x n_col x n_band] or 2D [n_row x n_band]
    2) wave: an array of wavelengths in nanometers that correspond to the n_bands in imgData
    3) mask: OPTIONAL - a binary array (same size as imgData) that designates which pixels should be included in analysis. Pixels with 1 are used, while pixels with 0 are not.
    OUTPUTS:
    1) vi: the calculated spectral index value for each pixel either returned as [n_row x n_col x 1] or [n_row x 1]

    03/2020 - Susan Meerdink
    """
    
    # Find band indexes
    idx_675 = (np.abs(wave - 675)).argmin()                          
    idx_800 = (np.abs(wave - 800)).argmin()                          
    print('Calls for bands 675 and 800 nm. Using bands ' + str(wave[idx_675])+', '+ str(wave[idx_800]))
    
    # 3D data, hyperspectral image, [n_row x n_col x n_band]
    if imgData.ndim > 2:
        data_675 = np.reshape(imgData[:,:,idx_675],[-1,1])                     
        data_800 = np.reshape(imgData[:,:,idx_800],[-1,1])
          
    # 2D data, flattened hyperspectral data, [n_row x n_band]
    else:
        data_675 = imgData[:,idx_675]                      
        data_800 = imgData[:,idx_800]
          
    # Calculate RVSI
    index = (data_800/data_675)
    
    # If data was 3D, reshape the index value back into 3D shape
    if imgData.ndim > 2:
        index = np.reshape(index,[imgData.shape[0],imgData.shape[1]])
    
    if isinstance(mask, int) is False:
        idx_x, idx_y = np.where(mask==0)
        index[idx_x,idx_y] = 0
        
    return index 

def wbi_vi(imgData, wave, mask=0):
    """
    Function that calculates the Water Band Index. 
    This functions uses bands 900 and 970 nm. The closest bands to these values will be used.
    Citation: Peñuelas, J., Pinol, J., Ogaya, R. and Lilella, I. 1997. Estimation of plant water content by the reflectance water index WI (R900/R970), International Journal of Remote Sensing, 18, 2869–2875.
    INPUTS:
    1) imgData: an array of hyperspectral data either as 3D [n_row x n_col x n_band] or 2D [n_row x n_band]
    2) wave: an array of wavelengths in nanometers that correspond to the n_bands in imgData
    3) mask: OPTIONAL - a binary array (same size as imgData) that designates which pixels should be included in analysis. Pixels with 1 are used, while pixels with 0 are not.
    OUTPUTS:
    1) vi: the calculated spectral index value for each pixel either returned as [n_row x n_col x 1] or [n_row x 1]

    03/2020 - Susan Meerdink
    """
    
    # Find band indexes
    idx_900 = (np.abs(wave - 900)).argmin()                          
    idx_970 = (np.abs(wave - 970)).argmin()                          
    print('Calls for bands 900 and 970 nm. Using bands ' + str(wave[idx_900])+', '+ str(wave[idx_970]))
    
    # 3D data, hyperspectral image, [n_row x n_col x n_band]
    if imgData.ndim > 2:
        data_900 = np.reshape(imgData[:,:,idx_900],[-1,1])                     
        data_970 = np.reshape(imgData[:,:,idx_970],[-1,1])
          
    # 2D data, flattened hyperspectral data, [n_row x n_band]
    else:
        data_900 = imgData[:,idx_900]                      
        data_970 = imgData[:,idx_970]
          
    # Calculate RVSI
    index = (data_900/data_970)
    
    # If data was 3D, reshape the index value back into 3D shape
    if imgData.ndim > 2:
        index = np.reshape(index,[imgData.shape[0],imgData.shape[1]])
    
    if isinstance(mask, int) is False:
        idx_x, idx_y = np.where(mask==0)
        index[idx_x,idx_y] = 0
        
    return index 
