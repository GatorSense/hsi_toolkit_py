#function [SpecDist] = PlotSpectraDistribution(Spectra, WaveLengths,SampStuff, FigNum);
###
######################################################################
### A FUNCTION THAT CREATES & DISPLAYS SPECTRA AS A 2D HISTOGRAM   ###
###    SPECTRA ARE ASSUMED REFLECTANCES OR EMISSIVITIES IN [0,1]   ###
###    SPECTRA ARE MAPPED TO INTEGERS BETWEEN 0 AND 100 (OR < 100) ###
######################################################################
###
### INPUTS:
###   I1. Spectra IS A NUMBER SPECTRA x NUMBER BANDS...        ###
###       ...ARRAY OF REFLECTANCES OR EMISSIVITIES             ###
###   I2. WaveLengths IS A VECTOR OF THE SPECTRAL WAVELENGTHS  ###
###   I3. SampStuff IS A VECTOR CONTAINING                     ###
###          SampInt:       FRACTIONAL SIZE OF HISTOGRAM BINS  ###
###          IntSampInt:    INT VERSION OF SampInt             ###
###          IntTopReflect: INT VALUE OF MAX REF/EMIS BIN      ###
###   I4. FigNum IS THE INDEX OF THE FIGURE TO USE FOR DISPLAY ###
###          IF FigNum < 1, DO NOT DISPLAY ANYTHING            ###
###
### OUTPUTS:
###   O1. SpecDist IS THE 2D HISTOGRAM                         ###
##################################################################
###                                                            ###
### MATLAB AUTHOR:      Darth Gader                            ###
### PYTHON AUTHOR:      Ron Fick                               ###
### LAST UPDATE: 021519                                        ###
###                                                            ###
##################################################################
def PlotSpectraDistribution(Spectra, WaveLengths, SampStuff, FigNum):

    import numpy as np
    from scipy import signal
    import cv2
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    
    ##
    ### INITIALIZE PARAMETERS ###
    
    SampInt       = SampStuff[0]
    IntSampInt    = SampStuff[1]
    IntTopReflect = SampStuff[2]
    SMOOTHSIZE    = [3,3]
    NumWave       = np.size(Spectra, 1)
    SpecDist      = np.zeros((IntTopReflect, NumWave))
    assert NumWave == np.size(WaveLengths), 'Wavelength sizes don''t match'
    
    ### MAP SPECTRA TO [0, 100] ###
    MappedSpectra = np.minimum(100, (Spectra*99)+1)
    MappedSpectra = np.maximum(1, np.round(MappedSpectra/SampInt)*SampInt)
    
    ##
    ### MAKE A HISTOGRAM FOR EACH WAVELENGTH ###
    for k in range(NumWave):
        SpecDist[:, k] = np.histogram(MappedSpectra[:, k], np.arange(0, IntTopReflect+IntSampInt, IntSampInt))[0]
    
    ### SMOOTH BY TAKING A LOCAL MAX FOLLOWED BY A LOCAL AVERAGE ###
    SpecDist   = cv2.dilate(SpecDist, np.ones((3,3)), iterations=1)
    SpecDist   = signal.convolve2d(SpecDist, (1/np.prod(SMOOTHSIZE))*np.ones(SMOOTHSIZE), 'same')
    
    ##
    ### DISPLAY AS MESH ###
    if(FigNum > 0):
        XAxis      = WaveLengths;
        YAxis      = np.arange(0, IntTopReflect, IntSampInt).T
        X, Y = np.meshgrid(XAxis, YAxis)
        fig        = plt.figure(FigNum)
        ax         = fig.gca(projection='3d')
        surf       = ax.plot_surface(X, Y, SpecDist, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.title('Spectra Histogram')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Reflectance')
        plt.show()

### END OF FUNCTION ###
#######################