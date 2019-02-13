#        [E, IdxOfE, Xpca] = VCA(X, M, r, v)
#
###############################################################################
#
# A FUNCTION TO CALCULATE ENDMEMBERS USING Vertex Component Analysis (VCA)
# REFERENCE:
# José M. P. Nascimento and José M. B. Dias
# "Vertex Component Analysis: A Fast Algorithm to Unmix Hyperspectral Data"
# IEEE Trans. Geosci. Remote Sensing,
# April, 2005
###############################################################################
###
### INPUTS:
###         X:          DATA MATRIX B WAVELENGTHS x N PIXELS
###         M:          NUMBER OF ENDMEMBERS TO ESTIMATE
### OUTPUTS:
###         E:          B x M MATRIX OF ESTIMATED ENDMEMBERS
###         IdxOfEinX:  INDICES OF ENDMEMBERS IN DATA X.
###                         (THE ENDMEMBERS ARE SELECTED FROM X)
###         XM:         X PROJECTED ONTO FIRST M COMPONENTS OF PCA
#
### OPTIONAL INPUTS:
### 'SNR'     r:   ESTIMATED SIGNAL TO NOISE RATIO IN dB
### 'verbose' v:   logical TOGGLE TO TURN DISPLAYS ON AND OFF
###############################################################################
###
### Authors: José Nascimento (zen@isel.pt)
###          José Bioucas Dias (bioucas@lx.it.pt)
### Copyright (c)
### version: 2.1 (7-May-2004)
###############################################################################
###
### ORIGINALLY MODIFED BY Darth Gader OCTOBER 2018
### TRANSLATED TO PYTHON BY Ronald Fick November 2018
###############################################################################

import numpy as np
import numpy.matlib
from scipy.sparse.linalg import svds
import math
import matplotlib.pyplot as plt

def VCA(X, M=2, r=-1, verbose=True):
    ##
    #############################################
    # Initializations
    #############################################
    
    if(X.size == 0):
        raise ValueError('There is no data')
    else:
        B, N=X.shape
    
    if (M<0 or M>B or M!=int(M)):
        raise ValueError('ENDMEMBER parameter must be integer between 1 and B')
        
    ##
    ####################
    ### ESTIMATE SNR ###
    ####################
    if(r==-1):
        ############################################################
        ### THE USER DID NOT INPUT AN SNR SO SNR IS CALCULATED HERE
        ############################################################
        
        #############################
        ### CALCULATE PCA AND PCT ###
        MuX          = np.mean(X, axis=1)     #axis 1 is the second dimension
        MuX.shape = (MuX.size,1)
        Xz           = X - np.matlib.repmat(MuX, 1, N)
        SigmaX       = np.cov(X)
        U,S,V        = np.linalg.svd(SigmaX)
        Xpca         = np.matmul(np.transpose(U[:,0:M]), Xz)
        ProjComputed = True
        
        ### ESTIMATE SIGNAL TO NOISE ###
        SNR = EstSNR(X,MuX,Xpca)
        
        ### PRINT SNR ###
        if verbose:
            print('Estimated SNR = %g[dB]'%SNR)
    else:
        ############################################################
        ### THE USER DID INPUT AN SNR SO NO SNR CALCUATION NEEDED
        ############################################################
           
        ProjComputed = False
        if verbose:
            print('Input SNR = %g[dB]'%SNR)
            
    ### SET THRESHOLD TO DETERMINE IF NOISE LEVEL IS LOW OR HIGH ###
    SNRThresh = 15 + 10*math.log10(M)
    
    if verbose:
        print('SNRThresh= %f SNR= %f Difference= %f'%(SNR,SNRThresh,SNRThresh-SNR))
    
    ###########################
    ### END OF ESTIMATE SNR ###
    ###########################
    ##
    ##################################################################
    ### PROJECTION.                                                ###
    ### SELECT AND CALCULATE PROJECTION ONTO M DIMS                ###
    ###                                                            ###
    ### IF SNR IS LOW,IT IS ASSUMED THAT THERE IS STILL NOISE IN   ###
    ### THE SIGNAL SO REDUCE DIM A BIT MORE.                       ###
    ### ADD A CONSTANT VECTOR TO KEEP IN DIM M.                    ###
    ###                                                            ###
    ### IF SNR IS HIGH, PROJECT TO SIMPLEX IN DIM M-1 TO REDUCE    ###
    ### EFFECTS OF VARIABLE ILLUMINATION                           ###
    ##################################################################
    
    if(SNR < SNRThresh):
    ##########################
    ### BEGIN LOW SNR CASE ###
    ##########################
    
        ### PRINT MESSAGE ###
        if verbose:
            print('Low SNR so Project onto Dimension M-1.')
        
        ### REDUCE SIZE OF PCT MATRIX TO M-1 ###
        Dim    = M-1
        MuX    = np.mean(X, axis=1)
        MuX.shape = (MuX.size, 1)
        BigMuX = np.matlib.repmat(MuX, 1, N)
        if ProjComputed:
            U= U[:,0:Dim]
        else:
            Xz      = X - BigMuX
            SigmaX  = np.cov(np.transpose(X))
            U,S,V   = np.linalg.svd(SigmaX)
            #U,S,V   = np.linalg.svd(SigmaX, full_matrices=True)
            Xpca    = np.matmul(np.transpose(U), Xz)
        
        ### REDUCE DIMENSIONALITY IN PCA DOMAIN ###
        XpcaReduced = Xpca[0:Dim,:]
        
        ### RECONSTRUCT X "WITHOUT NOISE" BY PROJECTING BACK TO ORIGINAL SPACE ###
        XNoiseFree =  np.matmul(U, XpcaReduced) + BigMuX
        
        ### CONCATENATE  CONSTANT VEC = MAX NORM OF ALL DATA POINTS ###
        BiggestNorm = np.sqrt(max(np.sum(np.square(XpcaReduced),1)))

        YpcaReduced = np.concatenate([XpcaReduced, BiggestNorm*np.ones((1,N))])
    ########################
    ### END LOW SNR CASE ###
    ########################
    else:
    ###########################
    ### BEGIN HIGH SNR CASE ###
    ###########################
        if verbose:
            print('High SNR so project onto dimension M')
        
        ### CONTRUCT "PCA-LIKE" MATRIX BY DIAGONALIZING CORRELATION MATRIX     ###
        ### IF SUBTRACT THE MEAN, THEN MUPCA WILL BE 0, SO NO MEAN SUBTRACTION ###
        # xb = a: solve b.T x.T = a.T instead
        U,S,V   = np.linalg.svd(np.matmul(X,np.transpose(X))/N)
        
        ### CALC PCT WITHOUT SUBTRACTING MEAN AND THEN RECONSTRUCT ###
        XpcaReduced = np.matmul(U.T,X)
        
        ### RECONSTRUCT X VIA INVERSE XFORM ON REDUCED DIM PCA DATA       ###
        ### XXX PDG NOT SURE IF THIS IS CORRECT IN ORIGINAL CODE OR HERE. ###
        ### I THINK THE MEAN SHOULD BE ADDED LIKE IN LOW SNR CASE         ###
        XNoiseFree =  np.matmul(U, XpcaReduced[0:B,:])      # again in dimension L (note that x_p has no null mean)

        #################################################################
        ### CALCULATE NORMALIZED PROJECTION                           ###
        ### SEE LAST PARAGRAPH OF SECTION A AND FIGURE 4 IN VCA PAPER.###
        ### MEAN OF THE PROJECT DATA IS USED AS u FROM THAT SECTION   ###
        Mupca       = np.mean(XpcaReduced,1)
        Mupca.shape = (Mupca.size, 1)
        Denom       = np.sum(np.multiply(XpcaReduced, np.tile(Mupca,[1, N])))
        YpcaReduced = np.divide(XpcaReduced, np.tile(Denom, [B, 1]))
        
    ##
    ###########################################################
    # VCA ALGORITHM
    ###########################################################
    ###
    ###   INITIALIZE ARRAY OF INDICES OF ENDMEMBERS, IdxOfE
    ###   INITIALIZE MATRIX OF ENDMEMBERS IN PCA SPACE, Epca
    ###   DO M TIMES
    ###      PICK A RANDOM VECTOR w
    ###      USE w TO FIND f IN NULL SPACE OF Epca
    ###      PROJECT f ONTO ALL PCA DATA PTS
    ###      FIND INDEX OF PCA DATA PT WITH MAX PROJECTION
    ###      USE INDEX TO ADD PCA DATA PT TO Epca
    ###   END DO
    ###   
    ###   USE INDICES TO SELECT ENDMEMBERS IN SPECTRAL SPACE
    ###########################################################
    
    VCAsize = YpcaReduced.shape[0]

    IdxOfE    = np.zeros(VCAsize)
    IdxOfE = IdxOfE.astype(int)
    Epca      = np.zeros((VCAsize,VCAsize))
    Epca[VCAsize-1,0] = 1
    for m in range(0,VCAsize):
        w                 = np.random.rand(VCAsize,1)
        f                 = w - np.matmul(np.matmul(Epca,np.linalg.pinv(Epca)), w)
        f                 = f / np.sqrt(np.sum(np.square(f)))
        v                 = np.matmul(f.T,YpcaReduced)
        absV = np.abs(v)
        absV = np.squeeze(absV)
        IdxOfE[m] = np.argmax(absV)
        v_max  = absV[IdxOfE[m]]
        Epca[:,m]         = YpcaReduced[:,IdxOfE[m]]
    
    E = XNoiseFree[:,IdxOfE]
    
    return E[:,0:M], IdxOfE[0:M], Xpca
    #############################################
    ### END OF VCA FUNCTION
    #############################################
    #############################################
    ##
#############################################
#############################################
### FUNCTION TO ESTIMATE SNR
#############################################

def EstSNR(X,MuX,Xpca):
    #####################################################
    ### THE ESTIMATED SNR IS EQUIVALENT TO THE RATIO OF 
    ### POWER OF SIGNAL TO POWER OF NOISE
    ### BUT IS COMPUTED AS
    ### POWER OF (SIGNAL + NOISE) TO POWER OF NOISE
    #####################################################
    B, N  = X.shape
    M, N  = Xpca.shape
    
    ### POWER OF SIGNAL + NOISE, SIGNAL, AND NOISE, RESP. ###
    Psn = np.sum(np.square(X.flatten()))/N
    Ps  = np.sum(np.square(Xpca.flatten()))/N + np.matmul(np.transpose(MuX),MuX)
    Pn  = Psn - Ps
    SNR = 10*np.log10(Ps/Pn)
    
    ### OLD SNR CODE AS IN VCA PAPER.  NOT MUCH DIFFERENT ###
    ### BECAUSE (M/B) IS SMALL ###
    ### SNR = 10*log10( (Ps - M/B*Psn)/(Psn - Ps) );
    ### fprintf('SNR= %8.4f   RatSNR= %8.4f\n', SNR, RatSNR);
    
    return SNR
    ###############
    ### THE END ###
    ###############