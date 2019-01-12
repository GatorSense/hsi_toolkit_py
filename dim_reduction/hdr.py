
"""
function [dimReductionStruct] = dimReduction(img, Parameters)

Hierarchical Dimensionality Reduction
Computes KL-divergence between every pair of bands in image and merges
bands that are similar based on KL-divergence.

Inputs:
  img: hyperspectral data cube (n_row x n_col x n_bands)
  Parameters: parameter structure defined by dimReductionParameters.m

Author: Alina Zare
Email Address: azare@ufl.edu
Created: September 12, 2008
Latest Revision: October 15, 2018
Translation to Python: Caleb Robey
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
import scipy.cluster.hierarchy as sch


class dimReductionParameters():
    def __init__(self):
        self.numBands = 7  # Reduced dimensionality Size
        self.type = 'complete'  # Type of hierarchical clustering used
        self.showH = 0  # Set to 1 to show clustering, 0 otherwise
        self.NumCenters = 255  # Number of centers used in computing KL-divergence


def dimReduction(img, Parameters=None):

    numRows, numCols, numDims = img.shape

    if Parameters is None:
        Parameters = dimReductionParameters()

    type = Parameters.type  # Type of Hierarchy
    showH = 1 # Parameters.showH  # Set to 1 to show clustering, 0 otherwise
    maxNumClusters = Parameters.numBands
    NumCenters = Parameters.NumCenters

    InputData = np.reshape(img, (numRows * numCols, numDims), order='F')
    _, KLDivergencesList, _ = computeKLDivergencesBetweenBands(InputData, NumCenters);

    Hierarchy = sch.linkage(KLDivergencesList, type)
    band_clusters = sch.fcluster(Hierarchy, t=maxNumClusters, criterion='maxclust')

    print(band_clusters.shape)
    if (showH):
        # 'mtica' gives matlab behavior
        D = sch.dendrogram(Hierarchy, 0, 'mtica')
        plt.show()

    mergedData = np.zeros((maxNumClusters, (numRows * numCols)))

    for i in range(1, maxNumClusters+1):
        mergedData[i-1, :] = np.mean(InputData[:, band_clusters == i], 1)

    mergedData = np.reshape(mergedData, (numRows, numCols, maxNumClusters), order='F')

    return mergedData


def computeKLDivergencesBetweenBands(InputData, NumCenters):

    # TESTED (keeping in mind that MATLAB and python reshape are different)
    DataList = InputData / InputData.max(1).max(0)
    # print('Datalist data: ', DataList[1,1])
    # TESTED
    # compute the histograms
    Centers = np.arange(1/NumCenters, 1 + 1/NumCenters, 1/NumCenters)

    hists = np.zeros((NumCenters-1, DataList.shape[0]))

    print(DataList.shape, Centers.shape)
    for count in range(DataList.shape[0]):
        hists[:, count], _ = np.histogram(DataList.T[:, count], Centers)

    #hists, bins = plt.hist(DataList.T[:], Centers, histtype='step', align='mid')

    hists = hists + np.spacing(1)

    # compute KL Divergence
    lim = InputData.shape[1]
    KLDivergences = np.zeros((lim, lim))

    for i in range(lim):
        for j in range(lim):
            KLDivergences[i, j] = np.sum(np.multiply(hists[i], np.log(hists[i]/hists[j]))) \
                                  + np.sum(np.multiply(hists[j], np.log(hists[j]/hists[j])))

    temp = KLDivergences - np.diag(np.diag(KLDivergences))
    KLDivergencesList = squareform(pdist(temp))

    return KLDivergences, KLDivergencesList, hists
