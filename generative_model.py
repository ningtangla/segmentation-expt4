# -*- coding: utf-8 -*-
"""
Created on Tue Oct 09 01:38:06 2018

@author: Edward Coen
"""

from __future__ import division
import cv2
import numpy as np
import networkx as nx
import itertools as it
import pandas as pd
import itertools as it

class sampleTexonsLocationsAndFeatures():
    def __init__(self, gridLengthX, gridLengthY, featureVariances):
        self.gridLengthX = gridLengthX
        self.gridLengthY = girdLengthY
        self.featureVariances = featureVariances

    def __call__(self, partitionX, partitionY, partionnFeatureMeans):
        partitionXMin = partitionX[0]
        partitionXMax = partitionX[1]
        partitionYMin = partitionY[0]
        partitionYMax = partitionY[1]
        gridNumInPartitionX = floor((partitionXMax - partitionXMin)/self.gridLengthX)
        gridNumInPartitionY = floor((partitionYMax - partitionYMin)/self.gridLengthY)
        locationXCenters = [partitionXMin + int(self.gridLengthX/2) + gridIndexX * self.gridLengthX for gridIndexX in range(girdNumInPartitionX)
        locationYCenters = [partitionYMin + int(self.gridLengthY/2) + gridIndexY * self.gridLengthY for gridIndexY in range(girdNumInPartitionY)
        girdNumInPartition = gridNumInPartitionX * girdNumInPartitionY
        locationNoises = np.random.multivariate_normal([0, 0], [[self.gridLengthX/3, 0], [0, self.gridLengthY/3]], girdNumInPartition)
        locationCenters = np.array(list(it.product(gridLocationXCenters, gridLocationYCenters)))
        
        texonLocations = locationCenters + locationNoises 
        pd.DataFrame(texon
        return texonLocations 

class sampleTexonFeatures():
    def __init__(self, featureVariances):
        self.featureVariances = featureVariances

    def __call__(self, )
        
        np.random.normal(,texonNum)

class visualizeTexons():
    def __init__(self, ):
    
    def __call__(self, ):
    

def main():
    gridLengthX = 10
    gridLengthY = 10

    folderDir = os.path.dirname(os.path.abspath(__file__))
    treeDir = os.path.join(folderDir, 'tree')
    assert (os.path.isdir(treeDir))
    treeBaseNames = [baseName for baseName in os.listdir(treeDir) if
    baseName.endswith('.gpickle')]
    treePathNames = [os.path.join(treeDir, treeBaseName) for treeBaseName in
            treeBaseNames]
    trees = [nx.read_pickle(treePathName) for treePathName in treePathNames]

    featuresRootNode = {'color': 1, 'size': 3, 'orientation': math.pi/4,
            'widthHeightRatio': 2}
    sampleNodeFeatureMeansGivenTrees = SampleNodeFeatureMeansGivenTrees()
    featuredTrees = [sampleNodeFeatureMeansGivenTrees(tree) for tree in trees]
    
    featureVariances = {'color': 0.1, 'size': 0.2, 'orientation': math.pi/16,
            'widthHeightRation': 0.3}    

if __name__=="__main__":
    main()
