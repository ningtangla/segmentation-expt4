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

class setFeaturesInPartitions():
    
    def __init__(self, initialFeaturesDistribution, changeFeaturePRange):
        self.initialFeaturesDistribution = initialFeaturesDistribution
        self.changeFeaturePRange = changeFeaturePRange
        
    def __call__(self, tree):
        treeWithFeatureDi = tree.copy() # 
        tree.node[1]['featuresDistribution'] = self.initialFeaturesDistribution

        featuresNum = len(self.initialFeaturesDistribution)
        changeFeatureIndexAtDiffDepth = range(featuresNum)
        np.random.shuffle(changeFeatureIndexAtDiffDepth)
        
        nonleafNodes = [ n for n,d in tree.out_degree().items() if d!=0]
        for nonleafNode in nonleafNodes:          
            childrenOfNonleafNode = nx.DiGraph.successors(tree, nonleafNode)
            childrenNum = len(childrenOfNonleafNode)
            parentFeatureDistrbution = tree.node[nonleafNode]['featuresDistribution'][:]
            childrenDepth = tree.node[nonleafNode]['depth']
            changeFeatureIndex = changeFeatureIndexAtDiffDepth[childrenDepth % featuresNum]
            changeFeaturePValueDecrese = parentFeatureDistrbution[changeFeatureIndex]
            changeFeaturePValueIncrese = parentFeatureDistrbution[(changeFeatureIndex + int(featuresNum/2)) % featuresNum]

            changeFeautrePExtent = self.changeFeaturePRange/(childrenNum - 1)
            
            for child in childrenOfNonleafNode:  

                parentFeatureDistrbution[changeFeatureIndex] =  changeFeaturePValueDecrese - (childrenOfNonleafNode.index(child)) * changeFeautrePExtent 
                parentFeatureDistrbution[(changeFeatureIndex + int(featuresNum/2)) % featuresNum] =  changeFeaturePValueIncrese + (childrenOfNonleafNode.index(child)) * changeFeautrePExtent                         
                tree.node[child]['featuresDistribution'] = parentFeatureDistrbution[:]                        
            
        leafNodes = [ n for n,d in tree.out_degree().items() if d==0]  
        featuresDistributionsInPartitions = np.array([tree.node[leafNode]['featuresDistribution'][:] for leafNode in leafNodes])    
        partitions = np.array([[tree.node[leafNode]['x'], tree.node[leafNode]['y']] for leafNode in leafNodes]) 
        return featuresDistributionsInPartitions, partitions

class drawTexonsWithFeatures():
    def __inint__(self, ):
        

class sampleTexon        
    
def main():

    
if __name__=="__main__":
    main()