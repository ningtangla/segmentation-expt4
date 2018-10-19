# -*- coding: utf-8 -*-
"""
created on tue oct 09 01:38:06 2018

@author: edward coen
"""

from __future__ import division
import pygame
import numpy as np
import networkx as nx
import itertools as it
import pandas as pd


class sampleNodeFeatureMean():
    def __init__(self, featuresRootNode, featuresRange)
        self.featuresRootNode = featuresrootnode
        self.featuresRange = featuresRange

    def __call__(self, tree)
        nonleafNodes = [n for n,d in tree.out_degree().items() if d!=0]
        for nonleafNode in nonleafNodes:
            childrenOfNonleafNode = nx.DiGraph.successors(tree, nonleafNode)
            childrenOfNonleafNode.sort()
            parentFeatureMean = tree.node[nonleafNode]['features'][:]


class sampleTexonsLocationsAndFeatures():
    def __init__(self, gridLengthX, gridLengthY, featureVariances):
        self.gridLengthX = gridLengthX
        self.gridLengthY = girdLengthY
        self.featureVariances = featureVariances

    def __call__(self, partitionX, partitionY, partitionFeatureMeans):
        partitionXMin, partitionXMax = partitionX    
        partitionYMin, partitionYMax = partitionY
        gridNumPartitionX = floor((partitionXMax -
            partitionXMin)/self.gridLengthX)
        gridNumPartitionY = floor((partitionYMax -
            partitionYMin)/self.gridLengthY)
        locationXCenters = [partitionXMin + int(self.gridLengthX/2) +
                gridIndexX * self.gridLengthX for gridIndexX in
                range(gridNumPartitionX)]
        locationYCenters = [partitionYMin + int(self.gridLengthY/2) +
                gridIndexY * self.gridLengthY for gridIndexY in
                range(gridNumPartitionY)
        girdNumPartition = gridNumPartitionX * gridNumPartitionY
        locationNoises = np.random.multivariate_normal([0, 0],
            [[self.gridLengthX/3, 0], [0, self.gridLengthY/3]],
            girdNumPartition)
        locationCenters = np.array(list(it.product(locationXCenters,
            locationYCenters)))
        texonsLocation = locationCenters + locationNoises
        texonsFeaturesValue =
        [sampleTexonsFeatureValue(partitionFeatureMeans[featureName],
            self.featureVariances[featureName], girdNumPartition) for
            featureName in partitionFeatureMeans.keys()]
        texonsParameter = pd.dataframe(zip(*texonsLocation) +
            texonsFeaturesValue, columns = pd.index(['x', 'y'] +
                partitionFeatureMeans.keys()))
        return texonsParameter

def transTexonParameterToPolygenDrawArguemnt(texonsParameter):
    texonsParameter['width'] = texonsParameter['length'] * texonsParameter['widthLengthRatio']
    texonsParameter['lengthRotatedProjectX'] = texonsParameter['length'] *
    math.cos(texonsParameter['angelRotated'])
    texonsParameter['widthRotatedProjectX'] = texonsParameter['width'] *
    math.sin(texonsParameter['angleRotated']) 
    texonsParameter['lengthRotatedProjectY'] = texonsParameter['length'] *
    math.sin(texonsParameter['angelRotated'])
    texonsParameter['widthRotatedProjectY'] = texonsParameter['width'] *
    math.cos(texonsParameter['angleRotated'])

    texonsParameter['vertexRightBottom'] = tuple(int((texonsParameter['x'] +
        texonsParameter['lengthRotatedProjectX'] +
        texonsParameter['widthRotatedProjectX'])/2.0),
        int((texonsParameter['y'] + texonsParameter['lengthRotatedProjectY'] -
            texonsParameter['widthRotatedProjectY'])/2.0))    
    texonsParameter['vertexRightTop'] = tuple(int((texonsParameter['x'] +
        texonsParameter['lengthRotatedProjectX'] -
        texonsParameter['widthRotatedProjectX'])/2.0),
        int((texonsParameter['y'] + texonsParameter['lengthRotatedProjectY'] +
            texonsParameter['widthRotatedProjectY'])/2.0))
    texonsParameter['vertexLeftTop'] = tuple(int((texonsParameter['x'] -
        texonsParameter['lengthRotatedProjectX'] -
        texonsParameter['widthRotatedProjectX'])/2.0),
        int((texonsParameter['y'] - texonsParameter['lengthRotatedProjectY'] +
            texonsParameter['widthRotatedProjectY'])/2.0))
    texonsParameter['vertexLeftBottom'] = tuple(int((texonsParameter['x'] -
        texonsParameter['lengthRotatedProjectX'] +
        texonsParameter['widthRotatedProjectX'])/2.0),
        int((texonsParameter['y'] - texonsParameter['lengthRotatedProjectY'] -
            texonsParameter['widthRotatedProjectY'])/2.0))
    return  texonsParameter

def sampletexonsfeaturevalue(featureMean, featureVariance, texonNum):
    return np.random.normal(featureMean, featureVariance, texonNum)

class VisualizeTexons():
    def __init__(self, heightImage, widthImage):
        self.screen = pygame.display.set_mode([widthImage, heightImage])
        self.screen.fill((0, 0, 0))

    def __call__(self, texonsParameter):
        for texonIndex in range(len(texonsParameter)):
            pygame.draw.polygon(screen,
                    texonsParameter.iloc[texonIndex]['color'] * (0, 255, 0),
                    texonsParameter.iloc[texonIndex][['vertexRightBottom','vertexRightTop',
                        'vertexLeftTop', 'vertexLeftBottom']], 0)
        screen.flip()
        pygame.image.save(screen, '~/segmentation-expt4/generateDemo.png')

def main():
    gridlengthx = 10
    gridlengthy = 10

#    folderdir = os.path.dirname(os.path.abspath(__file__))
#    treedir = os.path.join(folderdir, 'tree')
#    assert (os.path.isdir(treedir))
#    treebasenames = [basename for basename in os.listdir(treedir) if
#    basename.endswith('.gpickle')]
#    treepathnames = [os.path.join(treedir, treebasename) for treebasename in treebasenames]
#    trees = [nx.read_pickle(treepathname) for treepathname in treepathnames]
    tree = nx.Digraph()
    tree.add_node(1, x = [0, 200], y = [0, 150])
    tree.add_node(2, x = [0, 200], y = [0, 100])
    tree.add_node(3, x = [0, 200], y = [101, 150])
    tree.add_node(4, x = [0, 100], y = [0, 100])
    tree.add_node(5, x = [101, 200], y = [0, 100])
    tree.add_edges_from([(1,2),(1,3),(2,4),(2,5)])

    featuresrootnode = {'color': 0.5, 'size': 3, 'orientation': math.pi/4,
            'widthheightratio': 2}
    samplenodefeaturemeansgiventrees = samplenodefeaturemeansgiventrees()
    featuredTrees = [sampleNodeFeatureMeansGivenTrees(tree) for tree in trees]
    
    featureVariances = {'color': 0.05, 'size': 0.2, 'orientation': math.pi/16,
            'widthHeightRation': 0.03}    

if __name__=="__main__":
    main()
