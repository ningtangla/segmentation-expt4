import pygame
import math
import numpy as np
import networkx as nx
import itertools as it
import pandas as pd


class SampleNodeFeatureMean():
    def __init__(self, featureUpperBound, featureLowerBound):
        self.featureUpperBound = featureUpperBound
        self.featureLowerBound = featureLowerBound

    def __call__(self, tree):
        nonleafNodes = [n for n,d in dict(tree.out_degree()).items() if d!=0]
        featureIndex = self.featureUpperBound.columns.values
        featureName = featureIndex.copy()
        np.random.shuffle(featureName)
        
        for nonleafNode in nonleafNodes:
            childrenOfNonleafNode = list(tree.successors(nonleafNode))
            childrenOfNonleafNode.sort()
            parentFeatureMeans = tree.node[nonleafNode]['featureMeans'][:].copy()
            childrenDepth = tree.node[nonleafNode]['depth'] + 1
            changeFeature = featureName[childrenDepth % len(featureName)]
            lengthFeatureChange = abs(self.featureUpperBound[changeFeature].values - self.featureLowerBound[changeFeature].values) 
            gaussianMeanOfChangeFeatureMean = (self.featureUpperBound[changeFeature] + self.featureLowerBound[changeFeature])/2
            gaussianSDOfChangeFeatureMean = lengthFeatureChange/10
            childrenChangeFeatureMeans = np.random.normal(gaussianMeanOfChangeFeatureMean, gaussianSDOfChangeFeatureMean ** 2, len(childrenOfNonleafNode))
            for child in childrenOfNonleafNode:
                parentFeatureMeans[changeFeature] = childrenChangeFeatureMeans[childrenOfNonleafNode.index(child)]
                tree.node[child]['featureMeans'] = parentFeatureMeans.copy()
        
        leafNodes = [n for n,d in dict(tree.out_degree()).items() if d==0]
        featureMeansLeafPartion = pd.concat([tree.node[leafNode]['featureMeans'] for leafNode in leafNodes])
        leafPartitionParameterDf = pd.DataFrame(featureMeansLeafPartion)
        leafPartitionParameterDf['x'] = [tree.node[leafNode]['x'] for leafNode in leafNodes]
        leafPartitionParameterDf['y'] = [tree.node[leafNode]['y'] for leafNode in leafNodes]
        return leafPartitionParameterDf

class SampleTexonsLocationsAndFeatures():
    def __init__(self, gridLengthX, gridLengthY, featureStdVarinces):
        self.gridLengthX = gridLengthX
        self.gridLengthY = gridLengthY
        self.featureStdVarinces = featureStdVarinces

    def __call__(self, partitionX, partitionY, partitionFeatureMeans):
        partitionXMin, partitionXMax = partitionX    
        partitionYMin, partitionYMax = partitionY
        gridNumPartitionX = math.floor((partitionXMax -
            partitionXMin)/self.gridLengthX)
        gridNumPartitionY = math.floor((partitionYMax -
            partitionYMin)/self.gridLengthY)
        locationXCenters = [partitionXMin + int(self.gridLengthX/2) +
                gridIndexX * self.gridLengthX for gridIndexX in
                range(gridNumPartitionX)]
        locationYCenters = [partitionYMin + int(self.gridLengthY/2) +
                gridIndexY * self.gridLengthY for gridIndexY in
                range(gridNumPartitionY)]
        gridNumPartition = gridNumPartitionX * gridNumPartitionY
        
        locationNoises = np.random.multivariate_normal([0, 0],
            [[self.gridLengthX/3, 0], [0, self.gridLengthY/3]],
            gridNumPartition)
        locationCenters = np.array(list(it.product(locationXCenters,
            locationYCenters)))
        texonsLocation = locationCenters + locationNoises
        texonsFeaturesValue = np.array([sampleTexonsFeatureValue(partitionFeatureMeans[featureName],
            self.featureStdVarinces[featureName] ** 2, gridNumPartition) for
            featureName in partitionFeatureMeans.index]).T
        
        texonsFeatureParameter = pd.DataFrame(texonsFeaturesValue, columns = partitionFeatureMeans.index)
        texonsLocationParameter = pd.DataFrame(texonsLocation, columns = ['x', 'y'])
        texonsCenterLocParameter = pd.DataFrame(locationCenters, columns = ['centerX', 'centerY'])
        texonsParameter = pd.concat([texonsFeatureParameter, texonsLocationParameter, texonsCenterLocParameter], axis = 1)
        
        return texonsParameter

def transTexonParameterToPolygenDrawArguemnt(texonsParameter):
    texonsParameter['width'] = texonsParameter['length'] * np.power(np.e, texonsParameter['logWidthLengthRatio'])
    texonsParameter['lengthRotatedProjectX'] = texonsParameter['length'] * np.cos(texonsParameter['angleRotated'])
    texonsParameter['widthRotatedProjectX'] = texonsParameter['width'] * np.sin(texonsParameter['angleRotated']) 
    texonsParameter['lengthRotatedProjectY'] = texonsParameter['length'] * np.sin(texonsParameter['angleRotated'])
    texonsParameter['widthRotatedProjectY'] = texonsParameter['width'] * np.cos(texonsParameter['angleRotated'])
    texonsParameter['vertexRightBottom'] = list(zip(np.int_(texonsParameter['x'] +(
        texonsParameter['lengthRotatedProjectX'] +
        texonsParameter['widthRotatedProjectX'])/2.0),
        np.int_(texonsParameter['y'] + (texonsParameter['lengthRotatedProjectY'] -
            texonsParameter['widthRotatedProjectY'])/2.0)))
    texonsParameter['vertexRightTop'] = list(zip(np.int_(texonsParameter['x'] + (
        texonsParameter['lengthRotatedProjectX'] -
        texonsParameter['widthRotatedProjectX'])/2.0),
        np.int_(texonsParameter['y'] + (texonsParameter['lengthRotatedProjectY'] +
            texonsParameter['widthRotatedProjectY'])/2.0)))
    texonsParameter['vertexLeftBottom'] = list(zip(np.int_(texonsParameter['x'] - (
        texonsParameter['lengthRotatedProjectX'] -
        texonsParameter['widthRotatedProjectX'])/2.0),
        np.int_(texonsParameter['y'] - (texonsParameter['lengthRotatedProjectY'] +
            texonsParameter['widthRotatedProjectY'])/2.0)))
    texonsParameter['vertexLeftTop'] = list(zip(np.int_(texonsParameter['x'] - (
        texonsParameter['lengthRotatedProjectX'] +
        texonsParameter['widthRotatedProjectX'])/2.0),
        np.int_(texonsParameter['y'] - (texonsParameter['lengthRotatedProjectY'] -
            texonsParameter['widthRotatedProjectY'])/2.0)))
    return  texonsParameter

def sampleTexonsFeatureValue(featureMean, featureVariance, texonNum):
    return np.random.normal(featureMean, featureVariance, texonNum)

class VisualizeTexons():
    def __init__(self, widthImage, heightImage):
        self.screen = pygame.display.set_mode([widthImage, heightImage])
        self.screen.fill((0, 0, 0))

    def __call__(self, texonsParameter):
        for texonIndex in range(len(texonsParameter)):
            pygame.draw.polygon(self.screen,
                    texonsParameter.iloc[texonIndex]['color'] * np.array([0, 255, 0]),
                    texonsParameter.iloc[texonIndex][['vertexLeftBottom', 'vertexLeftTop', 'vertexRightTop',
                        'vertexRightBottom']].values, 0)
        pygame.display.flip()
        pygame.image.save(self.screen, 'generateDemo4.png')

def main():
    gridLengthX = 20
    gridLengthY = 20

#    folderdir = os.path.dirname(os.path.abspath(__file__))
#    treedir = os.path.join(folderdir, 'tree')
#    assert (os.path.isdir(treedir))
#    treebasenames = [basename for basename in os.listdir(treedir) if
#    basename.endswith('.gpickle')]
#    treepathnames = [os.path.join(treedir, treebasename) for treebasename in treebasenames]
#    trees = [nx.read_pickle(treepathname) for treepathname in treepathnames]
    tree = nx.DiGraph()
    tree.add_node(1, x = [0, 200], y = [0, 160], depth = 0)
    tree.add_node(2, x = [0, 200], y = [0, 100], depth = 1)
    tree.add_node(3, x = [0, 200], y = [100, 160], depth = 1)
    tree.add_node(4, x = [0, 100], y = [0, 100], depth = 2)
    tree.add_node(5, x = [100, 200], y = [0, 100], depth = 2)
    tree.add_node(6, x = [100, 160], y = [0, 100], depth = 3)
    tree.add_node(7, x = [160, 200], y = [0, 100], depth = 3)

    tree.add_edges_from([(1,2),(1,3),(2,4),(2,5),(5,6),(5,7)])
 
    
    featureMeansRootNode = pd.DataFrame({'color': [0.5], 'length':[min(gridLengthX,
       gridLengthY)/3], 'angleRotated': [math.pi/4], 'logWidthLengthRatio':
       [-0.5]})
    tree.node[1]['featureMeans'] = featureMeansRootNode
    
    featureStdVarinces = pd.DataFrame({'color': [0.05], 'length': [1], 'angleRotated':
            [math.pi/30], 'logWidthLengthRatio': [-0.1] })
    featureUpperBound = 2*featureMeansRootNode - featureStdVarinces * 3
    featureLowerBound = featureStdVarinces * 3
    
    sampleNodeFeatureMean = SampleNodeFeatureMean(featureUpperBound,
            featureLowerBound)
    leafPartitionParameterDf = sampleNodeFeatureMean(tree)
    
    sampleTexonsLocationsAndFeatures = SampleTexonsLocationsAndFeatures(gridLengthX, gridLengthY, featureStdVarinces)
    partitionXTotal, partitionYTotal, partitionFeatureMeansTotal = leafPartitionParameterDf['x'], leafPartitionParameterDf['y'], leafPartitionParameterDf.drop(['x','y'],axis = 1)
    texonsParameterTotal = pd.concat([sampleTexonsLocationsAndFeatures(partitionXTotal.iloc[partitionIndex],
        partitionYTotal.iloc[partitionIndex], partitionFeatureMeansTotal.iloc[partitionIndex]) for partitionIndex in
        range(len(leafPartitionParameterDf))], ignore_index = True)

    texonsParameterDrawing = transTexonParameterToPolygenDrawArguemnt(texonsParameterTotal)
    __import__('ipdb').set_trace() 
    visualizeTexons = VisualizeTexons(max(tree.node[1]['x']), max(tree.node[1]['y']))
    texonsParameterDrawing.to_csv('~/segmentation-expt4/demo4.csv')
    visualizeTexons(texonsParameterDrawing) 
if __name__=="__main__":
    main()
