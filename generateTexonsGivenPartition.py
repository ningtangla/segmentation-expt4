import pygame
import math
import numpy as np
import networkx as nx
import itertools as it
import pandas as pd
import generateTreeWithPrior as generateTree
import generatePartitionGivenTreeWithPrior  as generatePartition

class SampleNodesFeatureMeans():
    def __init__(self, meanOfFeatureMeans, stdVarincesOfFeatureMeans):
        self.meanOfFeatureMeans = meanOfFeatureMeans
        self.stdVarincesOfFeatureMeans = stdVarincesOfFeatureMeans

    def __call__(self, tree):
        nonleafNodes = [n for n,d in dict(tree.out_degree()).items() if d!=0]
        featureIndex = self.meanOfFeatureMeans.columns.values
        featureName = featureIndex.copy()
        np.random.shuffle(featureName)
        treeNodes = list(tree.nodes)
        treeNodesDepthes = [tree.node[treeNode]['depth'] for treeNode in treeNodes]
        changeFeatures = [featureName[changeFeatureIndex] for changeFeatureIndex in treeNodesDepthes]
        meansOfChangeFeatureMeans = [self.meanOfFeatureMeans[changeFeature] for changeFeature in changeFeatures]
        stdVarincesOfChangeFeatureMeans = [self.stdVarincesOfFeatureMeans[changeFeature] for changeFeature in changeFeatures]
        
        featureMeansInNodes = [np.random.normal(meanOfChangeFeatureMean, stdVarinceOfFeatureMean**2) for meanOfChangeFeatureMean, stdVarinceOfFeatureMean in zip(meansOfChangeFeatureMeans, stdVarincesOfChangeFeatureMeans)]
        print(featureMeansInNodes)
        for node in treeNodes:
            parentNode = list(tree.predecessors(node))
            if parentNode:
                parentFeatureMeans = tree.node[parentNode[0]]['featureMeans'][:].copy()
            else:
                parentFeatureMeans = self.meanOfFeatureMeans.copy()
            parentFeatureMeans[changeFeatures[treeNodes.index(node)]] = featureMeansInNodes[treeNodes.index(node)]
            tree.node[node]['featureMeans'] = parentFeatureMeans.copy()
        return tree

def makeLeafNodeParametersDataFrameWithPartitionAndFeatureMean(tree):
    leafNodes = [n for n,d in dict(tree.out_degree()).items() if d==0]
    featureMeansLeafPartion = pd.concat([tree.node[leafNode]['featureMeans'] for leafNode in leafNodes])
    leafPartitionParameterDf = pd.DataFrame(featureMeansLeafPartion)
    leafPartitionParameterDf['x'] = [tree.node[leafNode]['partition']['x'] for leafNode in leafNodes]
    leafPartitionParameterDf['y'] = [tree.node[leafNode]['partition']['y'] for leafNode in leafNodes]
    return leafPartitionParameterDf

class SampleTexonsLocationsAndFeatures():
    def __init__(self, gridLengthX, gridLengthY, featureStdVarinces):
        self.gridLengthX = gridLengthX
        self.gridLengthY = gridLengthY
        self.featureStdVarinces = featureStdVarinces

    def __call__(self, partitionX, partitionY, partitionFeatureMeans):
        partitionXMin, partitionXMax = partitionX    
        partitionYMin, partitionYMax = partitionY
        gridNumPartitionX = math.floor((partitionXMax - partitionXMin)/self.gridLengthX)
        gridNumPartitionY = math.floor((partitionYMax - partitionYMin)/self.gridLengthY)
        locationXCenters = [partitionXMin + int(self.gridLengthX/2) + gridIndexX * self.gridLengthX for gridIndexX in range(gridNumPartitionX)]
        locationYCenters = [partitionYMin + int(self.gridLengthY/2) + gridIndexY * self.gridLengthY for gridIndexY in range(gridNumPartitionY)]
        gridNumPartition = gridNumPartitionX * gridNumPartitionY 
        locationNoises = np.random.multivariate_normal([0, 0], [[self.gridLengthX/3, 0], [0, self.gridLengthY/3]], gridNumPartition)
        locationCenters = np.array(list(it.product(locationXCenters, locationYCenters)))
        texonsLocation = locationCenters + locationNoises
        texonsFeaturesValue = np.array([sampleTexonsFeatureValue(partitionFeatureMeans[featureName], self.featureStdVarinces[featureName] ** 2, gridNumPartition) for featureName in partitionFeatureMeans.index]).T
        
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

class VisualizeTexonsAndPartitionTruth():
    def __init__(self, widthImage, heightImage):
        self.screen = pygame.display.set_mode([widthImage, heightImage])
        self.screen.fill((0, 0, 0))

    def __call__(self, texonsParameter, sampledPartition, imageIndex):
        for texonIndex in range(len(texonsParameter)):
            pygame.draw.polygon(self.screen,
                    texonsParameter.iloc[texonIndex]['color'] * np.array([0, 255, 0]),
                    texonsParameter.iloc[texonIndex][['vertexLeftBottom', 'vertexLeftTop', 'vertexRightTop',
                        'vertexRightBottom']].values, 0)
        pygame.display.flip()
        pygame.image.save(self.screen, 'generate/demo' + str(imageIndex) + '.png')

        leafNodes = [n for n,d in dict(sampledPartition.out_degree()).items() if d==0] 
        for node in leafNodes:
            
            xMin, xMax = sampledPartition.node[node]['partition']['x']
            yMin, yMax = sampledPartition.node[node]['partition']['y']
            width = xMax - xMin
            height = yMax - yMin
            pygame.draw.rect(self.screen, (255, 255, 255), np.array([xMin, yMin, width, height]), 3)
        pygame.display.flip()
        pygame.image.save(self.screen, 'generate/DemoTruthPartition' + str(imageIndex) + '.png')

def main():
    imageNum = 5

    treeNum = 100
    gamma = 1
    maxDepth = 3 
    alphaDirichlet = 3.5    

    imageWidth = 400
    imageHeight = 300
    gridLengthX = 25 
    gridLengthY = 25
    partitionInterval = {'x': gridLengthX, 'y': gridLengthY}
     
    meansOfFeatureMeans = pd.DataFrame({'color': [0.5], 'length':[min(gridLengthX,
       gridLengthY)/3], 'angleRotated': [math.pi/4], 'logWidthLengthRatio':
       [-0.5]})
    stdVarincesOfFeatureMeans = pd.DataFrame({'color': [0.4], 'length': [2.67], 'angleRotated':
            [math.pi/6], 'logWidthLengthRatio': [-0.8] })
    featureStdVarinces = pd.DataFrame({'color': [0.05], 'length': [0.67], 'angleRotated':
            [math.pi/30], 'logWidthLengthRatio': [-0.1] })
    featureUpperBound = 2*meansOfFeatureMeans - stdVarincesOfFeatureMeans * 3
    featureLowerBound = stdVarincesOfFeatureMeans * 3
   
    treeHypothesesSpace, treeHypothesesSpacePrior = generateTree.generateNCRPTreesAndRemoveRepeatChildNodeAndMapPriorsToNewTrees(gamma, maxDepth, treeNum)
    
    for imageIndex in range(imageNum):
        sampledTree = treeHypothesesSpace[list(np.random.multinomial(1, treeHypothesesSpacePrior)).index(1)]
        sampledTree.node[0]['partition'] = {'x': [0, imageWidth], 'y': [0, imageHeight]}

        generateDiffPartitiedTrees = generatePartition.GenerateDiffPartitiedTreesAndDiffFeaturedTrees(partitionInterval, alphaDirichlet)
        partitionGivenTreeHypothesesSpace, partititionGivenTreeHypothesesSpacePrior = generateDiffPartitiedTrees(sampledTree)
        partitionNormalizedPrior = partititionGivenTreeHypothesesSpacePrior/sum(partititionGivenTreeHypothesesSpacePrior)
        sampledPartition = partitionGivenTreeHypothesesSpace[list(np.random.multinomial(1, partitionNormalizedPrior)).index(1)]

        sampleNodesFeatureMeans = SampleNodesFeatureMeans(meansOfFeatureMeans, stdVarincesOfFeatureMeans)
        sampledFeatureMeansInPartitions = sampleNodesFeatureMeans(sampledPartition)
        leafPartitionParameterDf = makeLeafNodeParametersDataFrameWithPartitionAndFeatureMean(sampledFeatureMeansInPartitions)

        sampleTexonsGivenPartitionsAndFeatureMeans = SampleTexonsLocationsAndFeatures(gridLengthX, gridLengthY, featureStdVarinces)
        partitionXTotal, partitionYTotal, partitionFeatureMeansTotal = leafPartitionParameterDf['x'], leafPartitionParameterDf['y'], leafPartitionParameterDf.drop(['x','y'],axis = 1)
        texonsParameterTotal = pd.concat([sampleTexonsGivenPartitionsAndFeatureMeans(partitionXTotal.iloc[partitionIndex], partitionYTotal.iloc[partitionIndex], partitionFeatureMeansTotal.iloc[partitionIndex]) for partitionIndex in range(len(leafPartitionParameterDf))], ignore_index = True)

        texonsParameterDrawing = transTexonParameterToPolygenDrawArguemnt(texonsParameterTotal)
        visualizeTexons = VisualizeTexonsAndPartitionTruth(imageWidth, imageHeight)
        texonsParameterDrawing.to_csv('~/segmentation-expt4/generate/demo' + str(imageIndex) + '.csv')
        visualizeTexons(texonsParameterDrawing, sampledPartition, imageIndex) 

if __name__=="__main__":
    main()
