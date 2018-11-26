import pygame
import math
import numpy as np
import networkx as nx
import itertools as it
import pandas as pd
import colorsys
import generateTreeWithPrior as generateTree
import generatePartitionGivenTreeWithPrior  as generatePartition

class SampleNodesFeatureMeans():
    def __init__(self, allFeatureMeans):
        self.allFeatureMeans = allFeatureMeans

    def __call__(self, tree):
        nonRootNodes = [n for n,d in dict(tree.in_degree()).items() if d!=0]
        featureIndex = self.allFeatureMeans.columns.values
        featureName = featureIndex.copy()
        np.random.shuffle(featureName)
        nonRootNodesDepthes = [tree.node[treeNode]['depth'] for treeNode in nonRootNodes]
        changeFeatures = [featureName[changeFeatureIndex - 1] for changeFeatureIndex in nonRootNodesDepthes]
        nodesPossibleChangeFeatureMeans = [self.allFeatureMeans[changeFeature] for changeFeature in changeFeatures]
        for node in nonRootNodes:
            parentNode = list(tree.predecessors(node))
            parentFeatureMeans = tree.node[parentNode[0]]['featureMeans'][:].copy()
            possibleChangeFeatureMeans = nodesPossibleChangeFeatureMeans[nonRootNodes.index(node)]
            parentFeatureMeans[changeFeatures[nonRootNodes.index(node)]] = possibleChangeFeatureMeans[np.random.randint(len(possibleChangeFeatureMeans))]
            #print(parentFeatureMeans, tree.node[node]['partition'], nonRootNodes, nonRootNodesDepthes)
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
    def __init__(self, gridLengthX, gridLengthY, featureStdVarinces, featureProportionScale):
        self.gridLengthX = gridLengthX
        self.gridLengthY = gridLengthY
        self.featureStdVarinces = featureStdVarinces
        self.featureProportionScale = featureProportionScale
    
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
        texonsFeaturesValue = np.array([sampleTexonsFeatureValue(partitionFeatureMeans[featureName], self.featureStdVarinces[featureName], gridNumPartition) for featureName in partitionFeatureMeans.index]).T
        while (np.any(texonsFeaturesValue <= 0) or np.any(texonsFeaturesValue >= self.featureProportionScale)): 
            texonsFeaturesValue = np.array([sampleTexonsFeatureValue(partitionFeatureMeans[featureName], self.featureStdVarinces[featureName], gridNumPartition) for featureName in partitionFeatureMeans.index]).T
        texonsFeatureParameter = pd.DataFrame(texonsFeaturesValue, columns = partitionFeatureMeans.index)
        texonsLocationParameter = pd.DataFrame(texonsLocation, columns = ['x', 'y'])
        texonsCenterLocParameter = pd.DataFrame(locationCenters, columns = ['centerX', 'centerY'])
        texonsParameter = pd.concat([texonsFeatureParameter, texonsLocationParameter, texonsCenterLocParameter], axis = 1)
        
        return texonsParameter

def transTexonParameterToPolygenDrawArguemnt(texonsUnscaledParameter, featureMappingScaleFromPropotionToValue):
    texonsParameter = texonsUnscaledParameter.copy()
    featureNames = list(featureMappingScaleFromPropotionToValue)
    for featureName in featureNames:
        texonsParameter[featureName] = texonsParameter[featureName] * featureMappingScaleFromPropotionToValue[featureName].values
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
                    colorsys.hsv_to_rgb(texonsParameter.iloc[texonIndex]['color'], 1, 1) * np.array([255, 255, 255]),
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

    treeNum = 1000
    gamma = 0.9
    maxDepth = 4 
    alphaDirichlet = 3.4   

    imageWidth = 960
    imageHeight = 960
    gridLengthX = 60 
    gridLengthY = 60
    gridForPartitionRate = 4
    partitionInterval = {'x': gridLengthX * gridForPartitionRate, 'y': gridLengthY * gridForPartitionRate}
    
    featuresValueMax = pd.DataFrame({'color': [1], 'length':[min(gridLengthX, gridLengthY)], 'angleRotated': [math.pi], 'logWidthLengthRatio': [-1.6]}) 
    featureProportionScale = 1
    featureMappingScaleFromPropotionToValue = featuresValueMax / featureProportionScale
    
    "represent featureValue as proportion in range(0, ProportionScale), eg(1, 2, 3, ..10) to normalized the diff feature dimension range "
    featureMeanIntevel = 0.12 * featureProportionScale
    featureStdVarince = 0.07 * featureProportionScale
    featurePossibleMeans = np.arange(2 * featureStdVarince, featureProportionScale - 2 * featureStdVarince + 0.001, featureMeanIntevel) 
    
    allDiscreteUniformFeaturesMeans = pd.DataFrame([[featureMean] * len(list(featureMappingScaleFromPropotionToValue)) for featureMean in featurePossibleMeans], columns = list(featureMappingScaleFromPropotionToValue))
    featuresStdVarince = pd.DataFrame([[featureStdVarince] * len(list(featureMappingScaleFromPropotionToValue))], columns = list(featureMappingScaleFromPropotionToValue))
    featurePossibleMeansNum = len(featurePossibleMeans)
    #stdVarincesOfFeatureMeans = pd.DataFrame({'color': [0.4], 'length': [2.67], 'angleRotated': [math.pi/6], 'logWidthLengthRatio': [-0.8] })
    #featureStdVarinces = pd.DataFrame({'color': [0.05], 'length': [0.67], 'angleRotated': [math.pi/30], 'logWidthLengthRatio': [-0.1] })
   
    treeHypothesesSpace, treeHypothesesSpacePrior = generateTree.generateNCRPTreesAndRemoveRepeatChildNodeAndMapPriorsToNewTrees(gamma, maxDepth, treeNum)
    
    for imageIndex in range(imageNum):
        sampledTree = treeHypothesesSpace[list(np.random.multinomial(1, treeHypothesesSpacePrior)).index(1)]
        rootNodeFeaturesMeanList = [allDiscreteUniformFeaturesMeans[feature][np.random.randint(featurePossibleMeansNum)] for feature in list(featureMappingScaleFromPropotionToValue)]
        rootNodeFeaturesMean = pd.DataFrame([rootNodeFeaturesMeanList], columns = list(featureMappingScaleFromPropotionToValue))
        sampledTree.node[0]['featureMeans'] = rootNodeFeaturesMean
        
        generateDiffPartitiedTrees = generatePartition.GenerateDiffPartitiedTrees(partitionInterval, alphaDirichlet, imageWidth, imageHeight)
        partitionGivenTreeHypothesesSpace, partititionGivenTreeHypothesesSpacePrior = generateDiffPartitiedTrees(sampledTree)
        partitionNormalizedPrior = partititionGivenTreeHypothesesSpacePrior/sum(partititionGivenTreeHypothesesSpacePrior)
        sampledPartition = partitionGivenTreeHypothesesSpace[list(np.random.multinomial(1, partitionNormalizedPrior)).index(1)]

        sampleNodesFeatureMeans = SampleNodesFeatureMeans(allDiscreteUniformFeaturesMeans)
        sampledFeatureMeansInPartitions = sampleNodesFeatureMeans(sampledPartition)
        leafPartitionParameterDf = makeLeafNodeParametersDataFrameWithPartitionAndFeatureMean(sampledFeatureMeansInPartitions)

        sampleTexonsGivenPartitionsAndFeatureMeans = SampleTexonsLocationsAndFeatures(gridLengthX, gridLengthY, featuresStdVarince, featureProportionScale)
        partitionXTotal, partitionYTotal, partitionFeatureMeansTotal = leafPartitionParameterDf['x'], leafPartitionParameterDf['y'], leafPartitionParameterDf.drop(['x','y'],axis = 1)
        texonsParameterTotal = pd.concat([sampleTexonsGivenPartitionsAndFeatureMeans(partitionXTotal.iloc[partitionIndex], partitionYTotal.iloc[partitionIndex], partitionFeatureMeansTotal.iloc[partitionIndex]) for partitionIndex in range(len(leafPartitionParameterDf))], ignore_index = True)

        texonsParameterDrawing = transTexonParameterToPolygenDrawArguemnt(texonsParameterTotal, featureMappingScaleFromPropotionToValue)
        visualizeTexons = VisualizeTexonsAndPartitionTruth(imageWidth, imageHeight)
        texonsParameterTotal.to_csv('generate/demoUnscaled' + str(imageIndex) + '.csv')
        texonsParameterDrawing.to_csv('generate/demo' + str(imageIndex) + '.csv')
        visualizeTexons(texonsParameterDrawing, sampledPartition, imageIndex) 

if __name__=="__main__":
    main()

