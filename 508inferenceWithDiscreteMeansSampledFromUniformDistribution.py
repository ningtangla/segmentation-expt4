import scipy.stats as stats
import pandas as pd
import numpy as np
import itertools as it
import networkx as nx
import math
import cv2
import pygame
import datetime 
import generateTreeWithPrior as generateTree
import generatePartitionGivenTreeWithPrior as generatePartition

class CalOnePartitionLikelihoodLog():
    def __init__(self, allDiscreteUniformFeaturesMeans, featureStdVarinces, texonsObserved):
        self.possibleChangeFeaturesOnDepthes = list(it.permutations(list(featureStdVarinces.columns)))
        self.possibleChangeFeatureDiscreteMeansOnDepthes = [[allDiscreteUniformFeaturesMeans[changeFeature] for changeFeature in changeFeaturesOnDepthes] for changeFeaturesOnDepthes in self.possibleChangeFeaturesOnDepthes]
        self.possibleChangeFeatureStdVarincesOnDepthes = [[featureStdVarinces[changeFeature] for changeFeature in changeFeaturesOnDepthes] for changeFeaturesOnDepthes in self.possibleChangeFeaturesOnDepthes]
        self.texonsObserved = texonsObserved 

    def __call__(self, partition):
        partitionLikelihoodLogUnderDiffChangeFeatureOrder = [calOnePartitionUnderOneChangeFeatureOrderLikelihoodLog(partition, self.texonsObserved, changeFeatureDiscreteMeansOnDepthes, changeFeatureStdVarincesOnDepthes, changeFeaturesOnDepthes) for changeFeatureDiscreteMeansOnDepthes, changeFeatureStdVarincesOnDepthes, changeFeaturesOnDepthes in zip(self.possibleChangeFeatureDiscreteMeansOnDepthes, self.possibleChangeFeatureStdVarincesOnDepthes, self.possibleChangeFeaturesOnDepthes)]
        partitionLikelihoodLogUnderBestFeatureChangeOrder = max(partitionLikelihoodLogUnderDiffChangeFeatureOrder)
        partitionLikelihoodLog = partitionLikelihoodLogUnderBestFeatureChangeOrder - np.log(len(self.possibleChangeFeaturesOnDepthes))
        return partitionLikelihoodLog

def calOnePartitionUnderOneChangeFeatureOrderLikelihoodLog(partition, texonsObserved, changeFeatureDiscreteMeansOnDepthes, changeFeatureStdVarincesOnDepthes, changeFeaturesOnDepthes):
    partitionNonRootNodes = [n for n,d in dict(partition.in_degree()).items() if d!=0]
    partitionNonRootNodesDepthes = [partition.node[partitionNode]['depth'] for partitionNode in partitionNonRootNodes]
    if partitionNonRootNodes != []:
        depthMax = max(partitionNonRootNodesDepthes)
    else:
        depthMax = 0
    changeFeaturesParameterDf = pd.DataFrame([[changeFeatureDiscreteMeansOnDepthes[nodeDepth - 1].values, changeFeatureStdVarincesOnDepthes[nodeDepth - 1].values, changeFeaturesOnDepthes[nodeDepth - 1]] for nodeDepth in partitionNonRootNodesDepthes], columns = ['featureDiscreteMean', 'featureStdVarince', 'feature'])
    changeFeaturesParameterDf['x'] = [partition.node[node]['partition']['x'] for node in partitionNonRootNodes]
    changeFeaturesParameterDf['y'] = [partition.node[node]['partition']['y'] for node in partitionNonRootNodes]
    
    leafNodes = [n for n,d in dict(partition.out_degree()).items() if d==0]
    notDeepstLeafNodes = list(it.chain(*[[leafNode] * (depthMax - partition.node[leafNode]['depth'])  for leafNode in leafNodes]))
    notDeepstLeafNodesDepthes = list(it.chain(*[range(partition.node[leafNode]['depth'], depthMax)  for leafNode in leafNodes]))
    changeFeaturesNodesUnchangeFeatureParameterDf = pd.DataFrame([[changeFeatureDiscreteMeansOnDepthes[nodeDepth - 1].values, changeFeatureStdVarincesOnDepthes[nodeDepth - 1].values, changeFeaturesOnDepthes[nodeDepth + 1 - 1]] for nodeDepth in notDeepstLeafNodesDepthes], columns = ['featureDiscreteMean', 'featureStdVarince', 'feature'])
    changeFeaturesNodesUnchangeFeatureParameterDf['x'] = [partition.node[node]['partition']['x'] for node in notDeepstLeafNodes]
    changeFeaturesNodesUnchangeFeatureParameterDf['y'] = [partition.node[node]['partition']['y'] for node in notDeepstLeafNodes]
    
    unchangedFeaturesDepthes = range(depthMax + 1, len(changeFeaturesOnDepthes) + 1)
    unchangeFeaturesParameterDf = pd.DataFrame([[changeFeatureDiscreteMeansOnDepthes[nodeDepth - 1].values, changeFeatureStdVarincesOnDepthes[nodeDepth - 1].values, changeFeaturesOnDepthes[nodeDepth - 1]] for nodeDepth in unchangedFeaturesDepthes], columns = ['featureDiscreteMean', 'featureStdVarince', 'feature'])
    unchangeFeaturesParameterDf['x'] = [partition.node[0]['partition']['x'] for node in unchangedFeaturesDepthes]
    unchangeFeaturesParameterDf['y'] = [partition.node[0]['partition']['y'] for node in unchangedFeaturesDepthes]
    parameterDf = pd.concat([changeFeaturesParameterDf, changeFeaturesNodesUnchangeFeatureParameterDf, unchangeFeaturesParameterDf])

    featureMeanDependentSamplesPriorLogInNonRootNodes = np.sum([np.log(len(changeFeatureDiscreteMeansOnDepthes[depth - 1])) for depth in partitionNonRootNodesDepthes])
#    if depthMax == 3:
    texonsStates = parameterDf.apply(calTexonsLikelihoodLog, args = (texonsObserved, 1), axis = 1)
    parameterDf['texonsFeatureValueLikelihoodLog'] = texonsStates
    partitionLikelihoodLogUnderOneChangeFeatureOrder = np.sum(parameterDf['texonsFeatureValueLikelihoodLog']) - featureMeanDependentSamplesPriorLogInNonRootNodes
    #nodesParameterDf['texonsLikelihoodLog'] = nodesParameterDf.apply(calTexonsLikelihoodLog, args = (texonsObserved, 1), axis = 1)
    #nodesParameterDf['texonsChangeFeatureValuesMean'], nodesParameterDf['texonsNumInPartition'] = zip(*texonsObservedStats)
    #nodesParameterDf['conjugateVarinces'] = 1.0/(1.0/np.power(nodesParameterDf['stdVarinceOfChangeFeatureMean'], 2) + nodesParameterDf['texonsNumInPartition'] / np.power(nodesParameterDf['changeFeatureStdVarince'], 2))
    #nodesParameterDf['conjugateMean'] = (nodesParameterDf['meanOfChangeFeatureMean'] / np.power(nodesParameterDf['stdVarinceOfChangeFeatureMean'], 2) + nodesParameterDf['texonsChangeFeatureValuesMean'] * nodesParameterDf['texonsNumInPartition'] / np.power(nodesParameterDf['changeFeatureStdVarince'], 2)) * nodesParameterDf['conjugateVarinces']

    #texonsUnchangedFeaturesLikelihoodLog = [texonsObserved[changeFeaturesOnDepthes[unchangedFeatureMeanDepth]].apply(lambda x: stats.norm.logpdf(x, meansOfChangeFeatureMeansOnDepthes[unchangedFeatureMeanDepth], np.power(changeFeatureStdVarincesOnDepthes[unchangedFeatureMeanDepth], 2))) for unchangedFeatureMeanDepth in unchangedFeaturesDepthes]

    #partitionLikelihoodLogUnderOneChangeFeatureOrder = np.sum(parameterDf['texonsFeatureValueLikelihoodLog'])
    #partitionLikelihoodLogUnderOneChangeFeatureOrder = np.sum(nodesParameterDf['texonsLikelihoodLog']) + np.sum(texonsUnchangedFeaturesLikelihoodLog)
    return partitionLikelihoodLogUnderOneChangeFeatureOrder

def calTexonsLikelihoodLog(row, texonsObserved, pandasOnlySupportArgNumBiggerTwoRowVectvize):
    xMin, xMax = row['x']
    yMin, yMax = row['y']
    feature = row['feature']
    featureDiscreteMean = row['featureDiscreteMean']
    texonsInPartition = texonsObserved[(texonsObserved['centerX'] > xMin) & (texonsObserved['centerX'] < xMax) & (texonsObserved['centerY'] > yMin) & (texonsObserved['centerY'] < yMax)]
    featureValueStdVarince = row['featureStdVarince'][0]
    observedTexonsFeatureValues = texonsInPartition[feature].values
    texonsFeatureValueLikelihoodLogOnDiffFeatureMean = [sum(stats.norm.logpdf(observedTexonsFeatureValues, featureMean, featureValueStdVarince)) - np.log(len(featureDiscreteMean))  for featureMean in featureDiscreteMean]
    texonsFeatureValueLikelihoodLog = max(texonsFeatureValueLikelihoodLogOnDiffFeatureMean)
    return texonsFeatureValueLikelihoodLog

class VisualizePossiblePartition():
    def __init__(self, imageWidth, imageHeight, imageWidthAllignedWithHumanAnswer, imageHeightAllignedWithHumanAnswer, blankProportion, lineWidth):
        self.imageWidth, self.imageHeight = imageWidth, imageHeight
        self.imageWidthAllignedWithHumanAnswer, self.imageHeightAllignedWithHumanAnswer = imageWidthAllignedWithHumanAnswer, imageHeightAllignedWithHumanAnswer
        self.blankProportion = blankProportion
        self.lineWidth = lineWidth

    def __call__(self, imageIndex, partition, sampleIndex):
        image = cv2.imread('generate/demo' + str(imageIndex)+'.png')
        partitionNodes = list(partition.nodes())
        partitionNodes.reverse()
        for node in partitionNodes:
            xMin, xMax = partition.node[node]['partition']['x']
            yMin, yMax = partition.node[node]['partition']['y']
            width = xMax - xMin
            height = yMax - yMin
            cv2.rectangle(image, (xMin, yMin), (xMax, yMax), (255, 255, 255), 1 + round(self.lineWidth * self.imageWidth / (self.imageWidthAllignedWithHumanAnswer * (1 - self.blankProportion))))
        
        resizeImage = cv2.resize(image,(int((1-self.blankProportion) * self.imageHeightAllignedWithHumanAnswer), int((1-self.blankProportion) * self.imageWidthAllignedWithHumanAnswer)), interpolation=cv2.INTER_CUBIC)
        inferenceImage = np.zeros([self.imageHeightAllignedWithHumanAnswer, self.imageWidthAllignedWithHumanAnswer, 3], 'uint8')
        inferenceImage[int((self.blankProportion/2) * self.imageHeightAllignedWithHumanAnswer) : int((1-self.blankProportion/2) * self.imageHeightAllignedWithHumanAnswer), int((self.blankProportion/2) * self.imageWidthAllignedWithHumanAnswer) : int((1-self.blankProportion/2) * self.imageWidthAllignedWithHumanAnswer)] = resizeImage
        cv2.rectangle(inferenceImage, (int(self.blankProportion/2 * self.imageHeightAllignedWithHumanAnswer), int(self.blankProportion/2 * self.imageWidthAllignedWithHumanAnswer)), (int((1 - self.blankProportion/2) * self.imageHeightAllignedWithHumanAnswer), int((1-self.blankProportion/2) * self.imageWidthAllignedWithHumanAnswer)), (255, 255, 255), self.lineWidth)
        cv2.imshow('image', image)
        cv2.imwrite('inference/demo' + str(imageIndex) + '_' + str(sampleIndex) + '.png', inferenceImage)
    
def main():
    parameterToKeepImageLikelihoodGivenPartitionComputable = 1200
    imageList = [4]
    
    treeNum = 1000
    gamma = 0.9
    maxDepth = 4
    alphaDirichlet = 3.4   

    imageWidthAllignedWithHumanAnswer = 720
    imageHeightAllignedWithHumanAnswer = 720
    imageWidth = 960
    imageHeight = 960
    blankProportion = 0.1
    lineWidth = 7

    gridLengthX = 60 
    gridLengthY = 60
    gridForPartitionRate = 4
    partitionInterval = {'x': gridLengthX * gridForPartitionRate, 'y': gridLengthY * gridForPartitionRate}
     
    featuresValueMax = pd.DataFrame({'color': [1], 'length':[min(gridLengthX, gridLengthY)], 'angleRotated': [math.pi], 'logWidthLengthRatio': [-1.6]}) 
    featureProportionScale = 2
    featureMappingScaleFromPropotionToValue = featuresValueMax / featureProportionScale
    "represent featureValue as proportion in range(0, ProportionScale), eg(1, 2, 3, ..10) to normalized the diff feature dimension range "
    featureMeanIntevel = 0.10 * featureProportionScale
    featureStdVarince = 0.06 * featureProportionScale
    featurePossibleMeans = np.arange(3.3 * featureStdVarince, featureProportionScale - 3.3 * featureStdVarince + 0.001, featureMeanIntevel) 
    allDiscreteUniformFeaturesMeans = pd.DataFrame([[featureMean] * len(list(featureMappingScaleFromPropotionToValue)) for featureMean in featurePossibleMeans], columns = list(featureMappingScaleFromPropotionToValue))
    featuresStdVarince = pd.DataFrame([[featureStdVarince] * len(list(featureMappingScaleFromPropotionToValue))], columns = list(featureMappingScaleFromPropotionToValue))

    treeHypothesesSpace, treeHypothesesSpacePrior = generateTree.generateNCRPTreesAndRemoveRepeatChildNodeAndMapPriorsToNewTrees(gamma, maxDepth, treeNum)
    generateDiffPartitiedTrees = generatePartition.GenerateDiffPartitiedTrees(partitionInterval, alphaDirichlet, imageWidth, imageHeight)
    partitionHypothesesSpaceGivenTreeHypothesesSpace = list(it.chain(*[generateDiffPartitiedTrees(treeHypothesis)[0] for treeHypothesis in treeHypothesesSpace]))
    partitionsPriorLog = [partitionHypothesis.node[0]['treePriorLog'] + partitionHypothesis.node[0]['partitionPriorLog'] for partitionHypothesis in partitionHypothesesSpaceGivenTreeHypothesesSpace]
    
    visualizePossiblePartition = VisualizePossiblePartition(imageWidth, imageHeight, imageWidthAllignedWithHumanAnswer, imageHeightAllignedWithHumanAnswer, blankProportion, lineWidth)                
    imagesInformationValues = []
    for imageIndex in imageList:
        texonsObserved = pd.read_csv('generate/demoUnscaled' + str(imageIndex) + '.csv')

        print(datetime.datetime.now())
        calOnePartitionLikelihoodLog = CalOnePartitionLikelihoodLog(allDiscreteUniformFeaturesMeans, featuresStdVarince, texonsObserved)
        partitionsLikelihoodLogConditionOnObservedData = [calOnePartitionLikelihoodLog(partitionHypothesis) for partitionHypothesis in partitionHypothesesSpaceGivenTreeHypothesesSpace]
        
        partitionsPosteriorLog = np.array(partitionsPriorLog) + np.array(partitionsLikelihoodLogConditionOnObservedData)
        partitionsPreNormalizedPosterior = np.exp(partitionsPosteriorLog - parameterToKeepImageLikelihoodGivenPartitionComputable) 
        partitionsPreNormalizedPosteriorSum = np.sum(partitionsPreNormalizedPosterior)
        partitionsNormalizedPosterior = partitionsPreNormalizedPosterior/partitionsPreNormalizedPosteriorSum
        
        imageInformation = - np.log2(partitionsPreNormalizedPosteriorSum)
        imagesInformationValues.append(imageInformation)
        indexDecending = np.argsort(partitionsNormalizedPosterior)
        for sampleIndex in range(16):    
            partition = partitionHypothesesSpaceGivenTreeHypothesesSpace[list(np.random.multinomial(1, partitionsNormalizedPosterior)).index(1)]
            visualizePossiblePartition(imageIndex, partition, sampleIndex)
        print(datetime.datetime.now())
        mostLikeliPartitiedTree = partitionHypothesesSpaceGivenTreeHypothesesSpace[np.argmax(partitionsNormalizedPosterior)]
        nx.write_gpickle(mostLikeliPartitiedTree, 'inference/mostLikeliPartitiedTree_' + str(imageIndex) + '.gpickle')
        
    imagesInformationDf = pd.DataFrame(imagesInformationValues, columns = ['informationContent'])
    imagesInformationDf.to_csv('inference/informationContent.csv')

if __name__ == "__main__":
    main()


