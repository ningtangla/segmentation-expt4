import scipy.stats as stats
import pandas as pd
import numpy as np
import itertools as it
import networkx as nx
import math
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
        partitionLikelihoodLogUnderDiffChangeFeatureOrder, t = zip(*[calOnePartitionUnderOneChangeFeatureOrderLikelihoodLog(partition, self.texonsObserved, changeFeatureDiscreteMeansOnDepthes, changeFeatureStdVarincesOnDepthes, changeFeaturesOnDepthes) for changeFeatureDiscreteMeansOnDepthes, changeFeatureStdVarincesOnDepthes, changeFeaturesOnDepthes in zip(self.possibleChangeFeatureDiscreteMeansOnDepthes, self.possibleChangeFeatureStdVarincesOnDepthes, self.possibleChangeFeaturesOnDepthes)])
        partitionLikelihoodLogUnderBestFeatureChangeOrder = max(partitionLikelihoodLogUnderDiffChangeFeatureOrder)
        partitionLikelihoodLog = partitionLikelihoodLogUnderBestFeatureChangeOrder - np.log(len(self.possibleChangeFeaturesOnDepthes))
        z = t[partitionLikelihoodLogUnderDiffChangeFeatureOrder.index(partitionLikelihoodLogUnderBestFeatureChangeOrder)][['x', 'y', 'bestFeatureMean', 'texonsFeatureValueLikelihoodLog', 'feature']]
        return partitionLikelihoodLog, z

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
    parameterDf['texonsFeatureValueLikelihoodLog'], parameterDf['bestFeatureMean'] = zip(*texonsStates)
    partitionLikelihoodLogUnderOneChangeFeatureOrder = np.sum(parameterDf['texonsFeatureValueLikelihoodLog']) - featureMeanDependentSamplesPriorLogInNonRootNodes
    #nodesParameterDf['texonsLikelihoodLog'] = nodesParameterDf.apply(calTexonsLikelihoodLog, args = (texonsObserved, 1), axis = 1)
    #nodesParameterDf['texonsChangeFeatureValuesMean'], nodesParameterDf['texonsNumInPartition'] = zip(*texonsObservedStats)
    #nodesParameterDf['conjugateVarinces'] = 1.0/(1.0/np.power(nodesParameterDf['stdVarinceOfChangeFeatureMean'], 2) + nodesParameterDf['texonsNumInPartition'] / np.power(nodesParameterDf['changeFeatureStdVarince'], 2))
    #nodesParameterDf['conjugateMean'] = (nodesParameterDf['meanOfChangeFeatureMean'] / np.power(nodesParameterDf['stdVarinceOfChangeFeatureMean'], 2) + nodesParameterDf['texonsChangeFeatureValuesMean'] * nodesParameterDf['texonsNumInPartition'] / np.power(nodesParameterDf['changeFeatureStdVarince'], 2)) * nodesParameterDf['conjugateVarinces']

    #texonsUnchangedFeaturesLikelihoodLog = [texonsObserved[changeFeaturesOnDepthes[unchangedFeatureMeanDepth]].apply(lambda x: stats.norm.logpdf(x, meansOfChangeFeatureMeansOnDepthes[unchangedFeatureMeanDepth], np.power(changeFeatureStdVarincesOnDepthes[unchangedFeatureMeanDepth], 2))) for unchangedFeatureMeanDepth in unchangedFeaturesDepthes]

    #partitionLikelihoodLogUnderOneChangeFeatureOrder = np.sum(parameterDf['texonsFeatureValueLikelihoodLog'])
    #partitionLikelihoodLogUnderOneChangeFeatureOrder = np.sum(nodesParameterDf['texonsLikelihoodLog']) + np.sum(texonsUnchangedFeaturesLikelihoodLog)
    return partitionLikelihoodLogUnderOneChangeFeatureOrder, parameterDf

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
    return texonsFeatureValueLikelihoodLog, featureDiscreteMean[texonsFeatureValueLikelihoodLogOnDiffFeatureMean.index(texonsFeatureValueLikelihoodLog)]

class VisualizePossiblePartition():
    def __init__(self, widthImage, heightImage, imageIndex):
        self.imageIndex = imageIndex
        self.screen = pygame.display.set_mode([widthImage, heightImage])

    def __call__(self, partition, partitionP, partitionPRank):
        observedImage = pygame.image.load('generate/demo' + str(self.imageIndex) + '.png').convert()
        self.screen.blit(observedImage, (0, 0))
        pygame.display.flip()
        partitionNodes = list(partition.nodes())
        partitionNodes.reverse()
        for node in partitionNodes:
            xMin, xMax = partition.node[node]['partition']['x']
            yMin, yMax = partition.node[node]['partition']['y']
            width = xMax - xMin
            height = yMax - yMin
            pygame.draw.rect(self.screen, np.array([255, 0, 0]) / (partition.node[node]['depth'] + 1), np.array([xMin, yMin, width, height]), 3)
        pygame.display.flip()
        pygame.image.save(self.screen, 'inference/possiblePartition_' + str(self.imageIndex) + '_' + str(-1 * partitionPRank) + '_' + str(partitionP) + '.png')

    
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

    treeHypothesesSpace, treeHypothesesSpacePrior = generateTree.generateNCRPTreesAndRemoveRepeatChildNodeAndMapPriorsToNewTrees(gamma, maxDepth, treeNum)
    generateDiffPartitiedTrees = generatePartition.GenerateDiffPartitiedTrees(partitionInterval, alphaDirichlet, imageWidth, imageHeight)
    partitionHypothesesSpaceGivenTreeHypothesesSpace = list(it.chain(*[generateDiffPartitiedTrees(treeHypothesis)[0] for treeHypothesis in treeHypothesesSpace]))
    partitionsPriorLog = [partitionHypothesis.node[0]['treePriorLog'] + partitionHypothesis.node[0]['partitionPriorLog'] for partitionHypothesis in partitionHypothesesSpaceGivenTreeHypothesesSpace]
    
    for imageIndex in range(imageNum):
        texonsObserved = pd.read_csv('generate/demoUnscaled' + str(imageIndex) + '.csv')
        
        print(datetime.datetime.now())
        calOnePartitionLikelihoodLog = CalOnePartitionLikelihoodLog(allDiscreteUniformFeaturesMeans, featuresStdVarince, texonsObserved)
        partitionsLikelihoodLogConditionOnObservedData = [calOnePartitionLikelihoodLog(partitionHypothesis)[0] for partitionHypothesis in partitionHypothesesSpaceGivenTreeHypothesesSpace]
        partitionsLikelihoodLogBestChangeFeatureOrderDf = [calOnePartitionLikelihoodLog(partitionHypothesis)[1] for partitionHypothesis in partitionHypothesesSpaceGivenTreeHypothesesSpace]
        
        partitionsPosteriorLog = np.array(partitionsPriorLog) + np.array(partitionsLikelihoodLogConditionOnObservedData)
        partitionsPreNormalizedPosterior = np.exp(partitionsPosteriorLog - max(partitionsPosteriorLog))
        partitionsPreNormalizedPosteriorSum = sum(partitionsPreNormalizedPosterior)
        partitionsNormalizedPosterior = partitionsPreNormalizedPosterior/partitionsPreNormalizedPosteriorSum

        indexDecending = np.argsort(partitionsNormalizedPosterior)
        visualizePossiblePartition = VisualizePossiblePartition(imageWidth, imageHeight, imageIndex)                
        for pRankIndex in range(-3, 0):
            
            partition = partitionHypothesesSpaceGivenTreeHypothesesSpace[indexDecending[pRankIndex]]
            partitionPosterior = partitionsNormalizedPosterior[indexDecending[pRankIndex]]
            partitionBestChangeFeatureOrderDf = partitionsLikelihoodLogBestChangeFeatureOrderDf[indexDecending[pRankIndex]]
            visualizePossiblePartition(partition, partitionPosterior, pRankIndex)
                        
        print(datetime.datetime.now())
        mostLikeliPartitiedTree = partitionHypothesesSpaceGivenTreeHypothesesSpace[np.argmax(partitionsNormalizedPosterior)]
        nx.write_gpickle(mostLikeliPartitiedTree, 'inference/mostLikeliPartitiedTree_' + str(imageIndex) + '.gpickle')
    
if __name__ == "__main__":
    main()

