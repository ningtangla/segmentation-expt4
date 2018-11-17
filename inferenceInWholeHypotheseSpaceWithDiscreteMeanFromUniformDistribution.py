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
    def __init__(self, possibleFeaturesMeans, featureStdVarince, possibleChangeFeatureOrderCombinations, texonsObserved):
        self.possibleFeaturesMeans = possibleFeaturesMeans
        self.featureStdVarince = featureStdVarince
        self.possibleChangeFeatureOrderCombinations = possibleChangeFeatureOrderCombinations
        self.texonsObserved = texonsObserved 
        self.featuresNum = len(list(possibleChangeFeatureOrderCombinations[0]))

    def __call__(self, partition):
        partitionNodes = list(partition.nodes())
        partitionNodesDepthes = [partition.node[partitionNode]['depth'] for partitionNode in partitionNodes]
        unchangedFeaturesDepthes = range(max(partitionNodesDepthes) + 1, self.featuresNum)
        possibleFeaturesOrder = [[featureOrder[nodeDepth] for nodeDepth in partitionNodesDepthes] + [featureOrder[nodeDepth] for nodeDepth in unchangedFeaturesDepthes] for featureOrder in self.possibleChangeFeatureOrderCombinations]
        partitionX = [partition.node[node]['partition']['x'] for node in partitionNodes] + [partition.node[0]['partition']['x'] for unchangeFeaturePartitionIndex in unchangedFeaturesDepthes]
        partitionY = [partition.node[node]['partition']['y'] for node in partitionNodes] + [partition.node[0]['partition']['y'] for unchangeFeaturePartitionIndex in unchangedFeaturesDepthes]
        diffParameterPartitionsNum = len(possibleFeaturesOrder[0])

        possibleFeatureMeansCombinations = list(it.product(self.possibleFeaturesMeans, repeat = diffParameterPartitionsNum))
        partitionLikelihoodLogUnderDiffChangeFeatureOrderAndDiffFeatureMean = [calOnePartitionUnderOneChangeFeatureOrderAndOneFeatureMeanLikelihoodLog(self.texonsObserved, featureDiscreteMeans, self.featureStdVarince, featureNames, partitionX, partitionY) for featureDiscreteMeans, featureNames in list(it.product(possibleFeatureMeansCombinations, possibleFeaturesOrder))]
        
        partitionLikelihoodLog = np.log(np.sum(np.exp(partitionLikelihoodLogUnderDiffChangeFeatureOrderAndDiffFeatureMean))/len(partitionLikelihoodLogUnderDiffChangeFeatureOrderAndDiffFeatureMean))
#        z = [tt[['bestFeatureMean', 'texonsFeatureValueLikelihoodLog', 'feature']] for tt in t]
#        print(z)
#        __import__('ipdb').set_trace()
        return partitionLikelihoodLog

def calOnePartitionUnderOneChangeFeatureOrderAndOneFeatureMeanLikelihoodLog(texonsObserved, featureDiscreteMeans, featureStdVarince, featureNames, partitionX, partitionY):
    parameterDf = pd.DataFrame(list(zip(featureDiscreteMeans, [featureStdVarince] * len(featureDiscreteMeans), featureNames, partitionX, partitionY)), columns = ['featureDiscreteMean', 'featureStdVarince', 'feature', 'x', 'y'])
    parameterDf['texonsFeatureValueLikelihood'] = parameterDf.apply(calTexonsLikelihoodLog, args = (texonsObserved, 1), axis = 1)
    #nodesParameterDf['texonsLikelihoodLog'] = nodesParameterDf.apply(calTexonsLikelihoodLog, args = (texonsObserved, 1), axis = 1)
    #nodesParameterDf['texonsChangeFeatureValuesMean'], nodesParameterDf['texonsNumInPartition'] = zip(*texonsObservedStats)
    #nodesParameterDf['conjugateVarinces'] = 1.0/(1.0/np.power(nodesParameterDf['stdVarinceOfChangeFeatureMean'], 2) + nodesParameterDf['texonsNumInPartition'] / np.power(nodesParameterDf['changeFeatureStdVarince'], 2))
    #nodesParameterDf['conjugateMean'] = (nodesParameterDf['meanOfChangeFeatureMean'] / np.power(nodesParameterDf['stdVarinceOfChangeFeatureMean'], 2) + nodesParameterDf['texonsChangeFeatureValuesMean'] * nodesParameterDf['texonsNumInPartition'] / np.power(nodesParameterDf['changeFeatureStdVarince'], 2)) * nodesParameterDf['conjugateVarinces']

    #texonsUnchangedFeaturesLikelihoodLog = [texonsObserved[changeFeaturesOnDepthes[unchangedFeatureMeanDepth]].apply(lambda x: stats.norm.logpdf(x, meansOfChangeFeatureMeansOnDepthes[unchangedFeatureMeanDepth], np.power(changeFeatureStdVarincesOnDepthes[unchangedFeatureMeanDepth], 2))) for unchangedFeatureMeanDepth in unchangedFeaturesDepthes]

    partitionLikelihoodLogUnderOneChangeFeatureOrder = np.sum(parameterDf['texonsFeatureValueLikelihood'])
    #partitionLikelihoodLogUnderOneChangeFeatureOrder = np.sum(nodesParameterDf['texonsLikelihoodLog']) + np.sum(texonsUnchangedFeaturesLikelihoodLog)
    return partitionLikelihoodLogUnderOneChangeFeatureOrder

def calTexonsLikelihoodLog(row, texonsObserved, pandasOnlySupportArgNumBiggerTwoRowVectvize):
    xMin, xMax = row['x']
    yMin, yMax = row['y']
    feature = row['feature']
    featureDiscreteMean = row['featureDiscreteMean']
    featureValueStdVarince = row['featureStdVarince']
    texonsInPartition = texonsObserved[(texonsObserved['centerX'] > xMin) & (texonsObserved['centerX'] < xMax) & (texonsObserved['centerY'] > yMin) & (texonsObserved['centerY'] < yMax)]
    observedTexonsFeatureValues = texonsInPartition[feature].values
    texonsFeatureValueLikelihoodLog = sum(stats.norm.logpdf(observedTexonsFeatureValues, featureDiscreteMean, featureValueStdVarince))
    return texonsFeatureValueLikelihoodLog

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

    treeNum = 100
    gamma = 0.9
    maxDepth = 3 
    alphaDirichlet = 3.3    

    imageWidth = 400
    imageHeight = 400
    gridLengthX = 40 
    gridLengthY = 40
    gridForPartitionRate = 2
    partitionInterval = {'x': gridLengthX * gridForPartitionRate, 'y': gridLengthY * gridForPartitionRate}
     
    featuresValueMax = pd.DataFrame({'color': [1], 'length':[min(gridLengthX, gridLengthY)], 'angleRotated': [math.pi], 'logWidthLengthRatio': [-1.6]}) 
    featureProportionScale = 4
    featureMappingScaleFromPropotionToValue = featuresValueMax / featureProportionScale
    "represent featureValue as proportion in range(0, ProportionScale), eg(1, 2, 3, ..10) to normalized the diff feature dimension range "
    featureMeanIntevel = 0.2 * featureProportionScale
    featureStdVarince = 0.1 * featureProportionScale
    possibleFeatureMeans = np.arange(2.5 * featureStdVarince, featureProportionScale - 2.5 * featureStdVarince + 0.001, featureMeanIntevel) 
    #allDiscreteUniformFeaturesMeans = pd.DataFrame([[featureMean] * len(list(featureMappingScaleFromPropotionToValue)) for featureMean in possibleFeatureMeans], columns = list(featureMappingScaleFromPropotionToValue))
    featureStdVarinces = pd.DataFrame([[featureStdVarince] * len(list(featureMappingScaleFromPropotionToValue))], columns = list(featureMappingScaleFromPropotionToValue))
    
    possibleChangeFeatureOrderCombinations = list(it.permutations(list(featureStdVarinces)))

    treeHypothesesSpace, treeHypothesesSpacePrior = generateTree.generateNCRPTreesAndRemoveRepeatChildNodeAndMapPriorsToNewTrees(gamma, maxDepth, treeNum)
    
    generateDiffPartitiedTrees = generatePartition.GenerateDiffPartitiedTrees(partitionInterval, alphaDirichlet, imageWidth, imageHeight)
    partitionHypothesesSpaceGivenTreeHypothesesSpace = list(it.chain(*[generateDiffPartitiedTrees(treeHypothesis)[0] for treeHypothesis in treeHypothesesSpace]))
    partitionsPriorLog = [partitionHypothesis.node[0]['treePriorLog'] + partitionHypothesis.node[0]['partitionPriorLog'] for partitionHypothesis in partitionHypothesesSpaceGivenTreeHypothesesSpace]
    
    for imageIndex in range(imageNum):
        texonsObserved = pd.read_csv('~/segmentation-expt4/generate/demoUnscaled' + str(imageIndex) + '.csv')
        
        print(datetime.datetime.now())
        calOnePartitionLikelihoodLog = CalOnePartitionLikelihoodLog(possibleFeatureMeans, featureStdVarince, possibleChangeFeatureOrderCombinations, texonsObserved)
        partitionsLikelihoodLogConditionOnObservedData = [calOnePartitionLikelihoodLog(partitionHypothesis) for partitionHypothesis in partitionHypothesesSpaceGivenTreeHypothesesSpace]
        
        partitionsPosteriorLog = np.array(partitionsPriorLog) + np.array(partitionsLikelihoodLogConditionOnObservedData)
        partitionsNormalizedPosterior = np.exp(partitionsPosteriorLog - max(partitionsPosteriorLog))

        indexDecending = np.argsort(partitionsNormalizedPosterior)
        visualizePossiblePartition = VisualizePossiblePartition(imageWidth, imageHeight, imageIndex)                
        for pRankIndex in range(-3, 0):
            
            partition = partitionHypothesesSpaceGivenTreeHypothesesSpace[indexDecending[pRankIndex]]
            partitionPosterior = partitionsNormalizedPosterior[indexDecending[pRankIndex]]
            visualizePossiblePartition(partition, partitionPosterior, pRankIndex)
                        
        print(datetime.datetime.now())
        mostLikeliPartitiedTree = partitionHypothesesSpaceGivenTreeHypothesesSpace[np.argmax(partitionsNormalizedPosterior)]
        nx.write_gpickle(mostLikeliPartitiedTree, 'inference/mostLikeliPartitiedTree_' + str(imageIndex) + '.gpickle')
    
if __name__ == "__main__":
    main()

