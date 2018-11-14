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
    def __init__(self, featureValueMaxs, featureStdVarinces, texonsObserved):
        self.possibleChangeFeaturesOnDepthes = list(it.permutations(list(featureStdVarinces.columns)))
        self.possibleChangeFeatureValueMaxsOnDepthes = [[featureValueMaxs[changeFeature].values[0] for changeFeature in changeFeaturesOnDepthes] for changeFeaturesOnDepthes in self.possibleChangeFeaturesOnDepthes]
        self.possibleChangeFeatureStdVarincesOnDepthes = [[featureStdVarinces[changeFeature].values[0] for changeFeature in changeFeaturesOnDepthes] for changeFeaturesOnDepthes in self.possibleChangeFeaturesOnDepthes]
        self.texonsObserved = texonsObserved 

    def __call__(self, partition):
        partitionLikelihoodLogUnderDiffChangeFeatureOrder, t = zip(*[calOnePartitionUnderOneChangeFeatureOrderLikelihoodLog(partition, self.texonsObserved, changeFeatureValueMaxOnDepthes, changeFeatureStdVarincesOnDepthes, changeFeaturesOnDepthes) for changeFeatureValueMaxOnDepthes, changeFeatureStdVarincesOnDepthes, changeFeaturesOnDepthes in zip(self.possibleChangeFeatureValueMaxsOnDepthes, self.possibleChangeFeatureStdVarincesOnDepthes, self.possibleChangeFeaturesOnDepthes)])
        partitionLikelihoodLog = max(partitionLikelihoodLogUnderDiffChangeFeatureOrder) - np.log(len(self.possibleChangeFeaturesOnDepthes))
        return partitionLikelihoodLog

def calOnePartitionUnderOneChangeFeatureOrderLikelihoodLog(partition, texonsObserved, changeFeatureValueMaxOnDepthes, changeFeatureStdVarincesOnDepthes, changeFeaturesOnDepthes):
    partitionNodes = list(partition.nodes())
    partitionNodesDepthes = [partition.node[partitionNode]['depth'] for partitionNode in partitionNodes]
    unchangedFeaturesDepthes = range(max(partitionNodesDepthes) + 1, len(changeFeaturesOnDepthes))
    changeFeaturesParameterDf = pd.DataFrame([[changeFeatureValueMaxOnDepthes[nodeDepth], changeFeatureStdVarincesOnDepthes[nodeDepth], changeFeaturesOnDepthes[nodeDepth]] for nodeDepth in partitionNodesDepthes], columns = ['featureValueMax', 'featureStdVarince', 'feature'])
    changeFeaturesParameterDf['x'] = [partition.node[node]['partition']['x'] for node in partitionNodes]
    changeFeaturesParameterDf['y'] = [partition.node[node]['partition']['y'] for node in partitionNodes]
    unchangeFeaturesParameterDf = pd.DataFrame([[changeFeatureValueMaxOnDepthes[nodeDepth], changeFeatureStdVarincesOnDepthes[nodeDepth], changeFeaturesOnDepthes[nodeDepth]] for nodeDepth in unchangedFeaturesDepthes], columns = ['featureValueMax', 'featureStdVarince', 'feature'])
    unchangeFeaturesParameterDf['x'] = [partition.node[0]['partition']['x'] for node in unchangedFeaturesDepthes]
    unchangeFeaturesParameterDf['y'] = [partition.node[0]['partition']['y'] for node in unchangedFeaturesDepthes]
    parameterDf = pd.concat([changeFeaturesParameterDf, unchangeFeaturesParameterDf])
    texonsStates = parameterDf.apply(calTexonsLikelihoodLog, args = (texonsObserved, 1), axis = 1)
    parameterDf['texonsFeatureValueLikelihoodLog'], parameterDf['bestFeatureMean'] = zip(*texonsStates)
    #nodesParameterDf['texonsLikelihoodLog'] = nodesParameterDf.apply(calTexonsLikelihoodLog, args = (texonsObserved, 1), axis = 1)
    #nodesParameterDf['texonsChangeFeatureValuesMean'], nodesParameterDf['texonsNumInPartition'] = zip(*texonsObservedStats)
    #nodesParameterDf['conjugateVarinces'] = 1.0/(1.0/np.power(nodesParameterDf['stdVarinceOfChangeFeatureMean'], 2) + nodesParameterDf['texonsNumInPartition'] / np.power(nodesParameterDf['changeFeatureStdVarince'], 2))
    #nodesParameterDf['conjugateMean'] = (nodesParameterDf['meanOfChangeFeatureMean'] / np.power(nodesParameterDf['stdVarinceOfChangeFeatureMean'], 2) + nodesParameterDf['texonsChangeFeatureValuesMean'] * nodesParameterDf['texonsNumInPartition'] / np.power(nodesParameterDf['changeFeatureStdVarince'], 2)) * nodesParameterDf['conjugateVarinces']

    #texonsUnchangedFeaturesLikelihoodLog = [texonsObserved[changeFeaturesOnDepthes[unchangedFeatureMeanDepth]].apply(lambda x: stats.norm.logpdf(x, meansOfChangeFeatureMeansOnDepthes[unchangedFeatureMeanDepth], np.power(changeFeatureStdVarincesOnDepthes[unchangedFeatureMeanDepth], 2))) for unchangedFeatureMeanDepth in unchangedFeaturesDepthes]
    partitionLikelihoodLogUnderOneChangeFeatureOrder = np.sum(parameterDf['texonsFeatureValueLikelihoodLog'])
    #partitionLikelihoodLogUnderOneChangeFeatureOrder = np.sum(nodesParameterDf['texonsLikelihoodLog']) + np.sum(texonsUnchangedFeaturesLikelihoodLog)
    return partitionLikelihoodLogUnderOneChangeFeatureOrder, parameterDf

def calTexonsLikelihoodLog(row, texonsObserved, pandasOnlySupportArgNumBiggerTwoRowVectvize):
    xMin, xMax = row['x']
    yMin, yMax = row['y']
    feature = row['feature']
    featureValueMax = row['featureValueMax']
    texonsInPartition = texonsObserved[(texonsObserved['centerX'] > xMin) & (texonsObserved['centerX'] < xMax) & (texonsObserved['centerY'] > yMin) & (texonsObserved['centerY'] < yMax)]
    featureValueStdVarince = row['featureStdVarince']
    observedTexonsFeatureValues = texonsInPartition[feature].values
    featureMeanDiscrete = np.arange(0.05, 1, 0.1) * featureValueMax
    texonsFeatureValueLikelihoodLogOnDiffFeatureMean = [sum(stats.norm.logpdf(observedTexonsFeatureValues, featureMean, featureValueStdVarince**2)) - np.log(10)  for featureMean in featureMeanDiscrete]
    texonsFeatureValueLikelihoodLog = max(texonsFeatureValueLikelihoodLogOnDiffFeatureMean)
    return texonsFeatureValueLikelihoodLog, featureMeanDiscrete[texonsFeatureValueLikelihoodLogOnDiffFeatureMean.index(texonsFeatureValueLikelihoodLog)]

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
    imageNum = 2 

    treeNum = 100
    gamma = 1
    maxDepth = 3 
    alphaDirichlet = 3.5    

    imageWidth = 320
    imageHeight = 320
    gridLengthX = 40 
    gridLengthY = 40
    gridForPartitionRate = 1 
    partitionInterval = {'x': gridLengthX * gridForPartitionRate, 'y': gridLengthY * gridForPartitionRate}
     
    meansOfFeatureMeans = pd.DataFrame({'color': [0.5], 'length':[min(gridLengthX, gridLengthY)/3], 'angleRotated': [math.pi/2], 'logWidthLengthRatio': [-0.8]})
    featureValueMaxs = meansOfFeatureMeans * 2
    stdVarincesOfFeatureMeans = featureValueMaxs / 10 
    featureStdVarinces = pd.DataFrame({'color': [0.05], 'length':[0.05], 'angleRotated': [0.05], 'logWidthLengthRatio': [-0.05]})
    #featureStdVarinces = pd.DataFrame({'color': [0.001], 'length':[1], 'angleRotated': [math.pi/1000], 'logWidthLengthRatio': [-0.001]})
    treeHypothesesSpace, treeHypothesesSpacePrior = generateTree.generateNCRPTreesAndRemoveRepeatChildNodeAndMapPriorsToNewTrees(gamma, maxDepth, treeNum)
    
    generateDiffPartitiedTrees = generatePartition.GenerateDiffPartitiedTrees(partitionInterval, alphaDirichlet, imageWidth, imageHeight)
    partitionHypothesesSpaceGivenTreeHypothesesSpace = list(it.chain(*[generateDiffPartitiedTrees(treeHypothesis)[0] for treeHypothesis in treeHypothesesSpace]))
    partitionsPriorLog = [partitionHypothesis.node[0]['treePriorLog'] + partitionHypothesis.node[0]['partitionPriorLog'] for partitionHypothesis in partitionHypothesesSpaceGivenTreeHypothesesSpace]
    
    for imageIndex in range(imageNum):
        texonsObserved = pd.read_csv('~/segmentation-expt4/generate/demo' + str(imageIndex) + '.csv')
        
        print(datetime.datetime.now())
        calOnePartitionLikelihoodLog = CalOnePartitionLikelihoodLog(featureValueMaxs, featureStdVarinces, texonsObserved)
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

