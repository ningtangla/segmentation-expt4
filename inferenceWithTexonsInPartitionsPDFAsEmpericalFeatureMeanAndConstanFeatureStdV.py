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
    def __init__(self, featureStdVarinces, texonsObserved):
        self.possibleChangeFeaturesOnDepthes = list(it.permutations(list(featureStdVarinces.columns)))
        self.possibleChangeFeatureStdVarincesOnDepthes = [[featureStdVarinces[changeFeature].values[0] for changeFeature in changeFeaturesOnDepthes] for changeFeaturesOnDepthes in self.possibleChangeFeaturesOnDepthes]
        self.texonsObserved = texonsObserved 

    def __call__(self, partition):
        partitionLikelihoodLogUnderDiffChangeFeatureOrder = [calOnePartitionUnderOneChangeFeatureOrderLikelihoodLog(partition, self.texonsObserved, changeFeatureStdVarincesOnDepthes, changeFeaturesOnDepthes) for changeFeatureStdVarincesOnDepthes, changeFeaturesOnDepthes in zip(self.possibleChangeFeatureStdVarincesOnDepthes, self.possibleChangeFeaturesOnDepthes)]
        partitionLikelihoodLog = np.log(np.sum(np.exp(partitionLikelihoodLogUnderDiffChangeFeatureOrder, dtype = np.float128))) - np.log(len(self.possibleChangeFeaturesOnDepthes))
        return partitionLikelihoodLog

def calOnePartitionUnderOneChangeFeatureOrderLikelihoodLog(partition, texonsObserved, changeFeatureStdVarincesOnDepthes, changeFeaturesOnDepthes):
    partitionNodes = list(partition.nodes())
    partitionNodesDepthes = [partition.node[partitionNode]['depth'] for partitionNode in partitionNodes]
    unchangedFeaturesDepthes = range(max(partitionNodesDepthes) + 1, len(changeFeaturesOnDepthes))
    changeFeaturesParameterDf = pd.DataFrame([[changeFeatureStdVarincesOnDepthes[nodeDepth], changeFeaturesOnDepthes[nodeDepth]] for nodeDepth in partitionNodesDepthes], columns = ['featureStdVarince', 'feature'])
    changeFeaturesParameterDf['x'] = [partition.node[node]['partition']['x'] for node in partitionNodes]
    changeFeaturesParameterDf['y'] = [partition.node[node]['partition']['y'] for node in partitionNodes]
    unchangeFeaturesParameterDf = pd.DataFrame([[changeFeatureStdVarincesOnDepthes[nodeDepth], changeFeaturesOnDepthes[nodeDepth]] for nodeDepth in unchangedFeaturesDepthes], columns = ['featureStdVarince', 'feature'])
    unchangeFeaturesParameterDf['x'] = [partition.node[0]['partition']['x'] for node in unchangedFeaturesDepthes]
    unchangeFeaturesParameterDf['y'] = [partition.node[0]['partition']['y'] for node in unchangedFeaturesDepthes]
    parameterDf = pd.concat([changeFeaturesParameterDf, unchangeFeaturesParameterDf])
    parameterDf['texonsLikelihoodLog'] = parameterDf.apply(calTexonsLikelihoodLog, args = (texonsObserved, 1), axis = 1)
    #nodesParameterDf['texonsLikelihoodLog'] = nodesParameterDf.apply(calTexonsLikelihoodLog, args = (texonsObserved, 1), axis = 1)
    #nodesParameterDf['texonsChangeFeatureValuesMean'], nodesParameterDf['texonsNumInPartition'] = zip(*texonsObservedStats)
    #nodesParameterDf['conjugateVarinces'] = 1.0/(1.0/np.power(nodesParameterDf['stdVarinceOfChangeFeatureMean'], 2) + nodesParameterDf['texonsNumInPartition'] / np.power(nodesParameterDf['changeFeatureStdVarince'], 2))
    #nodesParameterDf['conjugateMean'] = (nodesParameterDf['meanOfChangeFeatureMean'] / np.power(nodesParameterDf['stdVarinceOfChangeFeatureMean'], 2) + nodesParameterDf['texonsChangeFeatureValuesMean'] * nodesParameterDf['texonsNumInPartition'] / np.power(nodesParameterDf['changeFeatureStdVarince'], 2)) * nodesParameterDf['conjugateVarinces']

    #texonsUnchangedFeaturesLikelihoodLog = [texonsObserved[changeFeaturesOnDepthes[unchangedFeatureMeanDepth]].apply(lambda x: stats.norm.logpdf(x, meansOfChangeFeatureMeansOnDepthes[unchangedFeatureMeanDepth], np.power(changeFeatureStdVarincesOnDepthes[unchangedFeatureMeanDepth], 2))) for unchangedFeatureMeanDepth in unchangedFeaturesDepthes]
    partitionLikelihoodLogUnderOneChangeFeatureOrder = np.sum(parameterDf['texonsLikelihoodLog'])
    #partitionLikelihoodLogUnderOneChangeFeatureOrder = np.sum(nodesParameterDf['texonsLikelihoodLog']) + np.sum(texonsUnchangedFeaturesLikelihoodLog)
    return partitionLikelihoodLogUnderOneChangeFeatureOrder

def calTexonsLikelihoodLog(row, texonsObserved, pandasOnlySupportArgNumBiggerTwoRowVectvize):
    xMin, xMax = row['x']
    yMin, yMax = row['y']
    feature = row['feature']
    texonsInPartition = texonsObserved[(texonsObserved['centerX'] > xMin) & (texonsObserved['centerX'] < xMax) & (texonsObserved['centerY'] > yMin) & (texonsObserved['centerY'] < yMax)]
    
    featureValueMean = np.mean(texonsInPartition[feature])
    featureValueStdVarince = row['featureStdVarince']
    texonsLikelihoodLog = np.sum(stats.norm.logpdf(texonsInPartition[feature], featureValueMean, featureValueStdVarince**2))
    #m = row['meanOfChangeFeatureMean']
    #tau = np.abs(row['stdVarinceOfChangeFeatureMean'])
    #eta = np.abs(row['changeFeatureStdVarince'])
    texonsNum = len(texonsInPartition)
    #likelihoodLogCostant = np.log(eta) - n * np.log(eta * (2 * math.pi)**0.5) - 0.5 * np.log(n * tau**2 + eta**2)
    #likelihoodLogMain1 = -(sum([value**2 for value in texonsChangeFeatureValues])/(2*eta**2) + m**2/(2 * tau**2))
    #likelihoodLogMain2 = ((tau * featureValueTotal / eta)**2 + (eta * m / tau)**2 + 2*featureValueTotal*m) / (2 * (n * tau**2 + eta **2))
    #texonsLikelihoodLog = likelihoodLogMain1 + likelihoodLogMain2 + likelihoodLogCostant
    return texonsLikelihoodLog

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
    gridLengthX = 20 
    gridLengthY = 20
    gridForPartitionRate = 2 
    partitionInterval = {'x': gridLengthX * gridForPartitionRate, 'y': gridLengthY * gridForPartitionRate}
     
    meansOfFeatureMeans = pd.DataFrame({'color': [0.5], 'length':[min(gridLengthX, gridLengthY)/3], 'angleRotated': [math.pi/2], 'logWidthLengthRatio': [-0.7]})
    featureValueMax = meansOfFeatureMeans * 2
    stdVarincesOfFeatureMeans = featureValueMax * 100 
    featureStdVarinces = featureValueMax / 20
    treeHypothesesSpace, treeHypothesesSpacePrior = generateTree.generateNCRPTreesAndRemoveRepeatChildNodeAndMapPriorsToNewTrees(gamma, maxDepth, treeNum)
    
    generateDiffPartitiedTrees = generatePartition.GenerateDiffPartitiedTrees(partitionInterval, alphaDirichlet, imageWidth, imageHeight)
    partitionHypothesesSpaceGivenTreeHypothesesSpace = list(it.chain(*[generateDiffPartitiedTrees(treeHypothesis)[0] for treeHypothesis in treeHypothesesSpace]))
    partitionsPriorLog = [partitionHypothesis.node[0]['treePriorLog'] + partitionHypothesis.node[0]['partitionPriorLog'] for partitionHypothesis in partitionHypothesesSpaceGivenTreeHypothesesSpace]
    
    for imageIndex in range(imageNum):
        texonsObserved = pd.read_csv('~/segmentation-expt4/generate/demo' + str(imageIndex) + '.csv')
        
        print(datetime.datetime.now())
        calOnePartitionLikelihoodLog = CalOnePartitionLikelihoodLog(featureStdVarinces, texonsObserved)
        partitionsLikelihoodLogConditionOnObservedData = [calOnePartitionLikelihoodLog(partitionHypothesis) for partitionHypothesis in partitionHypothesesSpaceGivenTreeHypothesesSpace]
        
        partitionsPosterior = np.exp(np.array(partitionsPriorLog) + np.array(partitionsLikelihoodLogConditionOnObservedData))
        partitionsNormalizedPosterior = partitionsPosterior/np.sum(partitionsPosterior)

        indexDecending = np.argsort(partitionsNormalizedPosterior)
        visualizePossiblePartition = VisualizePossiblePartition(
imageWidth, imageHeight, imageIndex)                
        for pRankIndex in range(-3, 0):
            partition = partitionHypothesesSpaceGivenTreeHypothesesSpace[indexDecending[pRankIndex]]
            partitionPosterior = partitionsNormalizedPosterior[indexDecending[pRankIndex]]
            visualizePossiblePartition(partition, partitionPosterior, pRankIndex)
                    
        print(datetime.datetime.now())
        mostLikeliPartitiedTree = partitionHypothesesSpaceGivenTreeHypothesesSpace[np.argmax(partitionsNormalizedPosterior)]
        nx.write_gpickle(mostLikeliPartitiedTree, 'inference/mostLikeliPartitiedTree_' + str(imageIndex) + '.gpickle')
    
if __name__ == "__main__":
    main()

