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
    def __init__(self, meansOfFeatureMeans, stdVarincesOfFeatureMeans, featureStdVarinces, texonsObserved):
        self.possibleChangeFeaturesOnDepthes = list(it.permutations(list(meansOfFeatureMeans.columns)))
        self.possibleMeansOfChangeFeatureMeansOnDepthes = [[meansOfFeatureMeans[changeFeature].values[0] for changeFeature in changeFeaturesOnDepthes] for changeFeaturesOnDepthes in self.possibleChangeFeaturesOnDepthes]    
        self.possibleStdVarincesOfChangeFeatureMeansOnDepthes = [[stdVarincesOfFeatureMeans[changeFeature].values[0] for changeFeature in changeFeaturesOnDepthes] for changeFeaturesOnDepthes in self.possibleChangeFeaturesOnDepthes]
        self.possibleChangeFeatureStdVarincesOnDepthes = [[featureStdVarinces[changeFeature].values[0] for changeFeature in changeFeaturesOnDepthes] for changeFeaturesOnDepthes in self.possibleChangeFeaturesOnDepthes]
        self.texonsObserved = texonsObserved 

    def __call__(self, partition):
        partitionLikelihoodLogUnderDiffChangeFeatureOrder = [calOnePartitionUnderOneChangeFeatureOrderLikelihoodLog(partition, self.texonsObserved, meansOfChangeFeatureMeansOnDepthes, stdVarincesOfChangeFeatureMeansOnDepthes, changeFeatureStdVarincesOnDepthes, changeFeaturesOnDepthes) for meansOfChangeFeatureMeansOnDepthes, stdVarincesOfChangeFeatureMeansOnDepthes, changeFeatureStdVarincesOnDepthes, changeFeaturesOnDepthes in zip(self.possibleMeansOfChangeFeatureMeansOnDepthes, self.possibleStdVarincesOfChangeFeatureMeansOnDepthes, self.possibleChangeFeatureStdVarincesOnDepthes, self.possibleChangeFeaturesOnDepthes)]
        partitionLikelihoodLog = np.log(np.sum(np.exp(partitionLikelihoodLogUnderDiffChangeFeatureOrder, dtype = np.float128))) - np.log(len(self.possibleChangeFeaturesOnDepthes))
        return partitionLikelihoodLog

def calOnePartitionUnderOneChangeFeatureOrderLikelihoodLog(partition, texonsObserved, meansOfChangeFeatureMeansOnDepthes, stdVarincesOfChangeFeatureMeansOnDepthes, changeFeatureStdVarincesOnDepthes, changeFeaturesOnDepthes):
    partitionNodes = list(partition.nodes())
    partitionNodesDepthes = [partition.node[partitionNode]['depth'] for partitionNode in partitionNodes]
    nodesParameterDf = pd.DataFrame([[meansOfChangeFeatureMeansOnDepthes[nodeDepth], stdVarincesOfChangeFeatureMeansOnDepthes[nodeDepth], changeFeatureStdVarincesOnDepthes[nodeDepth], changeFeaturesOnDepthes[nodeDepth]] for nodeDepth in partitionNodesDepthes], columns = ['meanOfChangeFeatureMean', 'stdVarinceOfChangeFeatureMean', 'changeFeatureStdVarince', 'changeFeature'])
    nodesParameterDf['x'] = [partition.node[node]['partition']['x'] for node in partitionNodes]
    nodesParameterDf['y'] = [partition.node[node]['partition']['y'] for node in partitionNodes] 
    nodesParameterDf.index = [0] * len(partitionNodes)
    #nodesParameterDf['texonsChangeFeatureValuesMean'],['texonsNumInPartition']] 
    texonsObservedStats = nodesParameterDf.apply(calTexonsInNodeChangeFeatureMeanAndNum, args = (texonsObserved, 1), axis = 1)
    nodesParameterDf['texonsChangeFeatureValuesMean'], nodesParameterDf['texonsNumInPartition'] = zip(*texonsObservedStats)
    nodesParameterDf['conjugateVarinces'] = 1.0/(1.0/np.power(nodesParameterDf['stdVarinceOfChangeFeatureMean'], 2) + nodesParameterDf['texonsNumInPartition'] / np.power(nodesParameterDf['changeFeatureStdVarince'], 2))
    nodesParameterDf['conjugateMean'] = (nodesParameterDf['meanOfChangeFeatureMean'] / np.power(nodesParameterDf['stdVarinceOfChangeFeatureMean'], 2) + nodesParameterDf['texonsChangeFeatureValuesMean'] * nodesParameterDf['texonsNumInPartition'] / np.power(nodesParameterDf['changeFeatureStdVarince'], 2)) * nodesParameterDf['conjugateVarinces']
    nodesParameterDf['nodeTexonsChangedFeaturesLikelihoodLog'] = stats.norm.logpdf(nodesParameterDf['conjugateMean'], nodesParameterDf['conjugateVarinces'])

    partitionNodeDepthMax = max(partitionNodesDepthes)
    #unchangedFeaturesValuesMeans = texonsObserved[unchangedFeatures].apply(
    #texonsNumTotal = len(texonsObserved)
    texonsUnchangedFeaturesLikelihoodLog = [texonsObserved[changeFeaturesOnDepthes[unchangedFeatureMeanDepth]].apply(lambda x: stats.norm.logpdf(x, meansOfChangeFeatureMeansOnDepthes[unchangedFeatureMeanDepth], np.power(changeFeatureStdVarincesOnDepthes[unchangedFeatureMeanDepth], 2))) for unchangedFeatureMeanDepth in range(partitionNodeDepthMax, len(changeFeaturesOnDepthes))]
    partitionLikelihoodLogUnderOneChangeFeatureOrder = nodesParameterDf['nodeTexonsChangedFeaturesLikelihoodLog'].sum() + np.sum(texonsUnchangedFeaturesLikelihoodLog)
    return partitionLikelihoodLogUnderOneChangeFeatureOrder

def calTexonsInNodeChangeFeatureMeanAndNum(row, texonsObserved, pandasOnlySupportArgNumBiggerTwoRowVectvize):
    xMin, xMax = row['x']
    yMin, yMax = row['y']
    changeFeature = row['changeFeature']
    texonsInPartition = texonsObserved[(texonsObserved['centerX'] > xMin) & (texonsObserved['centerX'] < xMax) & (texonsObserved['centerY'] > yMin) & (texonsObserved['centerY'] < yMax)]

    texonsChangeFeatureValuesMean = np.mean(texonsInPartition[changeFeature].values)
    texonsNumInPartition = len(texonsInPartition)
    return texonsChangeFeatureValuesMean, texonsNumInPartition

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
    stdVarincesOfFeatureMeans = featureValueMax / 10 
    featureStdVarinces = featureValueMax / 20
    treeHypothesesSpace, treeHypothesesSpacePrior = generateTree.generateNCRPTreesAndRemoveRepeatChildNodeAndMapPriorsToNewTrees(gamma, maxDepth, treeNum)
    
    generateDiffPartitiedTrees = generatePartition.GenerateDiffPartitiedTrees(partitionInterval, alphaDirichlet, imageWidth, imageHeight)
    partitionHypothesesSpaceGivenTreeHypothesesSpace = list(it.chain(*[generateDiffPartitiedTrees(treeHypothesis)[0] for treeHypothesis in treeHypothesesSpace]))
    partitionsPriorLog = [partitionHypothesis.node[0]['treePriorLog'] + partitionHypothesis.node[0]['partitionPriorLog'] for partitionHypothesis in partitionHypothesesSpaceGivenTreeHypothesesSpace]
    
    for imageIndex in range(imageNum):
        texonsObserved = pd.read_csv('~/segmentation-expt4/generate/demo' + str(imageIndex) + '.csv')
        
        print(datetime.datetime.now())
        calOnePartitionLikelihoodLog = CalOnePartitionLikelihoodLog(meansOfFeatureMeans, stdVarincesOfFeatureMeans, featureStdVarinces, texonsObserved)
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
