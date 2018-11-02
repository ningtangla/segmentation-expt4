import scipy.stats as stats
import pandas as pd
import numpy as np
import itertools as it
import networkx as nx
import math
import pygame

class GenerateDiffPartitiedTrees():
    def __init__(self, gridLengthX, gridLengthY):
        self.gridLength = {'x': gridLengthX, 'y': gridLengthY}

    def __call__(self, tree):
        directions = ['x', 'y']
        nonleafNodes = [n for n,d in dict(tree.out_degree()).items() if d!=0]
        untilCurrNonleafNodePartitiedTrees = [tree]
        for nonleafNode in nonleafNodes:
            #newestPartitiedTrees = map(lambda x: generatePossiblePartitiedTreesCurrNonleafNode(x, directions, nonleafNode, self.gridLength), untilCurrNonleafNodePartitiedTrees)
            newestPartitiedTrees = [generatePossiblePartitiedTreesCurrNonleafNode(untilCurrNodeTree, directions, nonleafNode, self.gridLength) for untilCurrNodeTree in untilCurrNonleafNodePartitiedTrees]
            untilCurrNonleafNodePartitiedTrees = list(it.chain(*newestPartitiedTrees))
        
        return untilCurrNonleafNodePartitiedTrees

def generatePossiblePartitiedTreesCurrNonleafNode(tree, directions, nonleafNode, lengthIntervals):
    childrenOfNonleafNode = list(tree.successors(nonleafNode))
    childrenNodeNum = len(childrenOfNonleafNode)
    newPossiblePartitedTrees = []

    for direction in directions:
        parentPartitionRangeDirection = tree.node[nonleafNode][direction]
        lengthIntervalDirection = lengthIntervals[direction]
        parentPartitionMinDirection, parentPartitionMaxDirection = parentPartitionRangeDirection
        parentPartitionLengthDirection = parentPartitionMaxDirection - parentPartitionMinDirection
        
        childrenIntervalsNoLimit = np.arange(lengthIntervalDirection, parentPartitionLengthDirection, lengthIntervalDirection)
        childrenPossibleIntervalsGivenParentLength = filter(lambda x: sum(x) == parentPartitionLengthDirection, it.product(childrenIntervalsNoLimit, repeat = childrenNodeNum))
        
        childrenPossiblePartitionDirection = parentPartitionMinDirection + np.array([[[sum(childrenIntervals[:childrenIndex]), sum(childrenIntervals[:childrenIndex + 1])] for childrenIndex in range(len(childrenIntervals))] for childrenIntervals in childrenPossibleIntervalsGivenParentLength])

        noDirection = directions[len(directions) - directions.index(direction) - 1]
        childrenPossiblePartitionNoDirection = np.array([tree.node[nonleafNode][noDirection]] * childrenNodeNum)
       # allPossiblePartitionsNonleafNode = list(it.product(childrenPossiblePartitionDirection, childrenPossiblePartitionNoDirection))
        #possiblePartitedTrees = list(map(lambda x, y: mergeNewPartitionsIntoTrees(tree, childrenOfNonleafNode, direction, noDirection, x, y), list(zip(*allPossiblePartitionsNonleafNode))))
        possiblePartitedTrees = [mergeNewPartitionsIntoTrees(tree, childrenOfNonleafNode, direction, noDirection, partitionDirection, childrenPossiblePartitionNoDirection) for partitionDirection in childrenPossiblePartitionDirection]
        newPossiblePartitedTrees.extend(possiblePartitedTrees)

    return newPossiblePartitedTrees

def mergeNewPartitionsIntoTrees(tree, children, direction, noDirection, possiblePartitionsDirection, possiblePartitionsNoDirection):
    treeCopy = tree.copy()
    for childIndex in range(len(children)):
        treeCopy.node[children[childIndex]][direction] = possiblePartitionsDirection[childIndex]
        treeCopy.node[children[childIndex]][noDirection] = possiblePartitionsNoDirection[childIndex]
    return treeCopy


class CalPartionLikelihoodLog():
    def __init__(self, featureUpperBound, featureLowerBound, alphaDirichlet, featureStdVarinces):
        self.sampleNodeFeatureMean = SampleNodeFeatureMean(featureUpperBound, featureLowerBound, alphaDirichlet)
        self.featureStdVarinces = featureStdVarinces

    def __call__(self, partitiedTree, featureChangeLeafNodeSampleTimes, texonsObserved):
        partitiedTreeP = 0
        for sampleIndex in range(featureChangeLeafNodeSampleTimes):
            leafPartitionParameterDf = self.sampleNodeFeatureMean(partitiedTree)
            __import__('ipdb').set_trace()
            leafPartitionParameterDf['logP'] = leafPartitionParameterDf.apply(calWithOneLeafPartionLikelihoodLog, axis = 1,
                    args = (self.featureStdVarinces, texonsObserved))
            logPAllPartitions = leafPartitionParameterDf['logP'].sum()
            partitiedTreeP = partitiedTreeP + np.exp(logPAllPartitions)
        pOnePartition = partitiedTreeP/featureChangeLeafNodeSampleTimes
        return pOnePartition

class SampleNodeFeatureMean():
    def __init__(self, featureUpperBound, featureLowerBound, alphaDirichlet):
        self.featureUpperBound = featureUpperBound
        self.featureLowerBound = featureLowerBound
        self.alphaDirichlet = alphaDirichlet

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
            parentChangeFeatureMean = parentFeatureMeans[changeFeature]
            halfLength = (np.abs(parentChangeFeatureMean - self.featureUpperBound[changeFeature]), np.abs(parentChangeFeatureMean - self.featureLowerBound[changeFeature]))
            halfChangeLength = np.min(halfLength)
            proportionFeatureRange = np.random.dirichlet([self.alphaDirichlet]
                    * len(childrenOfNonleafNode))
            childrenChangeFeatureMeans = proportionFeatureRange * 2 * halfChangeLength - halfChangeLength + parentChangeFeatureMean.values
            for child in childrenOfNonleafNode:
                parentFeatureMeans[changeFeature] = childrenChangeFeatureMeans[childrenOfNonleafNode.index(child)]
                tree.node[child]['featureMeans'] = parentFeatureMeans.copy()
        
        leafNodes = [n for n,d in dict(tree.out_degree()).items() if d==0]
        featureMeansLeafPartion = pd.concat([tree.node[leafNode]['featureMeans'] for leafNode in leafNodes])
        leafPartitionParameterDf = pd.DataFrame(featureMeansLeafPartion)
        leafPartitionParameterDf['x'] = [tree.node[leafNode]['x'] for leafNode in leafNodes]
        leafPartitionParameterDf['y'] = [tree.node[leafNode]['y'] for leafNode in leafNodes]
        return leafPartitionParameterDf

def calWithOneLeafPartionLikelihoodLog(row, featureStdVarinces, texonsObserved):
    xMin, xMax = row['x']
    yMin, yMax = row['y']
    featureNames = featureStdVarinces.columns
    featureMeans = row[featureNames] 
    texonsInPartition = texonsObserved[(texonsObserved['centerX'] > xMin) & (texonsObserved['centerX'] < xMax) & (texonsObserved['centerY'] > yMin) & (texonsObserved['centerY'] < yMax)]
    texonsFeatureValue = texonsInPartition[featureNames]
    texonsInPartition['logP'] = texonsFeatureValue.apply(calTexonLikelihoodLog, args = (featureMeans, featureStdVarinces), axis = 1)
    sumLogPWithOneLeafNode = texonsInPartition['logP'].sum()
    return sumLogPWithOneLeafNode

def calTexonLikelihoodLog(texonFeatureValue, featureMeans, featureStdVarinces):
    featureVarinces = np.power(featureStdVarinces, 2)
    cov = np.diag(featureVarinces.values[0])
    logP = stats.multivariate_normal.logpdf(texonFeatureValue, featureMeans, cov)
    return logP

def visualizePartitiedTree(partitiedTrees, screen, inputImage, pAllPartitiedTrees):
    pDraw = 0
    for treeIndex in range(len(partitiedTrees)):
        if pAllPartitiedTrees[treeIndex] > pDraw:
            tree = partitiedTrees[treeIndex]
            leafNodes = [n for n,d in dict(tree.out_degree()).items() if d==0]
            screen.blit(inputImage, [160, 200])
            for leafNode in leafNodes:
                xRange = tree.node[leafNode]['x']
                yRange = tree.node[leafNode]['y']
                pygame.draw.polygon(screen, (255, 255, 255), list(it.product(xRange, yRange)))
                pygame.display.flip()
                pygame.time.wait(100)
            pDraw = pAllPartitiedTrees[treeIndex]
        pygame.image.save(screen, 'inferenceDemo.png')
        
    
def main():
    gridLengthX = 20
    gridLengthY = 20
    
    tree = nx.DiGraph()
    tree.add_node(1, x = [0, 200], y = [0, 160], depth = 0)
    tree.add_node(2, depth = 1)
    tree.add_node(3, depth = 1)
    tree.add_node(4, depth = 2)
    tree.add_node(5, depth = 2)
    tree.add_edges_from([(1,2),(1,3),(2,4),(2,5)])
 
    
    featureMeansRootNode = pd.DataFrame({'color': [0.5], 'length':[min(gridLengthX,
       gridLengthY)/3], 'angleRotated': [math.pi/4], 'logWidthLengthRatio':
       [-0.5]})
    tree.node[1]['featureMeans'] = featureMeansRootNode
    
    featureStdVarinces = pd.DataFrame({'color': [0.05], 'length': [1], 'angleRotated':
            [math.pi/30], 'logWidthLengthRatio': [-0.1] })
    featureUpperBound = 2*featureMeansRootNode - featureStdVarinces * 3
    featureLowerBound = featureStdVarinces * 3
    
    alphaDirichlet = 1
    
    texonsObserved = pd.read_csv('demo1.csv')
    featureChangeLeafNodeSampleTimes = 100

    generateDiffPartitiedTrees = GenerateDiffPartitiedTrees(gridLengthX, gridLengthY)
    diffPartitiedTrees = generateDiffPartitiedTrees(tree)
    calPartitionLikelihood = CalPartionLikelihoodLog(featureUpperBound, featureLowerBound, alphaDirichlet,
            featureStdVarinces)
    pAllPartition = [calPartitionLikelihood(partitiedTree, featureChangeLeafNodeSampleTimes, texonsObserved) for
            partitiedTree in diffPartitiedTrees]
    mostLikeliPartitiedTree = diffPartitiedTrees[pAllPartition.index(max(pAllPartition))]
    nx.write_gpickle(mostLikeliPartitiedTree, 'mostLikeliPartitiedTree.gpickle')
    
#    pygame.init()
 #   screen = pygame.display.set_mode([160, 200])
 #   inputImage = pygame.image.load('generateDemo.png')
 #   visualizePartitiedTree(diffPartitiedTrees, screen, inputImage, pAllPartition)

if __name__ == "__main__":
    main()















