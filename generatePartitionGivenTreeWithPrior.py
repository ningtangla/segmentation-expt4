import scipy.stats as stats
import pandas as pd
import numpy as np
import itertools as it
import networkx as nx
import math
import pygame

class GenerateDiffPartitiedTrees():
    def __init__(self, partitionInterval, alphaDirichlet, imageWidth, imageHeight):
        self.generatePossiblePartitiedTreesCurrNonleafNodeAndCalPartitonLogPrior = GeneratePossiblePartitiedTreesCurrNonleafNodeAndCalPartitonLogPrior(alphaDirichlet)
        self.partitionInterval = partitionInterval
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight

    def __call__(self, tree):
        tree.node[0]['partitionPriorLog'] = 0
        tree.node[0]['partition'] = {'x': [0, self.imageWidth], 'y': [0, self.imageHeight]}
        directionNames = list(self.partitionInterval)
        nonleafNodes = [n for n,d in dict(tree.out_degree()).items() if d!=0]
        untilCurrNonleafNodePartitiedTrees = [tree]
        for nonleafNode in nonleafNodes:
            #newestPartitiedTrees = map(lambda x: generatePossiblePartitiedTreesCurrNonleafNode(x, directions, nonleafNode, self.gridLength), untilCurrNonleafNodePartitiedTrees)
            newestPartitiedTrees = [self.generatePossiblePartitiedTreesCurrNonleafNodeAndCalPartitonLogPrior(untilCurrNodeTree, directionNames, nonleafNode, self.partitionInterval) for untilCurrNodeTree in untilCurrNonleafNodePartitiedTrees]

            untilCurrNonleafNodePartitiedTrees = list(it.chain(*newestPartitiedTrees))
        allDirectionOrderPartitiedTrees = untilCurrNonleafNodePartitiedTrees
        allDirectionOrderPartitiedTreesPriors = [np.exp(tree.node[0]['partitionPriorLog']) for tree in allDirectionOrderPartitiedTrees]
        return allDirectionOrderPartitiedTrees, allDirectionOrderPartitiedTreesPriors

class GeneratePossiblePartitiedTreesCurrNonleafNodeAndCalPartitonLogPrior():
    def __init__(self, alphaDirichlet):
        self.alphaDirichlet = alphaDirichlet
    
    def __call__(self, tree, directionNames, nonleafNode, lengthIntervals):
        childrenOfNonleafNode = list(tree.successors(nonleafNode))
        childrenNodeNum = len(childrenOfNonleafNode)
        
        newPossiblePartitiedTrees = []
        for changeDirection in directionNames:
            parentPartitionChangeDirection = tree.node[nonleafNode]['partition'][changeDirection]
            lengthIntervalDirection = lengthIntervals[changeDirection]
            parentPartitionMinDirection, parentPartitionMaxDirection = parentPartitionChangeDirection
            parentPartitionLengthChangeDirection = parentPartitionMaxDirection - parentPartitionMinDirection
            
            childrenIntervalsNoLimit = np.arange(lengthIntervalDirection, parentPartitionLengthChangeDirection, lengthIntervalDirection)
            childrenPossibleIntervalsGivenParentLength = list(filter(lambda x: math.isclose(sum(x), parentPartitionLengthChangeDirection) == True, it.product(childrenIntervalsNoLimit, repeat = childrenNodeNum)))
            partitionProportionsLogPriors = [stats.dirichlet.logpdf(np.array(childrenPossibleInterval)/parentPartitionLengthChangeDirection, [self.alphaDirichlet] * childrenNodeNum) for childrenPossibleInterval in childrenPossibleIntervalsGivenParentLength]
            
            childrenPossiblePartitionDirection = parentPartitionMinDirection + np.array([[[sum(childrenIntervals[:childrenIndex]), sum(childrenIntervals[:childrenIndex + 1])] for childrenIndex in range(len(childrenIntervals))] for childrenIntervals in childrenPossibleIntervalsGivenParentLength])
            directionPartitions = list(map(lambda x: [x], tree.node[nonleafNode]['partition'].values()))
            possiblePartitions = [[possiblePartitionChangeDirection if directionIndex == directionNames.index(changeDirection) else directionPartitions[directionIndex] for directionIndex in range(len(directionPartitions))] for possiblePartitionChangeDirection in childrenPossiblePartitionDirection]
            childrenPossiblePartitions = [[dict(zip(directionNames, diffChildPartition)) for diffChildPartition in list(it.product(*partitions))] for partitions in possiblePartitions] 
            possiblePartitiedTrees = [mergeNewParameterIntoTrees(tree, childrenOfNonleafNode, childrenPartitions, 'partition', partitionProportionLogPrior) for childrenPartitions, partitionProportionLogPrior in zip(childrenPossiblePartitions, partitionProportionsLogPriors)]
            newPossiblePartitiedTrees.extend(possiblePartitiedTrees)
        return newPossiblePartitiedTrees

def mergeNewParameterIntoTrees(tree, children, childrenParameter, parameterName, parameterPrior):
    treeCopy = tree.copy()
    for childIndex in range(len(children)):
        treeCopy.node[children[childIndex]][parameterName] = childrenParameter[childIndex]
    treeCopy.node[0]['partitionPriorLog'] = treeCopy.node[0]['partitionPriorLog'] + parameterPrior
    return treeCopy

