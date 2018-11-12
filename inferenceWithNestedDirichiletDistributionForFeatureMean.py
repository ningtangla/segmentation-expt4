import scipy.stats as stats
import pandas as pd
import numpy as np
import itertools as it
import networkx as nx
import math
import pygame
import datetime

class GenerateDiffPartitiedTreesAndDiffFeaturedTrees():
    def __init__(self, partitionInterval, featureMeansProportionInterval, featureUpperBound, featureLowerBound):
        self.partitionInterval = partitionInterval
        self.featureMeansProportionInterval = featureMeansProportionInterval
        self.featureUpperBound = featureUpperBound
        self.featureLowerBound = featureLowerBound

    def __call__(self, tree):
        directionNames = list(self.partitionInterval)
        nonleafNodes = [n for n,d in dict(tree.out_degree()).items() if d!=0]
        untilCurrNonleafNodePartitiedTrees = [tree]
        for nonleafNode in nonleafNodes:
            #newestPartitiedTrees = map(lambda x: generatePossiblePartitiedTreesCurrNonleafNode(x, directions, nonleafNode, self.gridLength), untilCurrNonleafNodePartitiedTrees)
            newestPartitiedTrees = [generatePossiblePartitiedTreesCurrNonleafNode(untilCurrNodeTree, directionNames, nonleafNode, self.partitionInterval) for untilCurrNodeTree in untilCurrNonleafNodePartitiedTrees]

            untilCurrNonleafNodePartitiedTrees = list(it.chain(*newestPartitiedTrees))
        allDirectionOrderPartitiedTrees = untilCurrNonleafNodePartitiedTrees

        featureNames = list(self.featureUpperBound)
        changeFeatureOrders = list(it.permutations(featureNames))
        allChangeOrderFeaturedTrees = []
        for changeFeatures in changeFeatureOrders:
            untilCurrNonleafNodeFeaturedTrees = [tree]
            for nonleafNode in nonleafNodes:
                depthNonleafNode = tree.node[nonleafNode]['depth']
                changeFeature = changeFeatures[depthNonleafNode % len(featureNames)]
                newestFeaturedTrees = [generatePossibleFeaturedTreesCurrNonleafNode(untilCurrNodeTree, changeFeature, nonleafNode, self.featureMeansProportionInterval, self.featureUpperBound, self.featureLowerBound) for untilCurrNodeTree in untilCurrNonleafNodeFeaturedTrees]
                untilCurrNonleafNodeFeaturedTrees = list(it.chain(*newestFeaturedTrees))
            allChangeOrderFeaturedTrees.extend(untilCurrNonleafNodeFeaturedTrees)
        
        return allDirectionOrderPartitiedTrees, allChangeOrderFeaturedTrees 
        
def generatePossiblePartitiedTreesCurrNonleafNode(tree, directionNames, nonleafNode, lengthIntervals):
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
        priorPartitionPorportions = [1] * len(childrenPossibleIntervalsGivenParentLength)

        childrenPossiblePartitionDirection = parentPartitionMinDirection + np.array([[[sum(childrenIntervals[:childrenIndex]), sum(childrenIntervals[:childrenIndex + 1])] for childrenIndex in range(len(childrenIntervals))] for childrenIntervals in childrenPossibleIntervalsGivenParentLength])
        directionPartitions = list(map(lambda x: [x], tree.node[nonleafNode]['partition'].values()))
        possiblePartitions = [[possiblePartitionChangeDirection if directionIndex == directionNames.index(changeDirection) else directionPartitions[directionIndex] for directionIndex in range(len(directionPartitions))] for possiblePartitionChangeDirection in childrenPossiblePartitionDirection]
        childrenPossiblePartitions = [[dict(zip(directionNames, diffChildPartition)) for diffChildPartition in list(it.product(*partitions))] for partitions in possiblePartitions] 

        possiblePartitiedTrees = [mergeNewParameterIntoTrees(tree, childrenOfNonleafNode, childrenPartitions, 'partition', priorPartitionPorportion) for childrenPartitions, priorPartitionPorportion in zip(childrenPossiblePartitions, priorPartitionPorportions)]
        newPossiblePartitiedTrees.extend(possiblePartitiedTrees)

    return newPossiblePartitiedTrees

def generatePossibleFeaturedTreesCurrNonleafNode(tree, changeFeature, nonleafNode, featureMeanProportionInterval, featureUpperBound, featureLowerBound):
    childrenOfNonleafNode = list(tree.successors(nonleafNode))
    childrenNodeNum = len(childrenOfNonleafNode)
    alphaDirichlet = 1
    
    parentMeanChangeFeature = tree.node[nonleafNode]['featureMeans'][changeFeature]
    parentMeanHalfLengthFeatureChange = np.min((featureUpperBound[changeFeature] - parentMeanChangeFeature, parentMeanChangeFeature - featureLowerBound[changeFeature]))
    parentMeanLengthChangeFeature = 2 * parentMeanHalfLengthFeatureChange
    childrenIntervalsNoLimit = np.arange(featureMeanProportionInterval, 1, featureMeanProportionInterval)
    childrenPossibleIntervalsGivenParentLength = list(filter(lambda x: math.isclose(sum(x), 1) == True, it.product(childrenIntervalsNoLimit, repeat = childrenNodeNum)))
    priorChangeProportions = [stats.dirichlet.logpdf(childrenPossibleInterval, [alphaDirichlet] * childrenNodeNum) for childrenPossibleInterval in childrenPossibleIntervalsGivenParentLength]
    childrenPossibleMeansChangeFeature = parentMeanChangeFeature.values - parentMeanHalfLengthFeatureChange + parentMeanLengthChangeFeature * np.array([childrenIntervals for childrenIntervals in childrenPossibleIntervalsGivenParentLength])
    featureNames = list(tree.node[1]['featureMeans'])
    featureMeans = list(map(lambda x: [x], tree.node[nonleafNode]['featureMeans'][:].values[0]))
    possibleFeatureMeans = [[possibleMeanChangeFeature if featureIndex == featureNames.index(changeFeature) else featureMeans[featureIndex] for featureIndex in range(len(featureMeans))]for possibleMeanChangeFeature in childrenPossibleMeansChangeFeature]
    childrenPossibleFeatureMeans = [[pd.DataFrame([diffChildFeatureMean], columns = featureNames) for diffChildFeatureMean in list(it.product(*featureMeans))] for featureMeans in possibleFeatureMeans] 

    possibleFeaturedTrees = [mergeNewParameterIntoTrees(tree, childrenOfNonleafNode, childrenFeatureMeans, 'featureMeans', priorChangeProportion)  for childrenFeatureMeans, priorChangeProportion in zip(childrenPossibleFeatureMeans, priorChangeProportions)]
    newPossibleFeaturedTrees = possibleFeaturedTrees
    return newPossibleFeaturedTrees
        
def mergeNewParameterIntoTrees(tree, children, childrenParameter, parameterName, parameterPrior):
    treeCopy = tree.copy()
    for childIndex in range(len(children)):
        treeCopy.node[children[childIndex]][parameterName] = childrenParameter[childIndex]
        treeCopy.node[1]['priorLog'] = tree.node[1]['priorLog'] + parameterPrior
    return treeCopy

def calOnePartitiedTreeLikelihood(partitiedTree, featuredTrees, texonsObserved, featureStdVarinces):
    leafPartitionParameterDfs = [generateDiffFeatureMeansPartitiedTree(partitiedTree, featuredTree) for featuredTree in featuredTrees]
    partitiedFeaturedTreesP = [leafPartitionParameterDf.apply(calOneLeafPartitionLikelihoodLog, args = (texonsObserved, featureStdVarinces), axis = 1).sum() for leafPartitionParameterDf in leafPartitionParameterDfs]
    partitiedTreeP = np.sum(list(map(np.exp, partitiedFeaturedTreesP)))
    return partitiedTreeP

def generateDiffFeatureMeansPartitiedTree(partitiedTree, featuredTree):
    leafNodes = [n for n,d in dict(partitiedTree.out_degree()).items() if d==0]
    featureMeansLeafPartion = pd.concat([featuredTree.node[leafNode]['featureMeans'] for leafNode in leafNodes])
    leafPartitionParameterDf = pd.DataFrame(featureMeansLeafPartion)
    leafPartitionParameterDf['partition'] = [partitiedTree.node[leafNode]['partition'] for leafNode in leafNodes]
    return leafPartitionParameterDf


def calOneLeafPartitionLikelihoodLog(row, texonsObserved, featureStdVarinces):
    __import__('ipdb').set_trace()
    xMin, xMax = row['partition']['x']
    yMin, yMax = row['partition']['y']
    featureNames = featureStdVarinces.columns
    featureMeans = row[featureNames] 
    texonsInPartition = texonsObserved[(texonsObserved['centerX'] > xMin) & (texonsObserved['centerX'] < xMax) & (texonsObserved['centerY'] > yMin) & (texonsObserved['centerY'] < yMax)]
    texonsFeatureValue = texonsInPartition[featureNames]
    sumLogPWithOneLeafNode = texonsFeatureValue.apply(calTexonLikelihoodLog, args = (featureMeans, featureStdVarinces), axis = 1).sum()
    return sumLogPWithOneLeafNode

def calTexonLikelihoodLog(texonFeatureValue, featureMeans, featureStdVarinces):
    featureVarinces = np.power(featureStdVarinces, 2)
    cov = np.diag(featureVarinces.values[0])
    logP = stats.multivariate_normal.logpdf(texonFeatureValue, featureMeans, cov)
    return logP

def transTexonParameterToPolygenDrawArguemnt(texonsParameter):
    texonsParameter['width'] = texonsParameter['length'] * np.power(np.e, texonsParameter['logWidthLengthRatio'])
    texonsParameter['lengthRotatedProjectX'] = texonsParameter['length'] * np.cos(texonsParameter['angleRotated'])
    texonsParameter['widthRotatedProjectX'] = texonsParameter['width'] * np.sin(texonsParameter['angleRotated']) 
    texonsParameter['lengthRotatedProjectY'] = texonsParameter['length'] * np.sin(texonsParameter['angleRotated'])
    texonsParameter['widthRotatedProjectY'] = texonsParameter['width'] * np.cos(texonsParameter['angleRotated'])
    texonsParameter['vertexRightBottom'] = list(zip(np.int_((texonsParameter['x'] +
        texonsParameter['lengthRotatedProjectX'] +
        texonsParameter['widthRotatedProjectX'])/2.0),
        np.int_((texonsParameter['y'] + texonsParameter['lengthRotatedProjectY'] -
            texonsParameter['widthRotatedProjectY'])/2.0)))
    texonsParameter['vertexRightTop'] = list(zip(np.int_((texonsParameter['x'] +
        texonsParameter['lengthRotatedProjectX'] -
        texonsParameter['widthRotatedProjectX'])/2.0),
        np.int_((texonsParameter['y'] + texonsParameter['lengthRotatedProjectY'] +
            texonsParameter['widthRotatedProjectY'])/2.0)))
    texonsParameter['vertexLeftTop'] = list(zip(np.int_((texonsParameter['x'] -
        texonsParameter['lengthRotatedProjectX'] -
        texonsParameter['widthRotatedProjectX'])/2.0),
        np.int_((texonsParameter['y'] - texonsParameter['lengthRotatedProjectY'] +
            texonsParameter['widthRotatedProjectY'])/2.0)))
    texonsParameter['vertexLeftBottom'] = list(zip(np.int_((texonsParameter['x'] -
        texonsParameter['lengthRotatedProjectX'] +
        texonsParameter['widthRotatedProjectX'])/2.0),
        np.int_((texonsParameter['y'] - texonsParameter['lengthRotatedProjectY'] -
            texonsParameter['widthRotatedProjectY'])/2.0)))
    return  texonsParameter

def visualize(partitiedTrees, texonsParameter, indexDecending, pList, i):    
    screen = pygame.display.set_mode([200, 160])
    screen.fill((0,0,0))
    
    texonsParameter = transTexonParameterToPolygenDrawArguemnt(texonsParameter)
    for texonIndex in range(len(texonsParameter)):
        pygame.draw.polygon(screen, texonsParameter.iloc[texonIndex]['color'] * np.array([0, 255, 0]), texonsParameter.iloc[texonIndex][['vertexRightBottom', 'vertexRightTop', 'vertexLeftTop', 'vertexLeftBottom']].values, 0)
    tree = partitiedTrees[indexDecending[i]]
    leafNodes = [n for n,d in dict(tree.out_degree()).items() if d==0] 
    for node in leafNodes:
        
        xMin, xMax = tree.node[node]['partition']['x']
        yMin, yMax = tree.node[node]['partition']['y']
        xC = (xMin + xMax)/2
        yC = (yMin + yMax)/2
        width = xMax - xMin
        height = yMax - yMin
        pygame.draw.rect(screen, (255, 255, 255), np.array([xMin, yMin, width, height])/2, 3)
    p = pList[indexDecending[i]]
    pygame.display.flip()
    pygame.image.save(screen, str(-i)+'_'+str(p)+'.png')
        
    
def main():
    gridLengthX = 20
    gridLengthY = 20
    imageLengthX = 200
    imageLengthY = 160

    tree = nx.DiGraph()
    tree.add_node(1, depth = 0, priorLog = 0)
    tree.add_node(2, depth = 1)
    tree.add_node(3, depth = 1)
    tree.add_node(4, depth = 2)
    tree.add_node(5, depth = 2)
    tree.add_node(6, depth = 3)
    tree.add_node(7, depth = 3)
    tree.add_edges_from([(1,2),(1,3),(2,4),(2,5),(5,6),(5,7)])
 
    partitionRootNode = {'x':[0, imageLengthX], 'y' :[0, imageLengthY]}
    tree.node[1]['partition'] = partitionRootNode
    partitonInterval = {'x': gridLengthX, 'y': gridLengthY}

    featureMeansRootNode = pd.DataFrame({'color': [0.5], 'length':[math.ceil(min(gridLengthX, gridLengthY)/3)], 'angleRotated': [math.pi/4], 'logWidthLengthRatio': [-0.5]})
    tree.node[1]['featureMeans'] = featureMeansRootNode
    featureMeansProportionInterval = 0.25

    featureStdVarinces = pd.DataFrame({'color': [0.05], 'length': [1], 'angleRotated': [math.pi/30], 'logWidthLengthRatio': [-0.1] })
    featureUpperBound = 2 * featureMeansRootNode - featureStdVarinces * 3
    featureLowerBound = featureStdVarinces * 3
    alphaDirichlet = 1
    
    texonsObserved = pd.read_csv('~/segmentation-expt4/generate/demo2.csv')
    print(datetime.datetime.now())
    generateDiffPartitiedTreesAndDiffFeaturedTrees = GenerateDiffPartitiedTreesAndDiffFeaturedTrees(partitonInterval, featureMeansProportionInterval, featureUpperBound, featureLowerBound)
    partitiedTrees, featuredTrees = generateDiffPartitiedTreesAndDiffFeaturedTrees(tree)
    pAllPartition = [calOnePartitiedTreeLikelihood(partitiedTree, featuredTrees, texonsObserved, featureStdVarinces) for partitiedTree in partitiedTrees]
    pNormalizedAllPartition = pAllPartition/np.sum(pAllPartition)
    indexDecending = np.argsort(pNormalizedAllPartition)
    for i in range(-3, 0):
        visualize(partitiedTrees, texonsObserved, indexDecending, pNormalizedAllPartition, i)
    print(datetime.datetime.now())
    mostLikeliPartitiedTree = partitiedTrees[np.argmax(pAllPartition)]
    nx.write_gpickle(mostLikeliPartitiedTree, 'mostLikeliPartitiedTree.gpickle')
    
#    pygame.init()
 #   screen = pygame.display.set_mode([160, 200])
 #   inputImage = pygame.image.load('generateDemo.png')
 #   visualizePartitiedTree(diffPartitiedTrees, screen, inputImage, pAllPartition)

if __name__ == "__main__":
    main()
