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
        cv2.imwrite('inference/demo' + str(imageIndex) + '_' + str(sampleIndex) + 'random.png', inferenceImage)
    
def main():
    parameterToKeepImageLikelihoodGivenPartitionComputable = 1200
    #imageList =  [0, 3, 5, 6, 8, 14, 17, 20, 22, 23, 30, 31, 32, 35, 37, 38, 39, 41, 43, 44, 45, 47, 48, 49, 53, 54, 57, 59, 60, 63, 69, 112, 123, 130, 136, 147, 160, 171, 176, 200, 207, 208, 213, 220, 223, 229, 235, 239]
    imageList = [51] 
    
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
    partitionsNum = len(partitionHypothesesSpaceGivenTreeHypothesesSpace)

    visualizePossiblePartition = VisualizePossiblePartition(imageWidth, imageHeight, imageWidthAllignedWithHumanAnswer, imageHeightAllignedWithHumanAnswer, blankProportion, lineWidth)                
    for imageIndex in imageList:

        for sampleIndex in range(16):    
            partition = partitionHypothesesSpaceGivenTreeHypothesesSpace[list(np.random.multinomial(1, [1/partitionsNum] * partitionsNum)).index(1)]
            visualizePossiblePartition(imageIndex, partition, sampleIndex)

if __name__ == "__main__":
    main()



