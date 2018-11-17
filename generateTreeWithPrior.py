import networkx as nx
import numpy as np
import pandas as pd
import itertools as it
import functools as ft
import math
import operator as op
from scipy import misc
import matplotlib.pyplot as pyplot
from scipy.special import gamma as gammaFunction

def generateNCRPTreesAndRemoveRepeatChildNodeAndMapPriorsToNewTrees(gamma, maxDepth, treeNum):
    trees = [sampleTreeUsingNestedChinsesResturantProcess(gamma, maxDepth) for treeIndex in range(treeNum)]
    uniqueRepeatChildTrees, uniqueRepeatChildTreesPriors = mergeDiffRepeatChildTrees(trees, gamma)
    noRepeatChildTrees, noRepeatChildTreesPriors = removeRepeatChildNodeAndMapPriorsToNewTrees(uniqueRepeatChildTrees, uniqueRepeatChildTreesPriors)
    for treeIndex in range(len(noRepeatChildTrees)):
        noRepeatChildTrees[treeIndex].node[0]['treePriorLog'] = np.log(noRepeatChildTreesPriors[treeIndex])
    return noRepeatChildTrees, noRepeatChildTreesPriors

def sampleTreeUsingNestedChinsesResturantProcess(gamma, maxDepth):
    guestsNumToNCRP = maxDepth
    tree = createNewTree(guestsNumToNCRP)
    sampleBranchUsingChinsesResturantProcess = SampleBranchUsingChinsesResturantProcess(gamma)
    currNode = 0 
    while currNode <= max(list(tree.nodes())):
        if tree.node[currNode]['depth'] < maxDepth - 1:
            tree = sampleBranchUsingChinsesResturantProcess(tree, currNode)
        currNode = currNode + 1
    return tree

def createNewTree(guestsNumToNCRP):
    newTree = nx.DiGraph()
    newTree.add_node(0, guests = list(range(guestsNumToNCRP)), depth = 0)
    return newTree

class SampleBranchUsingChinsesResturantProcess():
    def __init__(self, gamma):
        self.gamma = gamma
    def __call__(self, tree, currNode):
        guestsInNode = tree.node[currNode]['guests'].copy()
        guestsNum = len(guestsInNode)
        tables = []
        tables = assignTableForGuests(guestsInNode, tables, self.gamma)
        depth = tree.node[currNode]['depth'] + 1
        nodesNum = len(tables)
        depths = [depth] * nodesNum
       
        maxNodeIndex = max(list(tree.nodes()))
        newNodes = list(range(maxNodeIndex + 1, maxNodeIndex + 1 + nodesNum))
        tree.add_edges_from(list(it.product([currNode], newNodes)))
        attributesDf = pd.DataFrame(list(zip(tables, depths)), columns = ['guests', 'depth'], index = newNodes)        
        nx.set_node_attributes(tree, attributesDf['guests'], 'guests') 
        nx.set_node_attributes(tree, attributesDf['depth'], 'depth') 
        return tree

def assignTableForGuests(guests, tables, gamma):
    if guests == []:
        return tables
    currGuest = guests[0]
    tableAssignedPMF = calTablesAssignedPMF(tables, gamma)
    tableAssigned = list(np.random.multinomial(1, tableAssignedPMF)).index(1)
    if tableAssigned == len(tables):
        tables.append([])
    tables[tableAssigned].extend([currGuest])
    guests.remove(currGuest)
    return assignTableForGuests(guests, tables, gamma)

def calTablesAssignedPMF(tables, gamma):
    tableGuestsNum = [len(table) for table in tables]
    unNormalizedPMF = tableGuestsNum.copy()
    unNormalizedPMF.extend([gamma])
    tableAssignedPMF = np.array(unNormalizedPMF)/np.sum(unNormalizedPMF)
    return tableAssignedPMF

def mergeDiffRepeatChildTrees(trees, gamma):
    guestsNumInChildrenOfNonLeafNodesForTrees = [calGuestsNumInChildrenOfNonLeafNodes(tree) for tree in trees]
    guestsNumInChildrenOfNonLeafNodesForTreesForUniqueSelection = guestsNumInChildrenOfNonLeafNodesForTrees.copy()
    guestsNumInChildrenOfNonLeafNodesForTreesForUniqueSelection.sort()
    uniqueGuestsNumInChildrenOfNonLeafNodesForTrees = list(guestsNumInChildrenOfNonLeafNodesForTreesForUniqueSelection for guestsNumInChildrenOfNonLeafNodesForTreesForUniqueSelection,_ in it.groupby(guestsNumInChildrenOfNonLeafNodesForTreesForUniqueSelection))
    uniqueRepeatChildTrees = [trees[index] for index in [guestsNumInChildrenOfNonLeafNodesForTrees.index(uniqueGuestsNumInChildrenOfNonLeafNodesForTree) for uniqueGuestsNumInChildrenOfNonLeafNodesForTree in uniqueGuestsNumInChildrenOfNonLeafNodesForTrees]]
    
    uniqueRepeatChildTreesPriors = [ft.reduce(op.mul, [calNonLeafNodePrior(guestsNumInChildrenOfNonLeafNode, gamma) for guestsNumInChildrenOfNonLeafNode in guestsNumInChildrenOfNonLeafNodes]) for guestsNumInChildrenOfNonLeafNodes in uniqueGuestsNumInChildrenOfNonLeafNodesForTrees]
    return uniqueRepeatChildTrees, uniqueRepeatChildTreesPriors

def calGuestsNumInChildrenOfNonLeafNodes(tree):
    nonLeafNodes = [n for n,d in dict(tree.out_degree()).items() if d!=0]
    childrenOfNonLeafNodes = [list(tree.successors(nonLeafNode)) for nonLeafNode in nonLeafNodes]
    guestsNumInChildrenOfAllNonLeafNodes = [[len(tree.node[childNode]['guests']) for childNode in childrenOfNonLeafNode] for childrenOfNonLeafNode in childrenOfNonLeafNodes]
    return guestsNumInChildrenOfAllNonLeafNodes

def calNonLeafNodePrior(guestsNumInChildrenOfNonLeafNode, gamma):
    guestsNumTotal = sum(guestsNumInChildrenOfNonLeafNode)
    basicProbabiltyOfGuestsAssignmentIgnoreOrder = ft.reduce(op.mul, [gammaFunction(guestsNumOneTable + 1 - gamma) for guestsNumOneTable in guestsNumInChildrenOfNonLeafNode])/math.factorial(guestsNumTotal)
    possibleGuestsAssignments = list(set(it.permutations(guestsNumInChildrenOfNonLeafNode)))
    guestsPossibleCombinationsNumForAllEqualFormsAssignment = ft.reduce(op.add, [calGuestsPossibleCombinationsNumForGuestsAssignment(guestsAssignment) for guestsAssignment in possibleGuestsAssignments])
    totalProbalilityForOneAssignmentIgnorParticularGuests = basicProbabiltyOfGuestsAssignmentIgnoreOrder * guestsPossibleCombinationsNumForAllEqualFormsAssignment 
    probabilityDistributedToEveryEqualFormAssignment = totalProbalilityForOneAssignmentIgnorParticularGuests/len(possibleGuestsAssignments)
    return probabilityDistributedToEveryEqualFormAssignment

def calGuestsPossibleCombinationsNumForGuestsAssignment(guestsAssignment):
    guestsNumTotal = sum(guestsAssignment)
    restGuestsNumAfterSitDownInEachTable = [guestsNumTotal] + [guestsNumTotal - sum(guestsAssignment[:tableIndex]) for tableIndex in range(1, len(guestsAssignment))]
    guestsPossibleCombinationsNumForGuestsAssignment = ft.reduce(op.mul, [misc.comb(restGuestsNum - 1, guestsNumThisTable - 1) for restGuestsNum, guestsNumThisTable in zip(restGuestsNumAfterSitDownInEachTable, guestsAssignment)])
    return guestsPossibleCombinationsNumForGuestsAssignment

def removeRepeatChildNodeAndMapPriorsToNewTrees(uniqueRepeatChildTrees, uniqueRepeatChildTreesPriors):
    noRepeatChildTrees = []
    noRepeatChildTreesPriors = []
    noRepeatChildTreesNodesSituations = []
    for uniqueRepeatChildTree in uniqueRepeatChildTrees:
        noRepeatChildTree = removeRepeatChildNode(uniqueRepeatChildTree)
        noRepeatChildTreeGuestsAssignment = calChildrenNumOfNodes(noRepeatChildTree)
        repeatChildTreeIndex = uniqueRepeatChildTrees.index(uniqueRepeatChildTree) 
        if noRepeatChildTreeGuestsAssignment in noRepeatChildTreesNodesSituations:
            noRepeatChildTreeIndex = noRepeatChildTreesNodesSituations.index(noRepeatChildTreeGuestsAssignment)
            noRepeatChildTreesPriors[noRepeatChildTreeIndex] = noRepeatChildTreesPriors[noRepeatChildTreeIndex] + uniqueRepeatChildTreesPriors[repeatChildTreeIndex]
        else:
            noRepeatChildTrees.extend([noRepeatChildTree])
            noRepeatChildTreesNodesSituations.extend([noRepeatChildTreeGuestsAssignment])
            noRepeatChildTreesPriors.extend([uniqueRepeatChildTreesPriors[repeatChildTreeIndex]])
        print(uniqueRepeatChildTree, noRepeatChildTreesPriors, noRepeatChildTreesNodesSituations)
    return noRepeatChildTrees, noRepeatChildTreesPriors 

def calChildrenNumOfNodes(tree):
    nodes = list(tree.nodes)
    childrenNumOfNodes = [len(list(tree.successors(node))) for node in nodes]
    return childrenNumOfNodes

def removeRepeatChildNode(tree):
    currNode = 0
    while currNode <= max(list(tree.nodes())):
        if currNode in list(tree.nodes):
            childrenOfCurrNode = list(tree.successors(currNode))
            if len(childrenOfCurrNode) == 1:
                childrenOfChild = list(tree.successors(childrenOfCurrNode[0]))
                tree.remove_nodes_from(childrenOfCurrNode)
                tree.add_edges_from(it.product([currNode], childrenOfChild))
            else:
                currNode = currNode + 1
        else:
            currNode = currNode + 1
    return tree

def main():
    treeNum = 100
    gamma = 0.9
    maxDepth = 3
    trees = generateNCRPTreesAndRemoveRepeatChildNodeAndMapPriorsToNewTrees(gamma, maxDepth, treeNum)
#        nx.draw(tree)
#        nx.draw_networkx_labels(tree, pos=nx.spring_layout(tree))
#        pyplot.show()

#def codeRepeatChildTree(guestsNumInTablesOfChildrenNodes, codeBase):
#    nonLeafNodesCodes = [codeNonLeafNode(guestsNumInTablesOfChildrenNode, codeBase) for guestsNumInTablesOfChildrenNode in guestsNumInTablesOfChildrenNodes]
#    repeatChildTreeCode = codeTree(nonLeafNodesCodes, codeBase)
#    return repeatChildTreeCode
#
#def codeTree(nonLeafNodesCodes, codeBase):
#    treeCode = ft.reduce(lambda x, y: x*(codeBase**(math.floor(math.log(y)/math.log(codeBase)) + 1)) + y, nonLeafNodesCodes) 
#    return treeCode
#
#def codeNonLeafNode(guestsNumInTablesOfChildrenNode, codeBase):
#    nonLeafNodeCode = ft.reduce(lambda x, y: x*codeBase + y, guestsNumInTablesOfChildrenNode)
#    return nonLeafNodeCode
if __name__ == "__main__":
    main()
