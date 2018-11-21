import scipy.cluster.hierarchy as hierarchyCluster
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def clusterTexons(texonsFeatures):
    clusteringTree = hierarchyCluster.linkage(texonsFeatures, 'ward')
    #fig = plt.figure(figsize = (25, 10))
    #dn = hierarchyCluster.dendrogram(clusteringResults)
    #plt.show()
    clusteringResults = hierarchyCluster.fcluster(clusteringTree, 4, 'maxclust')
    return clusteringResults

def main():
    imageNum = 1
    for imageIndex in range(imageNum):
        texonsObserved = pd.read_csv('~/segmentation-expt4/generate/demoUnscaled' + str(imageIndex) + '.csv')
        texonsFeatures = np.array(texonsObserved[['color', 'length', 'angleRotated', 'logWidthLengthRatio']].values)
        clusteringResults = clusterTexons(texonsFeatures)
        texonsObserved['hierarchyClusterAssignment'] = clusteringResults
        texonsObserved.to_csv('~/segmentation-expt4/generate/demoClustering' + str(imageIndex) + '.csv')

if __name__ == "__main__":
    main()
