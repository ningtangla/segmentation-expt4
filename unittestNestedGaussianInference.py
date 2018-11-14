import unittest
from ddt import ddt, data, unpack
import pandas as pd
import numpy as np

@ddt
class TestAnalysisDataByPandas(unittest.TestCase):
    def setUp(self):
        pass

    @data((pd.DataFrame([np.random.normal(100, 1000, 100), np.random.uniform(-100, 200, 100)],columns = ['a','b']))) 

    @unpack 
    def testCalMeanByPandas(self, originalDf):
        outputDataFrame = target.calMeanByPandas(originalDf)
        self.assertTrue(np.allclose(outputDataFrame, originalDf.mean()))

    @data(([np.random.randint(0, 10, 4), np.random.randint(0, 10, 6), np.random.randint(5, 20, 4)], ['x','y','z']))

    @unpack
    def testInitPrior(self, multiIndexValuesOnDiffLevel, names):
        output =  targetCode.InitPrior(multiIndexValuesOnDiffLevel, names)
        truthMultiIndex = pd.MultiIndex.from_product(multiIndexValuesOnDiffLevel, names = names)
        self.assertTrue(type(outputDataFrame.index) == type(truthMultiIndex))
        self.assertTrue(np.all(outputDataFrame.index == truthMultiIndex))
        self.assertTrue(np.all(outputDataFrame.values == 1/ft.reduce(op.mul, [len(levelValues) for levelValues in multiIndexValuesOnDiffLevel])))
    
    @data((pd.DataFrame([np.random.normal(100, 100, 200), np.random.normal(-100, 1000, 200)], index = pd.MultiIndex.from_product([range(4), range(5), range(10)], names = ['x', 'y', 'z']), columns= ['a', 'b'])))
    @unpack 
    def testCalGroupbyMeans(self, originalDf):
        output = targetCode.calGroupbyMeans(originalDf)
        self.assertTrue(np.allclose(outputDataFrame, originalDfLogP.groupby(['x','z']).mean()))

    @data((pd.DataFrame([np.random.normal(100, 100, 200), np.random.normal(-100, 1000, 200)], index = pd.MultiIndex.from_product([range(4), range(5), range(10)], names = 'x', 'y', 'z'), columns= ['a', 'b']), pd.DataFrame([np.random.normal(100, 100, 10)], columns = ['c'])))
    @unpack
    def testTransDataByMultiIndex(self, hypothesesDf, dataDf):
        output = targetCode.testTransDataByMultiIndex(hypothesis, dataDf)
        hypotheses = hypothesesDf.index
        xLevelIndexValues = hypotheses.get_level_values('x')
        yLevelIndexValues = hypotheses.get_level_values('y')
        cValueDiffForXAndY = dataDf[xLevelIndexValues] - dataDf[yLevelIndexValues]
        self.assertTrue(np.allclose(output, cValueDiffForXAndY))

    def tearDown(self):
        pass

if __name__ == '__main__':
    pandasDataAnalysisSuit = unittest.TestLoader().loadTestsFromTestCase(TestAnalysisDataByPandas)
    unittest.TextTestRunner(verbosity = 2).run(pandasDataAnalysisSuit) 
