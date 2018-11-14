import unittest
from ddt import ddt, data, unpack
import pandas as pd
import numpy as np
import itertools as it
import math
import calPosterior as targetCode


@ddt
class TestCalPosterior(unittest.TestCase):
    def setUp(self): 
        #self.dataIndex = pd.MultiIndex.from_product([[0, 1, 2],['x', 'y']], names=['Identity', 'coordinate'])
        self.hypothesesIndex = pd.MultiIndex.from_product([[0, 1, 2], [0, 1, 2], [50, 11, 3.3]], names = ['wolfIdentity', 'sheepIdentity', 'chasingPrecision'])
        self.observedData = pd.DataFrame({'wolfDeviation': [3.14, 3.14, 3.14, 0.1, 0.1, 0.1, 0.78, 0.78, 0.78, 3.04, 3.04, 3.04, 3.14, 3.14, 3.14, 2.36, 2.36, 2.36, 2.36, 2.36, 2.36, 0.78, 0.78, 0.78, 3.14, 3.14, 3.14], 'sheepDeviation': [3.14, 3.14, 3.14, 0.1, 0.1, 0.1, 0.78, 0.78, 0.78, 3.04, 3.04, 3.04, 3.14, 3.14, 3.14, 2.36, 2.36, 2.36, 2.36, 2.36, 2.36, 0.78, 0.78, 0.78, 3.14, 3.14, 3.14]}, index = self.hypothesesIndex)
        self.hypothesesNum = len(self.hypothesesIndex)
        self.beforePosterior = pd.DataFrame(np.log([1] * self.hypothesesNum), index = self.hypothesesIndex, columns = ['logP'])
        self.beforePosterior['attentionStatus'] = [0] * self.hypothesesNum
    
    @unittest.skip    
    @data((pd.Series(np.log([0.2, 0.3])), pd.Series(np.log([0.4, 0.6]))),(np.log([0.1, 0.3]), np.log([0.25, 0.75])))   
    @unpack 
    def testNormalizeLogP(self, originalDfLogP, normalizedLogP):
        self.assertTrue(np.allclose(targetCode.normalizeLogP(originalDfLogP), normalizedLogP))
    
    @data((10000, 0.0001))
    @unpack
    def testPrecisionEffectOnIdealDecay(self, highPrecision, lowPrecision):
        highPrecisionHypothesesInformation = self.beforePosterior.copy()
        highPrecisionHypothesesInformation['memoryDecay'] = [1] * self.hypothesesNum
        highPrecisionHypothesesInformation['perceptionPrecision'] = [highPrecision] * self.hypothesesNum
        highPrecisionPosterior = targetCode.calPosteriorLog(highPrecisionHypothesesInformation, self.observedData)
        highPrecisionChasingLikelihoodAlmostGroundTruth = targetCode.calAngleLikelihoodLogModifiedForPiRangeAndMemoryDecay(self.observedData['wolfDeviation'], highPrecisionHypothesesInformation.index.get_level_values('chasingPrecision')) 
        highPrecisionEscapingLikelihoodAlmostGroundTruth = targetCode.calAngleLikelihoodLogModifiedForPiRangeAndMemoryDecay(self.observedData['sheepDeviation'], 1.94) 

        self.assertTrue(np.all(np.isclose(highPrecisionPosterior['chasingLikelihoodLog'].values, highPrecisionChasingLikelihoodAlmostGroundTruth, rtol = 0.01)))
        self.assertTrue(np.all(np.isclose(highPrecisionPosterior['escapingLikelihoodLog'].values, highPrecisionEscapingLikelihoodAlmostGroundTruth,rtol = 0.01)))

        lowPrecisionHypothesesInformation = self.beforePosterior.copy()
        lowPrecisionHypothesesInformation['memoryDecay'] = [1] * self.hypothesesNum
        lowPrecisionHypothesesInformation['perceptionPrecision'] = [lowPrecision] * self.hypothesesNum
        lowPrecisionPosterior = targetCode.calPosteriorLog(lowPrecisionHypothesesInformation, self.observedData)
        lowPrecisionChasingLikelihoodAlmostGroundTruth = targetCode.calAngleLikelihoodLogModifiedForPiRangeAndMemoryDecay(self.observedData['wolfDeviation'], lowPrecision)
        lowPrecisionEscapingLikelihoodAlmostGroundTruth = targetCode.calAngleLikelihoodLogModifiedForPiRangeAndMemoryDecay(self.observedData['sheepDeviation'], lowPrecision)
        self.assertTrue(np.all(np.isclose(lowPrecisionPosterior['chasingLikelihoodLog'].values, lowPrecisionChasingLikelihoodAlmostGroundTruth, rtol = 0.01)))
        self.assertTrue(np.all(np.isclose(lowPrecisionPosterior['escapingLikelihoodLog'].values, lowPrecisionEscapingLikelihoodAlmostGroundTruth, rtol = 0.01)))
         

    @data((0.7, 0), (0.8, 0.78), (1, 0.1), (1, 0)) 
    @unpack
    def testDecayEffectOnIdealAttention(self, highDecay, lowDecay):
        highDecayFrame1 = self.beforePosterior.copy()
        highDecayFrame1['perceptionPrecision'] = [1000] * self.hypothesesNum
        highDecayFrame1['memoryDecay'] = [highDecay] * self.hypothesesNum
        highDecayFrame2 = targetCode.calPosteriorLog(highDecayFrame1, self.observedData)
        highDecayFrame3 = targetCode.calPosteriorLog(highDecayFrame2, self.observedData)

        lowDecayFrame1 = self.beforePosterior.copy()
        lowDecayFrame1['perceptionPrecision'] = [1000] * self.hypothesesNum
        lowDecayFrame1['memoryDecay'] = [lowDecay] * self.hypothesesNum
        lowDecayFrame2 = targetCode.calPosteriorLog(lowDecayFrame1, self.observedData)
        lowDecayFrame3 = targetCode.calPosteriorLog(lowDecayFrame2, self.observedData)
        
        highPHighDecay = np.max(highDecayFrame3['logP'])
        highPLowDecay = np.max(lowDecayFrame3['logP'])
        lowPHighDecay = np.min(highDecayFrame3['logP'])
        lowPLowDecay = np.min(lowDecayFrame3['logP'])
        self.assertLess(highPLowDecay, highPHighDecay)
        self.assertLess(lowPHighDecay, lowPLowDecay)
        

    @unittest.skip
    @data((pd.DataFrame([[0, 0]], index = pd.Index(['x','y'])), 0),
          (pd.DataFrame([[0, 1]], index = pd.Index(['x','y'])), 1),
          (pd.DataFrame([[2, 0]], index = pd.Index(['x','y'])), 2))
    @unpack
    def testCalVectorNorm(self, vector, normExpected):
        self.assertEqual(targetCode.calVectorNorm(vector).values[0], normExpected)
    
    @unittest.skip
    @data((pd.DataFrame([[0, 1]], index = pd.Index(['x','y'])), pd.DataFrame([[0, 1]], index = pd.Index(['x','y'])), 0),
          (pd.DataFrame([[0, 1]], index = pd.Index(['x','y'])), pd.DataFrame([[1, 0]], index = pd.Index(['x','y'])), math.pi/2), 
          (pd.DataFrame([[0, 1]], index = pd.Index(['x','y'])), pd.DataFrame([[1, 1]], index = pd.Index(['x','y'])), math.pi/4), 
          (pd.DataFrame([[0, 1]], index = pd.Index(['x','y'])), pd.DataFrame([[-1, -1]], index = pd.Index(['x','y'])), math.pi*3/4)) 
    @unpack
    def testCalAngleBetweenVectors(self, vector1, vector2, angleExpected):
        self.assertAlmostEqual(targetCode.calAngleBetweenVectors(vector1, vector2).values[0], angleExpected, 3)
    
    @unittest.skip
    def testAttendCalLikelihood(self):
        likelihoodLogDf = targetCode.calAttendedLikelihoodLog(self.hypothesisIndex, self.beforeData, self.nowData)
        self.assertEqual(likelihoodLogDf['logP'].idxmax(), (0, 1, 50))
    

    @unittest.skip
    def testCalPosterior(self):
        calPosteriorLog = targetCode.CalPosteriorLog(0.99)
        posteriorLogDf = calPosteriorLog(self.beforePosterior, self.beforeData, self.nowData)
        self.assertEqual(posteriorLogDf['logP'].idxmax(), (0, 1, 50))
   
    @unittest.skip
    @data(([0, 1], [0, 1]),([0, 2], [0, 0]))
    @unpack
    def testAttentionEffectOnP(self, attentionPair, highestPosteriorPair):
        calPosterirLog = targetCode.CalPosterirLog(0.99)
        self.beforePosterior.loc[(self.beforePosterior.index.get_level_values('WolfIdentity') == attentionPair[0]) & (self.beforePosterior.index.get_level_values('SheepIdentity') == attentionPair[1]), 'attentionStatus'] = 1
        posteriorLogDf = calPosterirLog(self.beforePosterior, self.beforeData, self.nowData)
        groupData = posteriorLogDf.groupby(['WolfIdentity', 'SheepIdentity']).sum()
        self.assertEqual(groupData['logP'].idxmax(), tuple(highestPosteriorPair))
        
    def tearDown(self):
        pass

if __name__ == '__main__':
   
    chasingDetectionSuit = unittest.TestLoader().loadTestsFromTestCase(TestCalPosterior)
    unittest.TextTestRunner(verbosity = 2).run(chasingDetectionSuit) 
    

