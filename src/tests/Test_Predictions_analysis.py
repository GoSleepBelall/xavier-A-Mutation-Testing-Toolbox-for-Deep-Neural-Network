import unittest
import src.main.predictions_analysis as pa
"""
def test_get(self):
        
        expected_output = 
        
        self.assertEqual(pa(self.counters), expected_output)
"""
class TestPredictionsAnalysis(unittest.TestCase):

    def setUp(self):
        self.zero_edge = {0 : {'fn': 0, 'fp': 0,'tn':0, 'tp': 0 },
                        1 : {'fn': 0, 'fp': 0,'tn':0, 'tp': 0 },
                        2 : {'fn': 0, 'fp': 0,'tn':0, 'tp': 0 },
                        3 : {'fn': 0, 'fp': 0,'tn':0, 'tp': 0 },
                        4 : {'fn': 0, 'fp': 0,'tn':0, 'tp': 0 },
                        5 : {'fn': 0, 'fp': 0,'tn':0, 'tp': 0 },
                        6 : {'fn': 0, 'fp': 0,'tn':0, 'tp': 0 },
                        7 : {'fn': 0, 'fp': 0,'tn':0, 'tp': 0 },
                        8 : {'fn': 0, 'fp': 0,'tn':0, 'tp': 0 },
                        9 : {'fn': 0, 'fp': 0,'tn':0, 'tp': 0 }}

        self.zero_edge_result = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

        self.counters = {0: {'fn': 250, 'fp': 0, 'tn': 500, 'tp': 500},
                    1: {'fn': 0, 'fp': 250, 'tn': 500, 'tp': 500},
                    2: {'fn': 125, 'fp': 125, 'tn': 500, 'tp': 500},
                    3: {'fn': 500, 'fp': 500, 'tn': 500, 'tp': 500},
                    4: {'fn': 450, 'fp': 450, 'tn': 50, 'tp': 50},
                    5: {'fn': 250, 'fp': 0, 'tn': 500, 'tp': 500},
                    6: {'fn': 250, 'fp': 0, 'tn': 500, 'tp': 500},
                    7: {'fn': 0, 'fp': 0, 'tn': 100, 'tp': 100},
                    8: {'fn': 0, 'fp': 250, 'tn': 500, 'tp': 500},
                    9: {'fn': 0, 'fp': 0, 'tn': 500, 'tp': 500}}

    def test_getSpecificity(self):
        expected_output ={0: '1.0000',
                 1: '0.6667',
                 2: '0.8000',
                 3: '0.5000',
                 4: '0.1000',
                 5: '1.0000',
                 6: '1.0000',
                 7: '1.0000',
                 8: '0.6667',
                 9: '1.0000'}

        self.assertEqual(pa.getSpecificity(self.counters), expected_output)
        self.assertEqual(pa.getSpecificity(self.zero_edge), self.zero_edge_result)

    def test_getF1Score(self):
        expected_output = {0: '0.8000',
              1: '0.8000',
              2: '0.8000',
              3: '0.5000',
              4: '0.1000',
              5: '0.8000',
              6: '0.8000',
              7: '1.0000',
              8: '0.8000',
              9: '1.0000'}

        self.assertEqual(pa.getF1Score(self.counters), expected_output)
        self.assertEqual(pa.getF1Score(self.zero_edge), self.zero_edge_result)

    def test_getFBetaScore(self):
        expected_output = {0: '0.8000',
            1: '0.8000',
            2: '0.8000',
            3: '0.5000',
            4: '0.1000',
            5: '0.8000',
            6: '0.8000',
            7: '1.0000',
            8: '0.8000',
            9: '1.0000'}

        self.assertEqual(pa.getFBetaScore(self.counters,1), expected_output)
        self.assertEqual(pa.getFBetaScore(self.zero_edge,1), self.zero_edge_result)

    def test_getAUC(self):
        expected_output = {0: '0.6667',
         1: '0.6667',
         2: '0.6000',
         3: '0.0000',
         4: '-0.8000',
         5: '0.6667',
         6: '0.6667',
         7: '1.0000',
         8: '0.6667',
         9: '1.0000'}

        self.assertEqual(pa.getAuc(self.counters), expected_output)
        self.assertEqual(pa.getAuc(self.zero_edge), self.zero_edge_result)



    def test_getSensitivity(self):
        expected_output = {0: '0.6667',
                 1: '1.0000',
                 2: '0.8000',
                 3: '0.5000',
                 4: '0.1000',
                 5: '0.6667',
                 6: '0.6667',
                 7: '1.0000',
                 8: '1.0000',
                 9: '1.0000'}

        self.assertEqual(pa.getSensitivity(self.counters), expected_output)
        self.assertEqual(pa.getSensitivity(self.zero_edge), self.zero_edge_result)

    def test_getRecall(self):
        expected_output = {0: '0.6667',
            1: '1.0000',
            2: '0.8000',
            3: '0.5000',
            4: '0.1000',
            5: '0.6667',
            6: '0.6667',
            7: '1.0000',
            8: '1.0000',
            9: '1.0000'}

        self.assertEqual(pa.getRecall(self.counters), expected_output)
        self.assertEqual(pa.getRecall(self.zero_edge),self.zero_edge_result)

    def test_getPrecision(self):
        expected_output = {0: '1.0000',
               1: '0.6667',
               2: '0.8000',
               3: '0.5000',
               4: '0.1000',
               5: '1.0000',
               6: '1.0000',
               7: '1.0000',
               8: '0.6667',
               9: '1.0000'}

        self.assertEqual(pa.getPrecision(self.counters), expected_output)
        self.assertEqual(pa.getPrecision(self.zero_edge), self.zero_edge_result)


    def test_getAccuracy(self):
        expected_output = {0: '0.8000',
              1: '0.8000',
              2: '0.8000',
              3: '0.5000',
              4: '0.1000',
              5: '0.8000',
              6: '0.8000',
              7: '1.0000',
              8: '0.8000',
              9: '1.0000'}

        self.assertEqual(pa.getAccuracy(self.counters), expected_output)


if __name__ == '__main__':
    unittest.main()
