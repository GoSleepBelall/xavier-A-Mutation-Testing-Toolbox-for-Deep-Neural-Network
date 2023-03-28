import pprint
from mutation_operators import BiasLevel
import predictions_analysis as pa

BiasOperator = BiasLevel()
counters = {0: {'fn': 250, 'fp': 0, 'tn': 500, 'tp': 500},
            1: {'fn': 0, 'fp': 250, 'tn': 500, 'tp': 500},
            2: {'fn': 125, 'fp': 125, 'tn': 500, 'tp': 500},
            3: {'fn': 500, 'fp': 500, 'tn': 500, 'tp': 500},
            4: {'fn': 450, 'fp': 450, 'tn': 50, 'tp': 50},
            5: {'fn': 250, 'fp': 0, 'tn': 500, 'tp': 500},
            6: {'fn': 250, 'fp': 0, 'tn': 500, 'tp': 500},
            7: {'fn': 0, 'fp': 0, 'tn': 100, 'tp': 100},
            8: {'fn': 0, 'fp': 250, 'tn': 500, 'tp': 500},
            9: {'fn': 0, 'fp': 0, 'tn': 500, 'tp': 500}}

counters2 = {0 : {'fn': 0, 'fp': 0,'tn':0, 'tp': 0 },
            1 : {'fn': 0, 'fp': 0,'tn':0, 'tp': 0 },
            2 : {'fn': 0, 'fp': 0,'tn':0, 'tp': 0 },
            3 : {'fn': 0, 'fp': 0,'tn':0, 'tp': 0 },
            4 : {'fn': 0, 'fp': 0,'tn':0, 'tp': 0 },
            5 : {'fn': 0, 'fp': 0,'tn':0, 'tp': 0 },
            6 : {'fn': 0, 'fp': 0,'tn':0, 'tp': 0 },
            7 : {'fn': 0, 'fp': 0,'tn':0, 'tp': 0 },
            8 : {'fn': 0, 'fp': 0,'tn':0, 'tp': 0 },
            9 : {'fn': 0, 'fp': 0,'tn':0, 'tp': 0 }}


accuracy  = pa.getAllMetrics(counters2, 1)

pprint.pprint(accuracy)
