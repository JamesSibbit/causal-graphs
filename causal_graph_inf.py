import pandas as pd
import numpy as np
from causality.inference.search import IC
from causality.inference.independence_tests import RobustRegressionTest

"""
We want to create RVs that have causality between them, and see if we can use
IC to recognise this causality. Create RVs below that are dependent.

Graph will be as follows:

       --> x2 -->    --> x5
x0-->x1           x4        --> x7
       --> x3 -->    --> x6

"""

SIZE = 5000
x0 = np.random.normal(size=SIZE)
x1 = x0 + np.random.normal(size=SIZE)
x2 = x1 + np.random.normal(size=SIZE)
x3 = x1 + np.random.normal(size=SIZE)
x4 = x2 + x3 + np.random.normal(size=SIZE)
x5 = x4 + np.random.normal(size=SIZE)
x6 = x4 + np.random.normal(size=SIZE)
x7 = x6 + x5 + np.random.normal(size=SIZE)

#Load this data into pandas df, and make sure variable types are cts
graph_one = pd.DataFrame({'x0': x0, 'x1' : x1, 'x2' : x2, 'x3' : x3, 'x4' : x4, 'x5' : x5, 'x6' : x6, 'x7' : x7})
var_types = {'x0' : 'c','x1' : 'c', 'x2' : 'c', 'x3' : 'c', 'x4' : 'c', 'x5' : 'c', 'x6' : 'c', 'x7' : 'c'}

#Now run our IC algo graph search

def graph_search(graph, var_types):
    ic_algo = IC(RobustRegressionTest)
    return ic_algo.search(graph, var_types)

graph = graph_search(graph_one, var_types)

#What edges of our graph did this pick up?
print(graph.edges(data=True))
