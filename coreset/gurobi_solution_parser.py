import pickle
import matplotlib.pyplot as plt
import numpy as np

import numpy
import pickle
import numpy.matlib
import time
import pickle
import bisect

gurobi_solution_file = 'solution_2.86083525164.sol'
results = open(gurobi_solution_file).read().split('\n')
results_nodes = filter(lambda x: 'y' in x,filter(lambda x:'#' not in x, results))
string_to_id = lambda x:(int(x.split(' ')[0].split('_')[1]),int(x.split(' ')[1]))
result_node_ids = map(string_to_id, results_nodes)

results_as_dict = {v[0]:v[1] for v in result_node_ids}

centers = []
for node_result in result_node_ids:
    if node_result[1] > 0:
        centers.append(node_result[0])

pickle.dump(centers,open('centers.bn','wb'))
