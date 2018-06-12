import pickle
import matplotlib.pyplot as plt
import numpy as np

import numpy
import pickle
import numpy.matlib
import time
import pickle
import bisect

dat = pickle.load(open('chosen_data_5000_False'))
data = numpy.concatenate((dat['gt_f'],dat['f']), axis=0)

start = time.clock()
num_images = data.shape[0]
print "s"
dist_mat = numpy.matmul(data,data.transpose())

print "m"
sq = numpy.array(dist_mat.diagonal()).reshape(num_images,1)
dist_mat *= -2
dist_mat+=sq
dist_mat+=sq.transpose()

res = open('ssolution_2.86083525164.sol').read().split('\n')
y_res = filter(lambda x: 'y' in x,filter(lambda x:'#' not in x, res))
f = lambda x:(int(x.split(' ')[0].split('_')[1]),int(x.split(' ')[1]))
_c = map(f,y_res)
dic = {c[0]:c[1] for c in _c}
cent = []
for c in _c:
    if c[1] > 0:
        cent.append(c[0])

lab = numpy.argmin(dist_mat[cent,:], axis=0)
pickle.dump(lab,open('labels.bn','wb'))

lab_v = numpy.min(dist_mat[cent,:], axis=0)
pickle.dump(lab_v,open('labels_values.bn','wb'))

