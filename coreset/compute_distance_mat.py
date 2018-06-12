import pickle
import numpy
import numpy.matlib
import time
import pickle
import bisect

dat = pickle.load(open('chosen_data_5000_False'))
data = numpy.concatenate((dat['gt_f'],dat['f']), axis=0)
budget = 5000

start = time.clock() 
num_images = data.shape[0]

dist_mat = numpy.matmul(data,data.transpose())

sq = numpy.array(dist_mat.diagonal()).reshape(num_images,1)
dist_mat *= -2
dist_mat+=sq
dist_mat+=sq.transpose()

elapsed = time.clock() - start
print "Time spent in (function name) is: ", elapsed
numpy.save('distances.npy', dist_mat)

