import pickle
import numpy
import numpy.matlib
import time
import pickle
import bisect

dat = pickle.load(open('representations_saved'))
#data = numpy.concatenate((dat['a'][0],dat['a'][3]), axis=0)
budget = 20000 # k in the paper

# Compute Pairwise Distances between all points in 3 BLAS call
start = time.clock() 
num_images = data.shape[0]
dist_mat = numpy.matmul(data,data.transpose())

sq = numpy.array(dist_mat.diagonal()).reshape(num_images,1)
dist_mat *= -2
dist_mat+=sq
dist_mat+=sq.transpose()
elapsed = time.clock() - start
print "Time spent in (computing all pairwise l2 distances) is: ", elapsed

start = time.clock()
num_data = dist_mat.shape[0]
candidate_ids = range(num_data)
subset = []

# first insert all labeled ones
# This is the initial set S in the paper
for lab in dat['gt_f'].shape[0]):
    i_star = lab
    subset.append(i_star)

# Greedy facility location
k = 0
while k < budget:
    no = numpy.argmax(d)
    subset.append(no)
    add_d = dist_mat[:,no].reshape(dist_mat.shape[0],1)
    new_dj = numpy.concatenate((d.reshape(dist_mat.shape[0],1),add_d),axis=1)
    new_d = numpy.min(new_dj,axis=1)
    d = new_d
    k = k + 1

labels = numpy.argmin(dist_mat[subset, :], axis=0)
max_dist = numpy.max(numpy.min(dist_mat[subset, :], axis=0))
centers = subset
elapsed = time.clock() - start

pickle.dump({'centers':subset, 'ids':labels},open('results.bn','wb'))
print "Time spent in ", elapsed
print numpy.sum(dist_mat<max_dist)
print max_dist
