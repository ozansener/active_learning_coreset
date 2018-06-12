import numpy as np
from kmedoids import kMedoids
from sklearn.metrics.pairwise import pairwise_distances

a=np.load(open('feats_saved_10k.bn','rb'))

already_sel = np.load(open('selected10000.bn', 'rb')) 
remaining = np.setdiff1d(np.array(range(50000)), already_sel)
D = pairwise_distances(a[remaining,:], metric='euclidean')
M, C = kMedoids(D, 5000)
nd=np.array(list(already_sel) + list(M))
np.save(open('selected15000.bn','wb'),nd)
