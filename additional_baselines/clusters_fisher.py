import numpy as np
from kmedoids import kMedoids
from sklearn.metrics.pairwise import pairwise_distances
import pickle
from fi import fisher_information
import torch
import torch.nn.functional as F
import torch.autograd as autograd

a = pickle.load(open('new_feats_saved_15k.bn','rb'))
X = a['f']

_probs = a['o']
probs = F.softmax(torch.from_numpy(_probs)).data.numpy()

#already_sel = range(5000)
already_sel = np.load(open('fisher_15000.bn', 'rb')) 
remaining = np.setdiff1d(np.array(range(50000)), already_sel)
chosen = fisher_information(X,already_sel,remaining,0.2,probs,5000)
#D = pairwise_distances(a[remaining,:], metric='euclidean')
#M, C = kMedoids(D, 5000)
#nd=np.array(list(already_sel) + list(M))
np.save(open('selected_fisher_20000.bn','wb'),np.array(chosen))
