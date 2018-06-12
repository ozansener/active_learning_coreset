from triplet_clustering import TripletClustering
from clustering_util import run_loss_aug_clustering_on, evaluate_clustering, run_clustering_on
import mnist_data
import tensorflow as tf
import numpy as np
from datetime import datetime
from random import randrange

train_data = mnist_data.read_data_sets('./data/', one_hot=False)
test_im = train_data.test.images
test_l = train_data.test.labels

# HOS: shuffle but move every element. Creates copy.
def sattoloShuffle(x):
    items = np.copy(x)
    i = len(items)
    while i > 1:
        i = i - 1
        j = randrange(i)  # 0 <= j <= i-1
        items[j], items[i] = items[i], items[j]
    return items

def get_batch():
    im, l = train_data.train.next_batch(256)
    return im.astype(np.float32), l

def get_triplets(im, l):
    pos_idx = np.zeros((256,), dtype=int)
    neg_idx = np.zeros((256,), dtype=int)

    for class_id in np.unique(l):
        anchor_ids = np.where(l == class_id)[0]
        # shuffle and make sure no elements stay at the same loc
        pos_ids = sattoloShuffle(anchor_ids)
        neg_id = np.where(l != class_id)[0]
        neg_id_sampled = np.random.choice(neg_id, anchor_ids.size, replace=False)
        pos_idx[anchor_ids] = pos_ids
        neg_idx[anchor_ids] = neg_id_sampled
    anchor = im
    pos = im[pos_idx]
    neg = im[neg_idx]

    return anchor, pos, neg

def get_nmi(embeddings, labels):
    clustering_labels = run_clustering_on(embeddings, n_cluster=10)
    NMI = evaluate_clustering(labels, clustering_labels)
    return NMI

with tf.Session() as sesh:
    triplet_clustering = TripletClustering(1.0, 0.0001)
    print 'Initial Variable List:'
    print [tt.name for tt in tf.trainable_variables()]
    init = tf.initialize_all_variables()
    sesh.run(init)
    saver = tf.train.Saver()
    for batch_id in range(20000):
        im, l = get_batch()
        # compute anchor, pos and neg
        anch, pos, neg = get_triplets(im, l)
        loss = triplet_clustering.train_step(anch, pos, neg, sesh)
       
        if batch_id % 100 == 0:
            saver.save(sesh,'models/mnist_triplet',global_step=batch_id)

        if batch_id % 10 == 0:
            print "{}: step {}, loss {}".format(datetime.now(), batch_id, loss)

    embeddings = triplet_clustering.get_embeddings(test_im, sesh)
    NMI = get_nmi(embeddings, test_l)
    print 'Final NMI: ', NMI
