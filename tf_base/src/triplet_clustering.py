import tensorflow as tf
from network.lenetem import LeNetEm

class TripletClustering(object):
    def __init__(self, ALPHA, LR):
        self.ALPHA = ALPHA
        self.LR = LR
        self.ph_anchor = tf.placeholder(tf.float32, [None, 28*28])
        self.ph_pos = tf.placeholder(tf.float32, [None, 28*28])
        self.ph_neg = tf.placeholder(tf.float32, [None, 28*28])

        with tf.variable_scope("towers") as scope:
            anchor_tower = LeNetEm({'data':self.ph_anchor})
            scope.reuse_variables()
            pos_tower = LeNetEm({'data':self.ph_pos})
            neg_tower = LeNetEm({'data':self.ph_neg})

        self.anchor = anchor_tower.layers['ip2']
        self.positive = pos_tower.layers['ip2']
        self.negative = neg_tower.layers['ip2']

        self.loss = self.loss_function()
        self.train_op = tf.train.RMSPropOptimizer(self.LR).minimize(self.loss)

    def loss_function(self):
        pos_diff = self.anchor - self.positive
        neg_diff = self.anchor - self.negative

        pos_dist = tf.reduce_sum(tf.mul(pos_diff, pos_diff), 1)
        neg_dist = tf.reduce_sum(tf.mul(neg_diff, neg_diff), 1)

        triplet = tf.add(self.ALPHA, tf.add(pos_dist, tf.neg(neg_dist)))
        return tf.reduce_sum(tf.nn.relu(triplet))

    def train_step(self, anchors, positives, negatives, session):
        loss, _ = session.run([self.loss, self.train_op], feed_dict={self.ph_anchor: anchors, self.ph_pos: positives, self.ph_neg:negatives})
        return loss   

    def compute_embedding_for(self, images, session):
        # all towers are equal so it does not matter which tower we use
        embeddings = session.run(self.anchor, feed_dict={self.ph_anchor: images})
        return embeddings    
