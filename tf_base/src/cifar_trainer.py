import tensorflow as tf
from network.vgg import VGG16
from network.vgg_with_dropout import VGG16Dropout

class Cifar10Trainer(object):
    """
    Train a network using MNIST data or a dataset following 
    """
    def __init__(self, learning_rate, device_name, isdropout):
        self.ph_images = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.ph_labels = tf.placeholder(tf.float32, [None, 10])

        self.keep_prob = tf.placeholder(tf.float32)
        
        with tf.device(device_name):
            if not isdropout:
                net = VGG16({'data':self.ph_images})
            else:
                net = VGG16Dropout({'data':self.ph_images}, keep_prob=self.keep_prob)
        
            self.pred = net.get_output()
            self.sm_pred = tf.nn.softmax(self.pred)

            vars   = tf.trainable_variables() 
            self.l2_loss  = tf.add_n([ tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name ]) * 0.002

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pred, self.ph_labels), 0) + self.l2_loss
            self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        
            correct_prediction = tf.equal(tf.argmax(self.sm_pred, 1), tf.argmax(self.ph_labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        tf.summary.scalar("summary_training_accuracy", self.accuracy)
        tf.summary.scalar("summary_loss", self.loss)
        #tf.summary.image("summary_images", self.ph_images, max_outputs=5)
        #tf.summary.histogram("summary_hist_im", self.ph_images)
        self.summaries = tf.summary.merge_all()
        self.test_summary = tf.summary.scalar("summary_test_accuracy", self.accuracy) 
 
    def compute_accuracy(self, images, labels, session):
        acc = session.run(self.accuracy, feed_dict={self.ph_images: images, self.ph_labels:labels, self.keep_prob:1.0})
        return acc

    def train_step(self, images, labels, session, kp):
        train_op = session.run([self.train_op], feed_dict={self.ph_images: images, self.ph_labels: labels, self.keep_prob:kp})
        return train_op

    def summary_step(self, images, labels, session):
        summ, loss = session.run([self.summaries, self.loss], feed_dict={self.ph_images: images, self.ph_labels: labels, self.keep_prob:1.0})
        return loss, summ

    def test_step(self, images, labels, session):
        acc, summ = session.run([self.accuracy, self.test_summary], feed_dict={self.ph_images: images, self.ph_labels: labels, self.keep_prob:1.0})
        return acc, summ
