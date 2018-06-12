import tensorflow as tf
import numpy
from network.vgg_adverserial import VGG16Adverserial
from network.vgg_adverserial_100 import VGG16Adverserial100
from network.mnist_adverserial import MNISTAdverserial

class AdverserialTrainer(object):
    """
    Train a network using robust objective
    """
    def __init__(self, learning_rate_net, learning_rate_adv, device_name,class_size):
        """
        :return:
        """
        self.batch_size = 128
        real_weight_decay = 0.001
        if not class_size==9:
            self.ph_images = tf.placeholder(tf.float32, [None, 32, 32, 3])
            self.ph_labels = tf.placeholder(tf.float32, [None, class_size])
        else:
            self.ph_images = tf.placeholder(tf.float32, [None, 28*28]) 
            self.ph_labels = tf.placeholder(tf.float32, [None, 10])

        self.ph_domains = tf.placeholder(tf.float32, [None, 2])
        self.keep_prob = tf.placeholder(tf.float32)
        self.phase = tf.placeholder(tf.bool, name='phase')  # train or test for batch norm
        self.lr_adv = learning_rate_adv
        self.flip_factor = tf.placeholder(tf.float32)
        self.learning_rate = tf.placeholder(tf.float32)

        with tf.device(device_name):
            if class_size ==100:
                real_net = VGG16Adverserial100({'data': self.ph_images}, phase=self.phase, keep_prob=self.flip_factor)
            elif class_size==9:
                real_net = MNISTAdverserial({'data': self.ph_images}, phase=self.phase, keep_prob=self.flip_factor)
            else:
                real_net = VGG16Adverserial({'data': self.ph_images}, phase=self.phase, keep_prob=self.flip_factor)
 
            class_pred = real_net.layers['fc7']
            domain_pred = real_net.layers['adv_fc3']
            real_pred_sm = tf.nn.softmax(class_pred)
            self.pred_l = real_pred_sm
            self.features = tf.reshape(real_net.layers['feat'],[-1,512])

            real_net_vars = tf.trainable_variables()
            real_l2_loss = tf.add_n([tf.nn.l2_loss(v)
                                     for v in real_net_vars
                                     if 'bias' not in v.name and 'adv' not in v.name]) * real_weight_decay
            self.clipper = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in real_net_vars if 'adv' in v.name]

            per_image_loss = tf.nn.softmax_cross_entropy_with_logits(class_pred, self.ph_labels)
            self.real_loss = tf.reduce_mean(per_image_loss, 0) #+ real_l2_loss

            # this is simply adding batch norm moving average to train operations
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.real_train_op = tf.train.RMSPropOptimizer(0.0005).minimize(self.real_loss)
            print 'Start RMS'
            correct_prediction = tf.equal(tf.argmax(real_pred_sm, 1), tf.argmax(self.ph_labels, 1))
            self.accuracy_per_im = tf.cast(correct_prediction, "float")
            self.real_accuracy = tf.reduce_mean(self.accuracy_per_im)

            net_vars = tf.trainable_variables()
            # this is a regularizer, a small weight decay
            adv_l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in net_vars
                                    if 'bias' not in v.name and 'adv' in v.name])

            self.adv_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(domain_pred, self.ph_domains), 0)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.adv_train_op = tf.train.MomentumOptimizer(self.learning_rate,0.9).minimize(self.adv_loss)

        tf.summary.scalar("summary_real_training_accuracy", self.real_accuracy)
        tf.summary.scalar("summary_real_loss", self.real_loss)
        self.summaries = tf.summary.merge_all()
        self.test_summary = tf.summary.scalar("summary_real_test_accuracy", self.real_accuracy)

        self.adv_summ = tf.summary.merge([tf.summary.scalar("adverserial loss", self.adv_loss)])
    
    def assign_batch(self, real_batch, adv_batch):
        self.real_batch = real_batch
        self.adv_batch = adv_batch

    @staticmethod
    def sample_with_replacement(large_batch, loss_estimates, batch_size, gamma=0.5):
        assert gamma <= 1.0
        # bimodal probabilities
        num_examples = large_batch['images'].shape[0]
        uniform_prob = 1.0 / num_examples
        uniform_prob_list = numpy.ones((num_examples,1)) * uniform_prob
        bimodal_dist = (1-gamma) * loss_estimates.reshape((num_examples,1)) + gamma * uniform_prob_list
        bimodal_dist_flat = bimodal_dist[:, 0]
        # this distribution might be distorted because of numerical error
        bimodal_dist_flat = bimodal_dist_flat / bimodal_dist_flat.sum()
        choices = numpy.random.choice(num_examples, batch_size, replace=True, p=bimodal_dist_flat)
        small_batch = {'images': large_batch['images'][choices], 'labels': large_batch['labels'][choices]}
        return small_batch

    def learning_step(self, session, batch_percent, final_m):
        fact = 2. / (1. + numpy.exp(-10. * batch_percent)) - 1
        lr = 0.0005 / (1. + 10 * batch_percent)**0.75 # 0.01 was the frst
        _ = session.run([self.real_train_op], feed_dict={self.ph_images: self.real_batch['images'],
                                                         self.ph_labels: self.real_batch['labels'],
                                                         self.phase: 1, self.learning_rate: lr})

        im = numpy.concatenate((self.real_batch['images'], self.adv_batch['images']), axis=0)
        l = numpy.concatenate(
            (numpy.ones((self.real_batch['images'].shape[0], 1)), numpy.zeros((self.adv_batch['images'].shape[0], 1))),
            axis=0)
        ll = numpy.concatenate((l, 1-l), axis=1)

        _ = session.run([self.adv_train_op], feed_dict={self.ph_images: im, self.ph_domains: ll, self.phase: 1, self.flip_factor:(fact*final_m), self.learning_rate:lr})
        #print 'LS', self.real_batch['images'].shape, self.adv_batch['images'].shape

    def summary_step(self, session):
        summ, loss = session.run([self.summaries, self.real_loss],
                                 feed_dict={self.ph_images: self.real_batch['images'],
                                            self.ph_labels: self.real_batch['labels'], self.phase: 0})

        im = numpy.concatenate((self.real_batch['images'], self.adv_batch['images']), axis=0)
        l = numpy.concatenate(
            (numpy.ones((self.real_batch['images'].shape[0],1)),numpy.zeros((self.adv_batch['images'].shape[0],1))),
            axis=0)
        ll = numpy.concatenate((l,1-l), axis=1)

        adv_summ, adv_loss = session.run([self.adv_summ, self.adv_loss],
                                         feed_dict={self.ph_images: im,
                                                    self.ph_domains: ll, self.phase: 0})

        return loss, summ, adv_loss, adv_summ

    def test_step(self, images, labels, session):
        acc, summ = session.run([self.real_accuracy, self.test_summary],
                                feed_dict={self.ph_images: images,
                                           self.ph_labels: labels, self.phase: 0})
        return acc, summ

    def active_sample(self, session, data, how_many,cl):
        print "starting"
        num_images = data['images'].shape[0]
        num_batch = int(numpy.ceil(num_images*1.0/self.batch_size))
        all_feat = numpy.zeros((num_images,512))
        all_l = numpy.zeros((num_images,cl))
        gt_l = numpy.zeros((num_images,cl))
        for b in range(num_batch):
            set_min = b*self.batch_size
            set_max = min((b+1)*self.batch_size, num_images)
            if set_max > set_min:
                im = data['images'][set_min:set_max]
                l = data['labels'][set_min:set_max]
                feat_values, lb = session.run([self.features,self.pred_l],
                                                         feed_dict={self.ph_images: im,
                                                                    self.ph_labels: l,
                                                                    self.phase: 0})

                all_feat[set_min:set_max] = feat_values
                all_l[set_min:set_max] = lb
                gt_l[set_min:set_max] = l
        return all_feat, all_l, gt_l

