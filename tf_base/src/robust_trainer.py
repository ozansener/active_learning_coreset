import tensorflow as tf
import numpy
from network.vgg_robust import VGG16Adversery, VGG16Robust

class RobustTrainer(object):
    """
    Train a network using robust objective
    """
    def __init__(self, learning_rate_net, learning_rate_adv, device_name):
        """
        Simple diagram:
         Images -> | FeatureNet -> FC | -------------------x-> loss
                                |_(no grad pass) -> |Adv| _|
        :return:
        """
        self.batch_size = 128
        dim = 512
        real_weight_decay = 0.002
        self.ph_images = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.ph_labels = tf.placeholder(tf.float32, [None, 10])
        self.ph_features = tf.placeholder(tf.float32, [None, dim])
        self.ph_per_image_loss = tf.placeholder(tf.float32, [None, 2])
        self.keep_prob = tf.placeholder(tf.float32)
        self.phase = tf.placeholder(tf.bool, name='phase')  # train or test for batch norm
        self.lr_adv = learning_rate_adv

        with tf.device(device_name):
            real_net = VGG16Robust({'data': self.ph_images}, phase=self.phase)
            self.features = tf.reshape(real_net.layers['feat'], [-1, dim])
            real_pred = real_net.get_output()
            real_pred_sm = tf.nn.softmax(real_pred)  # this is for usage in accuracy, loss is combined with softmax

            real_net_vars = tf.trainable_variables()
            real_l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in real_net_vars if 'bias' not in v.name]) * real_weight_decay

            per_image_loss = tf.nn.softmax_cross_entropy_with_logits(real_pred, self.ph_labels)
            self.real_loss = tf.reduce_mean(per_image_loss, 0) + real_l2_loss

            # this is simply adding batch norm moving average to train operations
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.real_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate_net).minimize(self.real_loss)

            correct_prediction = tf.equal(tf.argmax(real_pred_sm, 1), tf.argmax(self.ph_labels, 1))
            self.accuracy_per_im = tf.cast(correct_prediction, "float")
            self.real_accuracy = tf.reduce_mean(self.accuracy_per_im)

            adv_net = VGG16Adversery({'net_features': self.ph_features}, phase=self.phase)
            self.adv_out = adv_net.get_output()  # num images x 2, 0 is error, 1 is not assuming binary classification loss
            # this is for observation only, softmax becomes numerically unstable if this is greater than a few thousands
            # note(ozan) never ever use this without a batch-norm, batch-norm stablize the hell out of it

            net_vars = tf.trainable_variables()
            # this is a regularizer, a small weight decay
            adv_l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in net_vars
                                      if 'bias' not in v.name and 'adv' in v.name ]) * 0.001

            self.adv_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.adv_out, self.ph_per_image_loss)) + adv_l2_loss
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.adv_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate_adv).minimize(self.adv_loss)

        tf.summary.scalar("summary_real_training_accuracy", self.real_accuracy)
        tf.summary.scalar("summary_real_loss", self.real_loss)
        self.summaries = tf.summary.merge_all()
        self.test_summary = tf.summary.scalar("summary_real_test_accuracy", self.real_accuracy)

        self.adv_summ = tf.summary.merge([tf.summary.scalar("adverserial loss", self.adv_loss)])
    
    def assign_batch(self, large_batch):
        self.large_batch = large_batch

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

    def learning_step(self, session, gamma, compute_adv_loss, train_adv, train_real):
        # compute features and loss for current batch
        feat_values, per_im_loss_d = session.run([self.features, self.accuracy_per_im],
                                                 feed_dict={self.ph_images: self.large_batch['images'],
                                                            self.ph_labels: self.large_batch['labels'],
                                                            self.phase: 0})

        per_im_miss_class = 1.0 - per_im_loss_d.astype(numpy.float32)
        per_im_loss = per_im_miss_class.reshape((per_im_miss_class.shape[0], 1))

        ce_lab = numpy.concatenate((per_im_loss, 1.0-per_im_loss), axis=1)

        if compute_adv_loss:
            adv_sum, loss_estimates = session.run([self.adv_summ, tf.nn.softmax(self.adv_out)], feed_dict={self.ph_features: feat_values,
                                                                                             self.ph_per_image_loss: ce_lab,
                                                                                             self.phase: 0})
        else:
            adv_sum = None
            loss_estimates = session.run(tf.nn.softmax(self.adv_out), feed_dict={self.ph_features: feat_values,
                                                                                 self.phase: 0})

        if train_adv:
            _ = session.run([self.adv_train_op], feed_dict={self.ph_features: feat_values,
                                                            self.ph_per_image_loss: ce_lab,
                                                            self.phase: 1})

        if train_real:
            unnormalized_prob = loss_estimates[:, 0]
            e_x = numpy.exp(unnormalized_prob - numpy.max(unnormalized_prob))
            prob = e_x / e_x.sum()

            small_batch = RobustTrainer.sample_with_replacement(self.large_batch,
                                                                prob,
                                                                self.batch_size,
                                                                gamma)

            _ = session.run([self.real_train_op], feed_dict={self.ph_images: small_batch['images'],
                                                             self.ph_labels: small_batch['labels']})
        return adv_sum

    def summary_step(self, images, labels, session):
        summ, loss = session.run([self.summaries, self.real_loss],
                                 feed_dict={self.ph_images: images,
                                            self.ph_labels: labels})
        return loss, summ

    def test_step(self, images, labels, session):
        acc, summ = session.run([self.real_accuracy, self.test_summary],
                                feed_dict={self.ph_images: images,
                                           self.ph_labels: labels})
        return acc, summ

    def active_sample(self, session, data, how_many):
        num_images = data['images'].shape[0]
        num_batch = int(numpy.ceil(num_images*1.0/self.batch_size))
        all_prob = numpy.zeros(num_images)
        all_cp = numpy.zeros(num_images)
        all_feat = numpy.zeros((num_images,512))
        all_l = numpy.zeros((num_images,10))
        for b in range(num_batch):
            set_min = b*self.batch_size
            set_max = min((b+1)*self.batch_size, num_images)
            if set_max > set_min:
                im = data['images'][set_min:set_max]
                l = data['labels'][set_min:set_max]
                feat_values, per_im_loss_d = session.run([self.features, self.accuracy_per_im],
                                                         feed_dict={self.ph_images: im,
                                                                    self.ph_labels: l,
                                                                    self.phase: 0})

                loss_estimates = session.run(tf.nn.softmax(self.adv_out), feed_dict={self.ph_features: feat_values, self.phase: 0})
                unnormalized_prob = loss_estimates[:, 0]
                #e_x = numpy.exp(unnormalized_prob - numpy.max(unnormalized_prob))
                #prob = e_x / e_x.sum()
                all_prob[set_min:set_max] = unnormalized_prob
                all_cp[set_min:set_max] = per_im_loss_d
                all_feat[set_min:set_max] = feat_values
                all_l[set_min:set_max] = l
        return all_prob, all_cp, all_feat, all_l

    def reinit_adam(self, session):
        temp = set(tf.all_variables())
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.adv_train_op = tf.train.AdamOptimizer(learning_rate=self.lr_adv).minimize(self.adv_loss)
        session.run(tf.variables_initializer(set(tf.all_variables()) - temp))

    def reinit_adverserial(self, session):
        net_vars = tf.trainable_variables()
        # this is a regularizer, a small weight decay
        adv_v = [ v for v in net_vars if 'adv' in v.name ]
        print 'Initializing ', [v.name for v in adv_v]
        init = tf.variables_initializer(adv_v)
        session.run(init)
        temp = set(tf.all_variables())
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.adv_train_op = tf.train.AdamOptimizer(learning_rate=self.lr_adv).minimize(self.adv_loss)
        session.run(tf.variables_initializer(set(tf.all_variables()) - temp))

    @staticmethod
    def samp(f, n, gamma):
        #th = numpy.sort(f)[-n]
        k = numpy.zeros(f.shape)
        k[f>0] = 1.0
        k = k / numpy.sum(k)
        uniform_prob = 1.0 / f.shape[0]
        num_examples = f.shape[0]
        uniform_prob_list = numpy.ones((num_examples,1)) * uniform_prob
        bimodal_dist = (1-gamma) * k.reshape((num_examples,1)) + gamma * uniform_prob_list
        bimodal_dist_flat = bimodal_dist[:, 0]
        bimodal_dist_flat = bimodal_dist_flat / bimodal_dist_flat.sum()
        choices = numpy.random.choice(num_examples, n, replace=False, p=bimodal_dist_flat)
        return choices
