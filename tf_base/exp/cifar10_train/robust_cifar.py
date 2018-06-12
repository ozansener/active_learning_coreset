from robust_trainer import RobustTrainer
import cifar10_data
import tensorflow as tf
from datetime import datetime
import click
import json
import numpy
import pickle


def read_params(file_name):
    class Param(object):
        pass

    params = Param()
    with open(file_name) as json_data:
        d = json.load(json_data)
        params.batch_size = d['batch_size']
        params.learning_rate = d['learning_rate']
        params.dropout = d['dropout']
    str_v = "robust_learning_rate_{}__batch_size_{}__dropout_{}".format(params.batch_size, params.learning_rate, params.dropout)

    return params, str_v

@click.command()
@click.option('--hold_out', default=0, help='Training data size.')
@click.option('--dev_name', default='/gpu:0', help='Device name to use.')
@click.option('--sample/--no-sample', default=True)
@click.option('--go_on/--no-go_on', default=False)
def train(hold_out, dev_name, sample, go_on):
    hold_out_s = int(hold_out)
    print hold_out_s
    if go_on:
        c = pickle.load(open('chosen_data_{}_{}'.format(hold_out_s, sample)))
        ch = c['chosen'] - hold_out_s
        train_data = cifar10_data.read_data_sets('./data/',
                                                 one_hot=True, hold_out_size=hold_out_s, active=True, choices=ch)
    else:
        train_data = cifar10_data.read_data_sets('./data/', one_hot=True, hold_out_size=hold_out_s)

    test_im = train_data.test.images[0:1000]
    test_l = train_data.test.labels[0:1000]

    params, str_v = read_params('settings.json')

    num_b = 5 * numpy.ceil( train_data.train.images.shape[0] / ((1.0)*params.batch_size) )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sesh:
        robust_cifar_train = RobustTrainer(params.learning_rate, params.learning_rate, dev_name)
        print 'Initial Variable List:'
        print [tt.name for tt in tf.trainable_variables()]
        print "Wait {} many batches".format(num_b)

        init = tf.global_variables_initializer()
        sesh.run(init)
        saver = tf.train.Saver(max_to_keep=100)
        sum_writer = tf.summary.FileWriter("./dumps2_robust__go_on_{}__sample_{}__hold_out_{}__{}/".format(go_on,sample,hold_out_s,str_v), sesh.graph)

        im, l = train_data.train.next_batch(params.batch_size*8)  # Sample a boosted set    
        for batch_id in range(200000):
            im, l = train_data.train.next_batch(params.batch_size*8)  # Sample a boosted set
            if batch_id < 500:
                kp = 1.0
            elif batch_id < 20000:
                kp = 0.9
            else:
                kp = 0.8

            robust_cifar_train.assign_batch({'images': im, 'labels': l})
            # first epoch only observe then do staff
            if batch_id < num_b:
                gamma = 1.0
            elif sample:
                gamma = 0.5
            else:
                gamma = 1.0
            # if gamma is 1.0, this is classical training otherwise it is L_max

            # here we first try learn everything
            create_summ = batch_id % 10 == 0
            d = robust_cifar_train.learning_step(sesh, gamma, create_summ, True, True)

            if create_summ:
                loss, summ = robust_cifar_train.summary_step(im, l, sesh)
                sum_writer.add_summary(summ, batch_id)
                sum_writer.add_summary(d, batch_id)

            # save every 100th batch model
            if batch_id % 500 == 0:
                saver.save(sesh, 'models/cifar10_go_on_{}_robust_{}_model_hold_out_{}__{}'.format(go_on,sample,hold_out_s,str_v), global_step=batch_id)
                acc, test_sum = robust_cifar_train.test_step(test_im, test_l, sesh)
                print "{}: step{}, test acc {}".format(datetime.now(), batch_id, acc)
                sum_writer.add_summary(test_sum, batch_id)

            if batch_id % 100 == 0:
                print "{}: step {}, loss {}".format(datetime.now(), batch_id, loss)

if __name__ == '__main__':
    train()
