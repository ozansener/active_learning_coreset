from adverserial_trainer_w import AdverserialTrainer
import svhn_data
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
@click.option('--semi/--no-semi',default=True)
def train(hold_out, dev_name, sample, go_on,semi):
    hold_out_s = int(hold_out)
    print hold_out_s
    if go_on:
        c = pickle.load(open('chosen_ball.bn'))
        ch = c['all']
        train_data = svhn_data.read_data_sets('./data/',
                                                 one_hot=True, hold_out_size=hold_out_s, active=True, choices=ch)
    else:
        train_data = svhn_data.read_data_sets('./data/', one_hot=True, hold_out_size=hold_out_s)

    test_im = train_data.test.images[0:1000]
    test_l = train_data.test.labels[0:1000]

    params, str_v = read_params('settings.json')
    print train_data.train.images.shape, train_data.hold_out.images.shape
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sesh:
        robust_cifar_train = AdverserialTrainer(params.learning_rate, 0.1*params.learning_rate, dev_name,100)
        print 'Initial Variable List:'
        print [tt.name for tt in tf.trainable_variables()]

        init = tf.global_variables_initializer()
        sesh.run(init)
        saver = tf.train.Saver(max_to_keep=100)
        sum_writer = tf.summary.FileWriter("./dumps_adverserial__semi_{}_go_on_{}__sample_{}__hold_out_{}__{}/".format(semi,go_on,sample,hold_out_s,str_v), sesh.graph)

        im, l = train_data.train.next_batch(params.batch_size)  # Sample a boosted set
        for batch_id in range(200000):
            im, l = train_data.train.next_batch(params.batch_size)  # Sample a boosted set
            adv_im, _ = train_data.hold_out.next_batch(params.batch_size)

            flip_f = float(batch_id)/200000.0

            robust_cifar_train.assign_batch({'images': im, 'labels': l},{'images': adv_im})
            # first epoch only observe then do staff

            create_summ = batch_id % 10 == 0
            if semi:
                robust_cifar_train.learning_step(sesh, flip_f,0.0)
            else:
                robust_cifar_train.learning_step(sesh, flip_f,1.0)
 
            if create_summ:
                loss, summ, a_l, a_s = robust_cifar_train.summary_step(sesh)
                sum_writer.add_summary(summ, batch_id)
                sum_writer.add_summary(a_s, batch_id)

            # save every 100th batch model
            if batch_id % 500 == 0:
                saver.save(sesh, 'models/svhn_go_on_{}_robust_{}_model_hold_out_{}__{}'.format(go_on,sample,hold_out_s,str_v), global_step=batch_id)
                acc, test_sum = robust_cifar_train.test_step(test_im, test_l, sesh)
                print "{}: step{}, test acc {}".format(datetime.now(), batch_id, acc)
                sum_writer.add_summary(test_sum, batch_id)

            if batch_id % 100 == 0:
                print "{}: step {}, loss {} adv_loss {}".format(datetime.now(), batch_id, loss, a_l)

if __name__ == '__main__':
    train()
