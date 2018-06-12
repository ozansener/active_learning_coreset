from cifar_trainer import Cifar10Trainer
import cifar10_data
import tensorflow as tf
from datetime import datetime
import click
import json

def read_params(file_name):
    class Param(object):
        pass

    params = Param()
    with open(file_name) as json_data:
        d = json.load(json_data)
        params.batch_size = d['batch_size']
        params.learning_rate = d['learning_rate']
        params.dropout = d['dropout']
    str_v = "learning_rate_{}__batch_size_{}__dropout_{}".format(params.batch_size, params.learning_rate, params.dropout)

    return params, str_v

@click.command()
@click.option('--hold_out', default=0, help='Training data size.')
@click.option('--dev_name', default='/gpu:0', help='Device name to use.')
def train(hold_out, dev_name):
    hold_out_s = int(hold_out)
    print hold_out_s
    train_data = cifar10_data.read_data_sets('./data/', one_hot=True, hold_out_size=hold_out_s)

    # partial test accuracy
    test_im = train_data.test.images[0:1000]
    test_l = train_data.test.labels[0:1000]

    params, str_v = read_params('settings.json')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sesh:
        cifar_train = Cifar10Trainer(learning_rate=params.learning_rate, device_name=dev_name, isdropout=params.dropout)
        print 'Initial Variable List:'
        print [tt.name for tt in tf.trainable_variables()]
        init = tf.global_variables_initializer()
        sesh.run(init)
        saver = tf.train.Saver(max_to_keep=100)
        sum_writer = tf.summary.FileWriter("./dumps_hold_out_{}__{}/".format(hold_out_s,str_v), sesh.graph)

        for batch_id in range(200000):
            im, l = train_data.train.next_batch(params.batch_size)
            # this is onlt active if dropout is enabled, otherwise nothing at all
            if batch_id < 500:
                kp = 1.0
            elif batch_id < 20000:
                kp = 0.9
            else:
                kp = 0.8

            top = cifar_train.train_step(im, l, sesh, kp)

            if batch_id %10 == 0:
                loss, summ = cifar_train.summary_step(im, l, sesh)
                sum_writer.add_summary(summ, batch_id)

            # save every 100th batch model
            if batch_id % 500 == 0:
                saver.save(sesh, 'models/cifar10_model_hold_out_{}__{}'.format(hold_out_s,str_v), global_step=batch_id)
                acc, test_sum = cifar_train.test_step(test_im, test_l, sesh)
                print "{}: step{}, test acc {}".format(datetime.now(), batch_id, acc)
                sum_writer.add_summary(test_sum, batch_id)

            if batch_id % 100 == 0:
                print "{}: step {}, loss {}".format(datetime.now(), batch_id, loss)

if __name__ == '__main__':
    train()
