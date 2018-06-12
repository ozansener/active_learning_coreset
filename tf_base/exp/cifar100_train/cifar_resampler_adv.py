from adverserial_trainer import AdverserialTrainer
import cifar100_data
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
@click.option('--batch_id')
@click.option('--go_on/--no-go_on', default=False)
def train(hold_out, dev_name, sample, batch_id,go_on):
    hold_out_s = int(hold_out)
    print hold_out_s

    if go_on:
        c = pickle.load(open('chosen_ball.bn'))
        ch = c['all']
        train_data = cifar100_data.read_data_sets('./data/',
                                                 one_hot=True, hold_out_size=hold_out_s, active=True, choices=ch)
    else:
        train_data = cifar100_data.read_data_sets('./data/', one_hot=True, hold_out_size=hold_out_s)

    valid_im = train_data.hold_out.images[-5000:]
    valid_l = train_data.hold_out.labels[-5000:]

    params, str_v = read_params('settings.json')

    num_b = 5 * numpy.ceil( train_data.train.images.shape[0] / ((1.0)*params.batch_size) )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sesh:
        robust_cifar_train = AdverserialTrainer(params.learning_rate, params.learning_rate, dev_name,100)
        # no need to initialize anything since ve are simply re-loading the data
        saver = tf.train.Saver(max_to_keep=100)
        saver.restore(sesh, 'models/cifar10_go_on_{}_robust_{}_model_hold_out_{}__{}-{}'.format(go_on,sample,hold_out_s,str_v, batch_id))
        # here options, running the adv with more data (validation), learn adv with more data from scratch
        sum_writer = tf.summary.FileWriter("./dumps_robust__sample_{}__hold_out_{}__{}/".format(sample,hold_out_s,str_v), sesh.graph)
        #robust_cifar_train.reinit_adverserial(sesh)
        #b = -1
        #for c in range(1000):
        #    b+=1
        #    if (b+1)*1000 > 5000:
        #        perm = numpy.arange(5000)
        #        numpy.random.shuffle(perm)
        #        valid_im = valid_im[perm]
        #        valid_l = valid_l[perm]
        #        b = 0
        #    print 'Sampling, retraining, batch ', b
        #    set_min=b*1000
        #    set_max=(b+1)*1000
        #    im = valid_im[set_min:set_max]
        #    l = valid_l[set_min:set_max]
        #    robust_cifar_train.assign_batch({'images':im, 'labels':l})
        #    robust_cifar_train.learning_step(sesh, 0.5, False, True, False)

        #saver.save(sesh, 'models/sampler_robust_{}_model_hold_out_{}__{}_{}'.format(sample,hold_out_s,str_v,batch_id),
        #               global_step=0)
        f,l,lgt = robust_cifar_train.active_sample(sesh,
                                                  {'images':train_data.hold_out.images,
                                                   'labels':train_data.hold_out.labels}, 5000,100)
        f_t,l_t,lgt_t = robust_cifar_train.active_sample(sesh,
                                                  {'images':train_data.train.images,
                                                   'labels':train_data.train.labels}, 5000,100)
        #ch = RobustTrainer.samp(1 - gt[:-5000], 5000, 0.25)
        pickle.dump({'a':[f,l,lgt,f_t,l_t,lgt_t]}, open('chosen_100_data_{}_{}'.format(hold_out, sample), 'wb'))

if __name__ == '__main__':
    train()
