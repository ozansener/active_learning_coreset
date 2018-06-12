from mnist_trainer import MnistTrainer
import mnist_data
import tensorflow as tf
from datetime import datetime

train_data = mnist_data.read_data_sets('./data/', one_hot=True)
test_im = train_data.test.images
test_l = train_data.test.labels

def get_batch():
    im, l = train_data.train.next_batch(256)
    return im, l

with tf.Session() as sesh:
    mnist_train = MnistTrainer()
    print 'Initial Variable List:'
    print [tt.name for tt in tf.trainable_variables()]
    init = tf.initialize_all_variables()
    sesh.run(init)
    saver = tf.train.Saver()
    for batch_id in range(20000):
        im, l = get_batch()
        loss = mnist_train.train_step(im, l, sesh)
       
        if batch_id % 100 == 0:
            saver.save(sesh,'models/mnist_model',global_step=batch_id)

        if batch_id % 10 == 0:
            print "{}: step {}, loss {}".format(datetime.now(), batch_id, loss)

    acc = mnist_train.compute_accuracy(test_im, test_l, sesh)
    print 'Final Accuracy: ', acc
