"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import scipy.io as sio
import gzip
import os
import glob

from six.moves import urllib
import numpy
import cPickle

def extract_test_data(work_directory, one_hot):
    print('Extracting', work_directory)
    t_dat = sio.loadmat(work_directory)
    num_images = t_dat['y'].shape[0]

    images = numpy.transpose(t_dat['X'],axes=[3,0,1,2]).astype('float32')
    labels = t_dat['y'].ravel()
   
    images[:,:,:,0] = images[:,:,:,0] - 112.0
    images[:,:,:,1] = images[:,:,:,1] - 113.0
    images[:,:,:,2] = images[:,:,:,2] - 121.0
 
    if one_hot:
        return images, dense_to_one_hot(labels)
    else:
        return images, labels

def extract_train_data(work_directory, one_hot, hold_out_size):
    print('Extracting', work_directory)
    print("Hold Out Size", hold_out_size)

    t_dat = sio.loadmat(work_directory)
    num_images = t_dat['y'].shape[0]

    _images = numpy.transpose(t_dat['X'],axes=[3,0,1,2]).astype('float32')
    _labels = t_dat['y'].ravel()
   
    _images[:,:,:,0] = _images[:,:,:,0] - 112.0
    _images[:,:,:,1] = _images[:,:,:,1] - 113.0
    _images[:,:,:,2] = _images[:,:,:,2] - 121.0


    images = numpy.zeros((num_images - hold_out_size, 32,32,3))
    labels = numpy.zeros((num_images - hold_out_size))

    if hold_out_size > 0:
        hold_out_images = numpy.zeros((hold_out_size, 32,32,3))
        hold_out_labels = numpy.zeros((hold_out_size))
    else:
        hold_out_images = None
        hold_out_labels = None

    hold_out_images=_images[0:hold_out_size,:,:,:]
    hold_out_labels=_labels[0:hold_out_size]

    remaining_size = num_images = hold_out_size
    images=_images[hold_out_size:,:,:,:]
    labels=_labels[hold_out_size:]

    if one_hot:
        if hold_out_size > 0: 
            return images, dense_to_one_hot(labels), hold_out_images, dense_to_one_hot(hold_out_labels)
        else:
            return images, dense_to_one_hot(labels), None, None
    else:
        return images, labels, hold_out_images, hold_out_labels


def dense_to_one_hot(labels_dense, num_classes=100):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.astype(int).ravel()] = 1
    return labels_one_hot


class DataSet(object):
    def __init__(self, images, labels, fake_data=False):
        if fake_data:
            self._num_examples = 10000
        else:
            assert images.shape[0] == labels.shape[0], (
                "images.shape: %s labels.shape: %s" % (images.shape,
                                                       labels.shape))
            self._num_examples = images.shape[0]

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            #assert images.shape[3] == 1
            #images = images.reshape(images.shape[0],
            #                        images.shape[1] * images.shape[2])
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(numpy.float32)
            images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1.0 for _ in xrange(784)]
            fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            #print('Re-order the dataset')
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir,
                   one_hot=True,
                   hold_out_size=0,
                   active=False,
                   choices=None):
    class DataSets(object):
        pass

    data_sets = DataSets()

    if hold_out_size > 0:
        if active:
            # this is active
            im, l, h_im, h_l = extract_train_data(train_dir+'/train_32x32.mat', one_hot, 0)
            print(im.size)
            all_im = im[choices]
            all_l = l[choices]
            data_sets.train = DataSet(all_im, all_l)
            rest = list(set(range(im.shape[0])) - set(choices))
            r_im = im[rest]
            r_l = l[rest]
            data_sets.hold_out = DataSet(r_im, r_l)
            im_t, l_t = extract_test_data(train_dir+'/test_32x43.mat', one_hot)
            data_sets.test = DataSet(im_t, l_t)
        else:
            im, l, h_im, h_l = extract_train_data(train_dir+'/train_32x32.mat', one_hot, hold_out_size)
            data_sets.train = DataSet(h_im,h_l)
            data_sets.hold_out = DataSet(im,l)
            im_t,l_t = extract_test_data(train_dir+'/test_32x32.mat', one_hot)
            data_sets.test = DataSet(im_t,l_t)
    else:
        im, l, h_im, h_l = extract_train_data(train_dir+'/train_32x32.mat', one_hot, hold_out_size)
        data_sets.train = DataSet(im, l)
        im_t, l_t = extract_test_data(train_dir+'/text_32x43.mat', one_hot)
        data_sets.test = DataSet(im_t, l_t)

    return data_sets

