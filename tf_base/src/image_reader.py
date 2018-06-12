import numpy as np
import tensorflow as tf


class ImageReader(object):
    """
    Asynchronous image reader and processer
    """

    def __init__(self, image_paths, num_concurrent=4, image_spec=None, batch_size=256, labels=None):
        if image_spec:
            self.image_spec = image_spec
        else:
            self.image_spec = {'scale_size': 256, 'crop_size': 227, 'mean': np.array([104., 117., 124.])}

        self.labels = labels
        self.setup_reader(image_paths, (self.image_spec['crop_size'], self.image_spec['crop_size'], 3), num_concurrent,
                          batch_size)

    @staticmethod
    def process_single_image(img, scale, crop, mean):
        """
        Processing an image for zero-mean/std-dev fix etc
        """
        new_shape = tf.pack([scale, scale])
        img = tf.image.resize_images(img, new_shape[0], new_shape[1])
        offset = (new_shape - crop) / 2
        img = tf.slice(img, begin=tf.pack([offset[0], offset[1], 0]), size=tf.pack([crop, crop, -1]))
        return tf.to_float(img) - mean

    def process(self):
        idx, image_path = self.path_queue.dequeue()
        img = tf.image.decode_jpeg(tf.read_file(image_path), channels=3)  # It is an RGB PNG
        img = tf.reverse(img, [False, False, True])  # RGB -> BGR
        return (idx, ImageReader.process_single_image(img, self.image_spec['scale_size'],
                                                      self.image_spec['crop_size'],
                                                      self.image_spec['mean']))

    def setup_reader(self, image_paths, image_shape, num_concurrent, batch_size):
        # Path queue is list of image paths which will further be processed by another queue
        num_images = len(image_paths)
        indices = tf.range(0, num_images, 1)

        self.path_queue = tf.FIFOQueue(capacity=num_images, dtypes=[tf.int32, tf.string], name='path_queue')
        self.enqueue_path = self.path_queue.enqueue_many([indices, image_paths])
        self.close_path = self.path_queue.close()

        processed_queue = tf.FIFOQueue(capacity=num_images,
                                       dtypes=[tf.int32, tf.float32],
                                       shapes=[(), image_shape],
                                       name='processed_queue')

        (idx, processed_image) = self.process()
        enqueue_process = processed_queue.enqueue([idx, processed_image])
        self.dequeue_batch = processed_queue.dequeue_many(batch_size)

        self.queue_runner = tf.train.QueueRunner(processed_queue, [enqueue_process] * num_concurrent)

    def start_reader(self, session, coordinator):
        session.run(self.enqueue_path)
        session.run(self.close_path)
        return self.queue_runner.create_threads(session, coord=coordinator, start=True)

    def get_batch(self, session):
        (indices, images) = session.run(self.dequeue_batch)
        if self.labels:
            labels = [self.labels[idx] for idx in indices]
            return (labels, images)
        return (indices, images)

    def batches(self, session, num_batch):
        for _ in xrange(num_batch):
            yield self.get_batches(session=session)
