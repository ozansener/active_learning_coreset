import numpy as np
import tensorflow as tf
from network.alexnet import AlexNet
from image_reader import ImageReader
import glob

def display_results(image_paths, probs):
    '''Displays the classification results given the class probability for each image'''
    # Get a list of ImageNet class labels
    with open('imagenet-classes.txt', 'rb') as infile:
        class_labels = map(str.strip, infile.readlines())
    # Pick the class with the highest confidence for each image
    class_indices = np.argmax(probs, axis=1)
    # Display the results
    print('\n{:20} {:30} {}'.format('Image', 'Classified As', 'Confidence'))
    print('-' * 70)
    for img_idx, image_path in enumerate(image_paths):
        img_name = image_path
        class_name = class_labels[class_indices[img_idx]]
        confidence = round(probs[img_idx, class_indices[img_idx]] * 100, 2)
        print('{:20} {:30} {} %'.format(img_name, class_name, confidence))


input_data = tf.placeholder(tf.float32, shape=(None, 227, 227, 3))
input_data_2 = tf.placeholder(tf.float32, shape=(None, 227, 227, 3))
input_data_3 = tf.placeholder(tf.float32, shape=(None, 227, 227, 3))



net = AlexNet({'data':input_data})

print 'Initial Variable List:'
print [tt.name for tt in tf.trainable_variables()]

image_paths = glob.glob('data/*.JPEG')
image_reader = ImageReader(image_paths=image_paths, batch_size=100)

with tf.Session() as sesh:
    # load model weights
    model_data = 'alexnet_weights.npy'

    net.load(model_data, sesh)

    # start image reading
    coordinator = tf.train.Coordinator()
    threads = image_reader.start_reader(session = sesh, coordinator = coordinator)
    
    # get a batch
    indices, input_images = image_reader.get_batch(sesh)

    # get labels
    probs = sesh.run(net.get_output(), feed_dict={input_data: input_images})
    display_results([image_paths[i] for i in indices], probs)

    coordinator.request_stop()
    coordinator.join(threads, stop_grace_period_secs=2)

