import tensorflow as tf
import matplotlib.pyplot as plt


if __name__ == "__main__":
    filename_queue = tf.train.string_input_producer(["./data/tfrecords/test.tfrecords"])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = features['label']

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        num = 12
        for j in range(num):
            img, lab = sess.run([image, label])
            plt.subplot(3, num/3, j+1)
            plt.title(lab)
            plt.imshow(img.reshape([28, 28]), cmap='gray')
        plt.show()
        coord.request_stop()
        coord.join(threads)
