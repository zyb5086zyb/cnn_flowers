from skimage import io, transform
import tensorflow as tf
import numpy as np


path_dir = []
path_dir.append('G:/flower_photos/test_data/21652746_cc379e0eea_m.jpg')
path_dir.append('G:/flower_photos/test_data/8181477_8cb77d2e0f_n.jpg')
path_dir.append('G:/flower_photos/test_data/12240303_80d87f77a3_n.jpg')
path_dir.append('G:/flower_photos/test_data/6953297_8576bf4ea3.jpg')
path_dir.append('G:/flower_photos/test_data/10791227_7168491604.jpg')

flower_dict = {0: 'dasiy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}
w = 100
h = 100
c = 3


def read_one_image(path):
    img = io.imread(path)
    img = transform.resize(img, (w, h))
    return np.asarray(img)


with tf.Session() as sess:
    data = []
    for path in path_dir:
        data_dir = read_one_image(path)
        data.append(data_dir)
    saver = tf.train.import_meta_graph('G:/flower_photos/model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('G:/flower_photos/'))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("input-x:0")
    feed_dict = {x: data}
    logits = graph.get_tensor_by_name("logits_eval:0")
    classification_result = sess.run(logits, feed_dict)

    print(classification_result)
    print(tf.argmax(classification_result, 1).eval())
    output = []
    output = tf.argmax(classification_result, 1).eval()
    for i in range(len(output)):
        print("第", i + i, "朵花预测结果是:" + flower_dict[output[i]])


