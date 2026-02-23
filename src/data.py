import os
import tensorflow as tf
from sklearn.utils import shuffle

SEED = 111

def get_data_labels(directory, shuffle = True, random_state = 0):
    from sklearn.utils import shuffle
    data_path = []
    data_index = []
    class_names = sorted(os.listdir(directory))
    label_dict = {class_name: idx for idx, class_name in enumerate(class_names)}
    # label_dict = {label: index for label, index in enumerate(sorted(os.listdir(directory)))}

    for label, index in label_dict.items():
        label_dir = os.path.join(directory, label)
        for image in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image)
            data_path.append(image_path)
            data_index.append(index)

    if shuffle:
        data_path, data_index = shuffle(data_path, data_index, random_state = random_state)

    return data_path, data_index

def parse_function(filename, label, image_size, n_channels):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, n_channels)
    image = tf.image.resize(image, image_size)
    return image, label


def get_dataset(paths, labels, image_size, n_channels=1, num_classes = 4, batch_size = 32):
    path_ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    
    image_label_ds = path_ds.map(lambda path, label: parse_function(path, label, image_size, n_channels),
                                 num_parallel_calls = tf.data.AUTOTUNE)
    return image_label_ds.batch(batch_size).prefetch(buffer_size = tf.data.AUTOTUNE)
    