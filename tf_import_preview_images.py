import tensorflow as tf
import os
import random
from os import listdir
from os.path import join
from shutil import copyfile

from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.data_utils import get_file
import matplotlib.pyplot as plot

# Use absolute path to save downloaded and datasets
# Default download is /Users/[Username]/.keras/
# Default cache datasets is /Users/[Username]/.keras/datasets/
# Otherwise datasets will be put under working directory.
#
# If "{R_PATH}/" is not provided all files will be placed under  "/tmp/.keras/datasets"
# or "/tmp/" automatically.
R_PATH = os.getcwd()  # os.getcwd() "/tmp/.keras"


def download_image_package(dest_dir, cache_dir):
    """
    Download classical cats & dogs image page.

    :param dest_dir: Location to put downloaded file, the extract place is the same otherwise.
    :param cache_dir: Location to put "datasets".
    """
    assert (dest_dir is not None)
    assert (cache_dir is not None)

    print(f"Download to {dest_dir}")
    return get_file(fname=dest_dir,
                    cache_dir=cache_dir,
                    origin="https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip",
                    extract=True)


downloaded_file = download_image_package(f"{R_PATH}/cats_and_dogs.zip", R_PATH)
print(f"Just downloaded: {downloaded_file}")


def split_data(source, training, testing, split_size):
    """
     Write a python function called split_data which takes

     :param source: a source directory containing the files
     :param training: a training directory that a portion of the files will be copied to
     :param testing: a testing directory that a portion of the files will be copied to
     :param split_size: a split_size SIZE to determine the portion

     The files should also be randomized, so that the training set is a random
     X% of the files, and the test set is the remaining files
     SO, for example, if SOURCE is PetImages/Cat, and SPLIT SIZE is .9
     Then 90% of the images in PetImages/Cat will be copied to the TRAINING dir
     and 10% of the images will be copied to the TESTING dir
     Also -- All images should be checked, and if they have a zero file length,
     they will not be copied over

     os.listdir(DIRECTORY) gives you a listing of the contents of that directory
     os.path.getsize(PATH) gives you the size of the file
     copyfile(source, destination) copies a file from source to destination
     random.sample(list, len(list)) shuffles a list
    """
    assert (source is not None)
    assert (training is not None)
    assert (testing is not None)
    assert (split_size is not None)

    list_of_files = os.listdir(source)
    if len(list_of_files) == 0:
        return

    count_of_files = len(list_of_files)

    split = int(count_of_files * split_size)
    list_of_files = random.sample(list_of_files, count_of_files)

    list_of_training_files = list_of_files[:split]
    list_of_testing_files = list_of_files[split:]

    def copy_file(s, d):
        if os.path.getsize(s) > 0:
            copyfile(s, d)
        else:
            print(s)

    [copy_file(source + f, training + f) for f in list_of_training_files]
    [copy_file(source + f, testing + f) for f in list_of_testing_files]


def create_training_testing_data(datasets, base_dir, split_size):
    """
     Create directory to store all training data and testing data.

     :param datasets: Path of datasets where the origin data is stored.
     :param base_dir: The root of location to put training and testing data.
     :param split_size: a split_size SIZE to determine the portion.

    base_dir
             training
                 cats
                     cat1.png
                     cat2.png
                 dogs
                     dog1.png
                     dog2.png
             testing
                 cats
                     cat1.png
                     cat2.png
                 dogs
                     dog1.png
                     dog2.png

    """
    assert (datasets is not None)
    assert (base_dir is not None)

    cat_source_dir = join(datasets, "PetImages/Cat/")
    dog_source_dir = join(datasets, "PetImages/Dog/")

    assert (os.path.exists(cat_source_dir))
    assert (os.path.exists(dog_source_dir))

    training_dir = join(base_dir, "training/")
    testing_dir = join(base_dir, "testing/")

    training_cats_dir = join(training_dir, "cats/")
    testing_cats_dir = join(testing_dir, "cats/")

    training_dogs_dir = join(training_dir, "dogs/")
    testing_dogs_dir = join(testing_dir, "dogs/")

    os.makedirs(training_cats_dir, exist_ok=True)
    os.makedirs(testing_cats_dir, exist_ok=True)

    os.makedirs(training_dogs_dir, exist_ok=True)
    os.makedirs(testing_dogs_dir, exist_ok=True)

    assert (os.path.exists(training_cats_dir))
    assert (os.path.exists(testing_cats_dir))
    assert (os.path.exists(training_dogs_dir))
    assert (os.path.exists(testing_dogs_dir))

    split_data(cat_source_dir, training_cats_dir, testing_cats_dir, split_size)
    split_data(dog_source_dir, training_dogs_dir, testing_dogs_dir, split_size)


create_training_testing_data(datasets=f"{R_PATH}/datasets/",
                             base_dir=f"{R_PATH}/cats_and_dogs/",
                             split_size=.9)
listdir(f"{R_PATH}")
[print(f, end=" ") for f in listdir(f"{R_PATH}/cats_and_dogs/")]
print()
[print(f, end=" ") for f in listdir(f"{R_PATH}/cats_and_dogs/training/")]
[print(f, end=" ") for f in listdir(f"{R_PATH}/cats_and_dogs/testing/")]
print()
[print(f, end=" ") for f in listdir(f"{R_PATH}/cats_and_dogs/training/dogs/")[:5]]
print()
[print(f, end=" ") for f in listdir(f"{R_PATH}/cats_and_dogs/training/cats/")[:5]]
print()
[print(f, end=" ") for f in listdir(f"{R_PATH}/cats_and_dogs/testing/dogs/")[:5]]
print()
[print(f, end=" ") for f in listdir(f"{R_PATH}/cats_and_dogs/testing/cats/")[:5]]
print()


def plot_images(images_labels_arr, classes):
    import math
    """
    Draw images

    :param images_labels_arr: A list of (image, label) tuples 
                              which are generated by ImageDataGenerator:
                              list-0: ([img0, img1...], [label0, label1...])
                              list-1: ([img6, img7...], [label6, label7...])
                              ........
    :param classes: The list of classes, use classes[label].
    """
    assert (images_labels_arr)
    assert (classes)

    for images_labels in images_labels_arr:
        images = images_labels[0]
        labels = images_labels[1]

        row_length = 5
        row_count = math.ceil(len(images) / row_length)

        iter = 0
        for i in range(row_count):
            fig, axes = plot.subplots(1, row_length, figsize=(20, 20))

            axes = axes.flatten()
            for img, label, ax in zip(images[iter:iter + row_length],
                                      labels[iter:iter + row_length],
                                      axes):
                ax.imshow(img)
                ax.set_title(f"{classes[int(label)]}???")

            plot.tight_layout()
            iter += row_length

        plot.show()


def generate_images_from_directory(from_directory,
                                   batch_size=5,
                                   width=150,
                                   height=150,
                                   rescale=1 / 255):
    """
    Read images from a directory.
    """
    assert (from_directory is not None)

    return ImageDataGenerator(rescale=rescale,
                              featurewise_std_normalization=True,
                              ).flow_from_directory(from_directory,
                                                    class_mode="sparse",
                                                    target_size=(width, height),
                                                    batch_size=batch_size)


# Return tuple
# https://tensorflow.google.cn/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory?hl=en
batch_size = 5
gen = generate_images_from_directory(
    from_directory=f"{R_PATH}/cats_and_dogs/training/",
    batch_size=batch_size
)

# Read some batches, each batch-size defined by generate_images_from_directory
augmented_images = []
for batch in range(10):
    images_labels = gen[batch]
    print(type(images_labels))

    images = gen[batch][0]
    print(images.shape)

    labels = gen[batch][1]
    print(labels.shape)

    augmented_images.append(images_labels)

print(gen.classes[0])
plot_images(augmented_images, dict((v, k) for k, v in gen.class_indices.items()))
