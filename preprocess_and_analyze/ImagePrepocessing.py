import csv
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# Function that will read a image, resize it from 244 x 244 ---> 128 x 128
# using interpolation, then transform it from RGB ---> grayscale; the value returned
# will be 1D array, normalized
def process_image_CNN(img):
    image = cv2.imread(img)
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image / 255

def process_image_NB(img):
    image = cv2.imread(img)
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image.flatten()

# Function that will receive numpy array of images of size 
# (images, 128, 128, 1) and their corresponding labe
def generate_augumented_images(images, labels, num_of_images_to_generate, batch_size):
    # Transform the number of images so it can be matched with the batch size
    num_of_images_to_generate = num_of_images_to_generate + num_of_images_to_generate % batch_size

    x_augumented = np.empty((num_of_images_to_generate, 128, 128, 1))
    y_augumented = np.empty(num_of_images_to_generate,)

    # Keras utility that will generated augmented images, by shifting up & down, 
    # flipping, rotatings
    train_datagen = ImageDataGenerator(
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        shear_range=0.3
    )

    num_images_generated = 0
    for batch in train_datagen.flow(images, labels, batch_size=batch_size):
        x_batch, y_batch = batch

        # Generate images until there is no more space in the
        # array of generated images
        if x_augumented[num_images_generated:num_images_generated + batch_size].shape != x_batch.shape:
            break

        x_augumented[num_images_generated:num_images_generated + batch_size] = x_batch
        y_augumented[num_images_generated:num_images_generated + batch_size] = y_batch

        num_images_generated += batch_size

    return x_augumented, y_augumented

# Function that will help with reading the
# testing and validation files
def read_labels(file_name):
    f = open(file_name, "r")
    # ignore the first line, which points out the structure of the file
    f.readline()
    # read all the upcoming lines
    all_strings = f.readlines()
    all_labels = []

    for string in all_strings:
        # Line structure: num_of_image, label
        all_labels.append(int(string.split(",")[1][:1]))

    return all_labels

def read_images(images_paths, classification_type):
    # Read every image in the given array of paths
    # and return a 2D numpy vector of [image, pixels]
    images = []
    if classification_type == "CNN":
        for img in images_paths:
            images.append(process_image_CNN(img))
    else:
        for img in images_paths:
            images.append(process_image_NB(img))
    return np.array(images)

# Write to csv submission file using DictWriter ---> dictionary writer
# where each entry will be: [id, class]
# class = 0 / 1, id = [17001, 22149]
def write_submission(predicted_labels):
    with open("sample_submission.csv", mode="w", newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['id', 'class'])

        writer.writeheader()
        for cnt in range(len(predicted_labels)):
            writer.writerow({'id': f"0{cnt+17001}", 'class': predicted_labels[cnt]})