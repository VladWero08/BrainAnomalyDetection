import glob

import sys, os
sys.path.append(os.path.join('..', 'preprocess_and_analyze'))

import preprocess_and_analyze.ImagePrepocessing as img_preprocess
import preprocess_and_analyze.ClassifierAnalyzer as analyzer

import numpy as np
from sklearn.naive_bayes import MultinomialNB

# Function that will generated a certain number of intervals
# with the smallest edge value being 0, and largest being 255
def generate_intervals(num_intervals):
    bins = np.linspace(start=0, stop=256, num=num_intervals)
    return bins

# Transform the corresponding pixels so they can fit into an interval
def transform_images_values_to_bins(images, bins):
    for cnt, image in enumerate(images):
        images[image] = np.digitize(image, bins)
    return images - 1

# Hyperparameter tuning for the number of intervals
# in which the pixels will be split
def Naive_Bayes_classification():
    global training_images, training_labels, validation_images, validation_labels
    naive_bayes_model = MultinomialNB()

    intervals = [3, 4, 5, 6, 7, 8]
    for interval in intervals:
        new_bins = generate_intervals(interval)

        x_train = transform_images_values_to_bins(training_images, new_bins)
        x_validation = transform_images_values_to_bins(validation_images, new_bins)

        naive_bayes_model.fit(x_train, training_labels)
        validation_predictions = naive_bayes_model.predict(x_validation)
        analyzer.generate_metrics(validation_labels, validation_predictions)

# Read training, testing and validation images
images_paths = glob.glob("./data/*.png")
training_images_paths = images_paths[:15000]
validation_images_paths = images_paths[15000:17000]
testing_images_paths = images_paths[17000:]

training_images = img_preprocess.read_images(training_images_paths, "NB")
training_labels = img_preprocess.read_labels("train_labels.txt")
validation_images = img_preprocess.read_images(validation_images_paths, "NB")
validation_labels = img_preprocess.read_labels("validation_labels.txt")
testing_images = img_preprocess.read_images(testing_images_paths, "NB")

naive_bayes_model = MultinomialNB()
new_bins = generate_intervals(4)

x_train = transform_images_values_to_bins(training_images, new_bins)
x_validation = transform_images_values_to_bins(validation_images, new_bins)

naive_bayes_model.fit(x_train, training_labels)
validation_predictions = naive_bayes_model.predict(x_validation)
analyzer.generate_metrics(validation_labels, validation_predictions)
