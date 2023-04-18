import glob
import preprocess_and_analyze.ImagePrepocessing as ImagePrepocessing
import preprocess_and_analyze.ClassifierAnalyzer as analyzer
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary

import CNN_training_and_testing
import models.ResNet18 as RESNET18

# Splitting the images in the file into training and validation
images_paths = glob.glob("../input/unibuc-brain-ad/data/data/*.png")
images_paths.sort()

training_images_paths = images_paths[:15000]
validation_images_paths = images_paths[15000:17000]
testing_images_paths = images_paths[17000:]

training_images = np.expand_dims(ImagePrepocessing.read_images(training_images_paths, "CNN"), axis=3)
print("100% --- Loaded training images")
training_labels = np.asarray(ImagePrepocessing.read_labels("../input/unibuc-brain-ad/data/train_labels.txt"))
print("100% --- Loaded training labels")

# Generate 20000 augmented images and concatenate them to the initial
# testing images & testing_labels array
training_images_augumented, training_labels_augumented = ImagePrepocessing.generate_augumented_images(training_images, training_labels, 20000, 32)
training_images = np.concatenate((training_images, training_images_augumented), axis=0)
training_labels = np.concatenate((training_labels, training_labels_augumented), axis=0)

validation_images = np.expand_dims(ImagePrepocessing.read_images(validation_images_paths, "CNN"), axis=3)
print("100% --- Loaded validation images")
validation_labels = np.asarray(ImagePrepocessing.read_labels("../input/unibuc-brain-ad/data/validation_labels.txt"))
print("100% --- Loaded validations labels")
testing_images = np.expand_dims(ImagePrepocessing.read_images(testing_images_paths, "CNN"), axis=3)
print("100% --- Loaded testing images")

def transform_numpy_to_tensors(images, labels):
    return Tensor(images), Tensor(labels)

# Function that will resize tensor from (num_images, height, width, 1) ==> (num_images, 1, height, width)
def resize_tensor(tensor):
    return tensor.view(tensor.size()[0], 1, tensor.size()[1], tensor.size()[2])

# Trasnform & resize numpy arrays ==> cuda tensors, so they can run on GPU
# than create DataLoaders for training & validation
training_images_tensor, training_labels_tensor = transform_numpy_to_tensors(training_images, training_labels)
training_images_tensor = resize_tensor(training_images_tensor)
training_images_tensor = training_images_tensor.type(torch.cuda.FloatTensor)
training_labels_tensor = training_labels_tensor.type(torch.cuda.LongTensor)
training_data_set = TensorDataset(training_images_tensor, training_labels_tensor)
training_loader = DataLoader(training_data_set, batch_size=64)

validation_images_tensor, validation_labels_tensor = transform_numpy_to_tensors(validation_images, validation_labels)
validation_images_tensor = resize_tensor(validation_images_tensor)
validation_images_tensor = validation_images_tensor.type(torch.cuda.FloatTensor)
validation_labels_tensor = validation_labels_tensor.type(torch.cuda.LongTensor)
validation_data_set = TensorDataset(validation_images_tensor, validation_labels_tensor)
validation_loader = DataLoader(validation_data_set, batch_size=64)

# Train the CNN
device = "cuda" if torch.cuda.is_available() else "cpu"
model = RESNET18.ResNet_18(1, 2)
model = model.to(device)
# Print the summary of the arhitecture, for debugging
summary(model, input_size=(1, 128, 128))
# Choose the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
CNN_training_and_testing.train_CNN(model, training_loader, validation_loader, optimizer,num_epochs=20, device=device)

# Validate the CNN
validation_labels_predicted = CNN_training_and_testing.test_CNN(model, validation_loader, device)
analyzer.generate_metrics(validation_labels, np.asarray(validation_labels_predicted))

# Transform testing images & labels to tensor and
# create a DataLoader for them
testing_images_tensor = Tensor(testing_images)
testing_images_tensor = resize_tensor(testing_images_tensor)
testing_images_tensor = testing_images_tensor.type(torch.cuda.FloatTensor)

pseud_testing_labels = torch.randn(testing_images_tensor.size())
testing_data_set = TensorDataset(testing_images_tensor, pseud_testing_labels)
testing_loader = DataLoader(testing_data_set, batch_size=64)
testing_labels_predicted = []

# For each batch of images and labels, construct 
# the array of predictions 
for image_batch, labels_batch in testing_loader:
    image_batch = image_batch.to(device)
    
    pred = model(image_batch)
    predictions = torch.sigmoid(pred)
    predicted_labels = (predictions > 0.5).int()

    for i in predicted_labels:
        testing_labels_predicted.append(torch.argmax(i).item())

ImagePrepocessing.write_submission(testing_labels_predicted)
