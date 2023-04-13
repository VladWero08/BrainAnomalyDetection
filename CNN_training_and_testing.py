import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def train_CNN(model, train_dataloader, val_dataloader, optimizer, num_epochs=10, device="cpu", loss_function=nn.BCEWithLogitsLoss()):
    model.train(True)

    # Array that will store information about every epoch
    train_loss_per_epoch = []
    train_accuracy_per_epoch = []
    validation_loss_per_epoch = []
    validation_accuracy_per_epoch = []    
        
    # Loop through each epoch    
    for i in range(num_epochs):
        print(f"--> Epoch {i + 1}:")
        
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        # Loop through batches
        for batch, (image_batch, labels_batch) in enumerate(train_dataloader):
            optimizer.zero_grad()
            image_batch = image_batch.to(device)
            labels_batch = labels_batch.to(device)

            # Labels will be of shape (batch_size), so they need to be
            # transformed to (batch_size, 2) in order for the loss function to be calculated
            pred = model(image_batch)
            labels_batch = F.one_hot(labels_batch, num_classes=2)
            loss = loss_function(pred, labels_batch.float())

            # Transform into actual predictions with sigmoid
            predictions = torch.sigmoid(pred)
            predicted_labels = (predictions > 0.5).int()

            # Backpropagation
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * image_batch.size(0)
            epoch_accuracy += (predicted_labels == labels_batch.int()).all(dim=1).sum().item()
            if batch % 100 == 0:
                loss = loss.item()
                print(f"-->--> Batch index {batch}, loss: {loss:>7f}")
        
        train_loss_per_epoch.append(epoch_loss / len(train_dataloader.dataset))
        train_accuracy_per_epoch.append(epoch_accuracy / len(train_dataloader.dataset))

        # Calculate the loss and accuracy of validation data as well        
        val_epoch_loss = 0.0
        val_epoch_accuracy = 0.0
        with torch.no_grad():
            for image_batch, labels_batch in val_dataloader:
                image_batch = image_batch.to(device)
                labels_batch = labels_batch.to(device)

                pred = model(image_batch)
                predictions = torch.sigmoid(pred)
                predicted_labels = (predictions > 0.5).int()
                
                labels_batch = F.one_hot(labels_batch, num_classes=2)
                loss = loss_function(pred, labels_batch.float())

                val_epoch_loss += loss.item() * image_batch.size(0)
                val_epoch_accuracy += (predicted_labels == labels_batch.int()).all(dim=1).sum().item()
        
        validation_loss_per_epoch.append(val_epoch_loss / len(val_dataloader.dataset))
        validation_accuracy_per_epoch.append(val_epoch_accuracy / len(val_dataloader.dataset))
        
    return train_loss_per_epoch, validation_loss_per_epoch, train_accuracy_per_epoch, validation_accuracy_per_epoch

def test_CNN(model, test_dataloader, device, loss_function=nn.BCEWithLogitsLoss()):
    testing_predictions = []
    test_loss = 0.
    size = len(test_dataloader.dataset)
    model.eval()
    
    with torch.no_grad():
        for image_batch, labels_batch in test_dataloader:
            image_batch = image_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            pred = model(image_batch)
            labels_batch = F.one_hot(labels_batch, num_classes=2)
            loss = loss_function(pred, labels_batch.float())
            
            # Apply sigmoid so the prediction can be interpreted properly
            predictions = torch.sigmoid(pred)
            predicted_labels = (predictions > 0.5).int()
            
            # From the (batch_size, 2) tensor, get the prediction
            # by finding the index of the maxixmum value for each 'batch_size'
            for i in predicted_labels:
                testing_predictions.append(torch.argmax(i).item())
            
            test_loss += loss.item()
            print(f"-->--> Batch testing loss: {loss.item()}")

    test_loss /= size
    return testing_predictions