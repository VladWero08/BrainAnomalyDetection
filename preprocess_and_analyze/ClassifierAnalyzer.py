from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

def generate_metrics(labels, labels_predicted):
    print("Accuracy: ", accuracy_score(y_true=labels, y_pred=labels_predicted))
    print("Precision: ", precision_score(y_true=labels, y_pred=labels_predicted))
    print("Recall: ", recall_score(y_true=labels, y_pred=labels_predicted))
    print("F1 score: ", f1_score(y_true=labels, y_pred=labels_predicted))
    print("Confusion matrix: ")
    print(confusion_matrix(y_true=labels, y_pred=labels_predicted))

def generate_loss_and_accuracy(history):
    # Firstly compute the grid layer for the plots
    figure, plots = plt.subplots(1, 2, figsize=(10,5))

    # Compute the loss
    train_loss = history.history['loss']
    validation_loss = history.history['val_loss']
    plots[0].plot(train_loss, 'g', label="Training loss")
    plots[0].plot(validation_loss, 'b', label="Validation loss")
    plots[0].set_title('Loss comparison')
    plots[0].set_xlabel('Epochs')
    plots[0].set_ylabel('Loss')
    plots[0].set_ylim([0,1])
    plots[0].legend()

    # Compute the accuracy
    train_accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']
    plots[1].plot(train_accuracy, 'g', label="Training accuracy")
    plots[1].plot(validation_accuracy, 'b', label="Validation accuracy")
    plots[1].set_title('Accuracy comparison')
    plots[1].set_xlabel('Epochs')
    plots[1].set_ylabel('Accuracy')
    plots[1].set_ylim([0,1])
    plots[1].legend()

    plt.show()

