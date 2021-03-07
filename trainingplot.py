import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import keras
import numpy as np


class TrainingPlot(keras.callbacks.Callback):

    def __init__(self, filename='output/training_plot'):
        super().__init__()
        self.filename = filename
        self.epochs = 0

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        self.logs = []
        self.val_acc = []
        self.val_losses = []
        self.acc = []
        self.losses = []
        self.epochs = 0

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        self.epochs = epoch

        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:
            N = np.arange(0, len(self.losses))

            # You can chose the style of your preference
            plt.style.use("seaborn")

            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.plot(N, self.losses, label="train_loss")
            plt.plot(N, self.acc, label="train_acc")
            plt.plot(N, self.val_losses, label="val_loss")
            plt.plot(N, self.val_acc, label="val_acc")
            plt.title("Training Loss and Accuracy Epoch {}".format(epoch))
            plt.xlabel("Epoch # {}".format(epoch))
            plt.ylabel("Loss/Accuracy")
            plt.savefig(self.filename + ".jpg")
            plt.legend()
            plt.close()

    # Return the current epoch value
    def get_epoch_number(self):
        return self.epochs
