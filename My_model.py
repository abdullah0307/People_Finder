import collections
import os

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import History
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential, load_model, save_model

from trainingplot import TrainingPlot


class AD_Model:

    def __init__(self):
        super().__init__()
        self.plot_losses = TrainingPlot()
        self.model = Sequential()
        self.faces = []
        self.Ids = []
        self.history = History()

    def load_model(self, path):
        if os.path.exists(path):
            self.model = load_model(filepath=path)
        else:
            print("model loading failed")

    def save_model(self, path):
        save_model(model=self.model, filepath=path)

    def get_epoch(self):
        return self.plot_losses.get_epoch_number()

    def train(self, faces, Ids, epocs=100):
        self.faces = faces
        self.Ids = Ids

        img_size = 100
        output_cells = len([item for item, count in collections.Counter(Ids).items() if count > 1])

        data = np.array(faces) / 255.0
        data = np.reshape(data, (data.shape[0], img_size, img_size, 1))
        target = np.array(Ids)

        from keras.utils import np_utils
        new_target = np_utils.to_categorical(target)

        self.model.add(Conv2D(100, (3, 3), input_shape=data.shape[1:]))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # The first CNN layer followed by Relu and MaxPooling layers

        self.model.add(Conv2D(100, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # The Second convolution layer followed by Relu and MaxPooling layers

        self.model.add(Conv2D(100, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # The third convolution layer followed by Relu and MaxPooling layers

        self.model.add(Conv2D(100, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # The fourth convolution layer followed by Relu and MaxPooling layers

        self.model.add(Flatten())
        # Flatten layer to stack the output convolutions from second convolution layer

        self.model.add(Dense(50, activation='relu'))
        # Dense layer of 64 neurons

        self.model.add(Dense(int(output_cells), activation='softmax'))
        # The Final layer with two outputs for two categories

        opt = keras.optimizers.Adam(lr=0.001)

        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        history = self.model.fit(data, new_target, epochs=int(epocs), callbacks=[self.plot_losses],
                                 validation_split=0.1, batch_size=16)
        self.history = history

    def predict(self, face):
        normalized = face / 255
        reshaped = np.reshape(normalized, (1, 100, 100, 1))
        conf = int(max(self.model.predict(reshaped)[0])*100)
        label = self.model.predict_classes(reshaped)[0]
        return label, conf

    def show_result(self):
        plt.plot(self.history.history['acc'], 'r', label='training accuracy')
        plt.plot(self.history.history['val_acc'], 'b', label='validation accuracy')
        plt.plot(self.history.history['loss'], 'g', label='training loss')
        plt.plot(self.history.history['val_loss'], 'y', label='validation loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()
