__author__ = 'bsl'

from keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


class LossAccRTPlot(Callback):
    def on_train_begin(self, logs={}):
        self.nb_epoch = 0
        self.train_losses = []
        self.train_accs = []
        self.valid_losses = []
        self.valid_accs = []
        self.figure = plt.figure(1)
        self.subplot_loss = self.figure.add_subplot(121)
        self.subplot_loss.set_title('loss function')
        self.subplot_loss.set_ylim([0, 1])
        self.line_train_loss, = self.subplot_loss.plot(np.arange(1,self.nb_epoch+1), self.train_losses)
        self.line_valid_loss, = self.subplot_loss.plot(np.arange(1,self.nb_epoch+1), self.valid_losses)
        self.subplot_acc = self.figure.add_subplot(122)
        self.subplot_acc.set_title('accuracy')
        self.subplot_acc.set_xlim([1, 10])
        self.subplot_acc.set_ylim([0.5, 1])
        self.line_train_acc, = self.subplot_acc.plot(np.arange(1, self.nb_epoch + 1), self.train_accs)
        self.line_valid_acc, = self.subplot_acc.plot(np.arange(1, self.nb_epoch + 1), self.valid_accs)
        self.figure.show()

    def on_epoch_end(self, epoch, logs={}):
        self.nb_epoch += 1
        self.train_losses.append(logs.get('loss'))
        self.train_accs.append(logs.get('acc'))
        self.valid_losses.append(logs.get('val_loss'))
        self.valid_accs.append(logs.get('val_acc'))
        self.update_loss_plot()
        self.update_acc_plot()

    # if need to leave the figure in the air
    def on_train_end(self, logs={}):
        raw_input()

    def update_loss_plot(self):
        self.line_train_loss.set_xdata(np.append(self.line_train_loss.get_xdata(), self.nb_epoch))
        self.line_train_loss.set_ydata(self.train_losses)
        self.line_valid_loss.set_xdata(np.append(self.line_valid_loss.get_xdata(), self.nb_epoch))
        self.line_valid_loss.set_ydata(self.valid_losses)
        self.subplot_loss.set_xlim([1, np.ceil(self.nb_epoch/10.0)*10])  # extend x axis if necessary
        self.figure.canvas.draw()

    def update_acc_plot(self):
        self.line_train_acc.set_xdata(np.append(self.line_train_acc.get_xdata(), self.nb_epoch))
        self.line_train_acc.set_ydata(self.train_accs)
        self.line_valid_acc.set_xdata(np.append(self.line_valid_acc.get_xdata(), self.nb_epoch))
        self.line_valid_acc.set_ydata(self.valid_accs)
        self.subplot_acc.set_xlim([1, np.ceil(self.nb_epoch / 10.0) * 10])
        self.figure.canvas.draw()


class LossRTPlot(Callback):
    def on_train_begin(self, logs={}):
        self.nb_epoch = 0
        self.train_losses = []
        self.valid_losses = []
        self.figure = plt.figure(1)
        self.subplot_loss = self.figure.add_subplot(111)
        self.subplot_loss.set_title('loss function')
        self.subplot_loss.set_ylim([0, 1])
        self.line_train_loss, = self.subplot_loss.plot(np.arange(1,self.nb_epoch+1), self.train_losses)
        self.line_valid_loss, = self.subplot_loss.plot(np.arange(1,self.nb_epoch+1), self.valid_losses)
        self.figure.show()

    def on_epoch_end(self, epoch, logs={}):
        self.nb_epoch += 1
        self.train_losses.append(logs.get('loss'))
        self.valid_losses.append(logs.get('val_loss'))
        self.update_loss_plot()

    # # if need to leave the figure in the air
    # def on_train_end(self, logs={}):
    #     raw_input()

    def update_loss_plot(self):
        self.line_train_loss.set_xdata(np.append(self.line_train_loss.get_xdata(), self.nb_epoch))
        self.line_train_loss.set_ydata(self.train_losses)
        self.line_valid_loss.set_xdata(np.append(self.line_valid_loss.get_xdata(), self.nb_epoch))
        self.line_valid_loss.set_ydata(self.valid_losses)
        self.subplot_loss.set_xlim([1, np.ceil(self.nb_epoch/10.0)*10])  # extend x axis if necessary
        self.figure.canvas.draw()


if __name__ == '__main__':
    # for a single-input model with 2 classes (binary):

    model = Sequential()
    model.add(Dense(1, input_dim=784, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    data = np.random.random((100000, 784))
    labels = np.random.randint(2, size=(100000, 1))

    loss_acc_rtplot = LossAccRTPlot()
    model.fit(data, labels, nb_epoch=10, batch_size=32, callbacks=[loss_acc_rtplot])