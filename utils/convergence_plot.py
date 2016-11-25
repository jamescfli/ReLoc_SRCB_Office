__author__ = 'bsl'

import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import os


class ConvergencePlot():
    def __init__(self, filename=None, nb_line_skipped=0):
        filetype = filename.split('.')[-1]
        if filetype == 'csv':
            with open(filename, 'r') as epoch_file:
                lines = epoch_file.readlines()[nb_line_skipped:]
            self.nb_total_epoch = lines.__len__()
            epoch_convergence_array = np.zeros((self.nb_total_epoch, 4), dtype='float32')
            for index, line in enumerate(lines):
                items = line.split(',')
                # fill the rest with 0's if items length < 5 (1 index + 4 values)
                item_array = np.append(np.array(items[1:]),     # cut epoch index off
                                       np.zeros(4-items.__len__()+1, dtype='float32'))
                epoch_convergence_array[index, :] = item_array
            column_list = ['train_loss', 'train_acc', 'valid_loss', 'valid_acc']
            self.epoch_convergence_df = pd.DataFrame(epoch_convergence_array, columns=column_list)
        else:   # txt file will not be supported
            print 'wrong file name extension'

    def plt_train_valid_loss(self, nb_epoch=None, save=False):
        if nb_epoch == None:
            # draw full length time series, i.e. self.nb_total_epoch
            nb_epoch = self.nb_total_epoch
        assert nb_epoch <= self.nb_total_epoch, 'set nb_epoch exceeds nb_total_epoch = {}'.format(self.nb_total_epoch)
        # draw series with epoch limitation
        plt.figure()
        plt.plot(np.arange(nb_epoch)+1, self.epoch_convergence_df['train_loss'][0:nb_epoch])
        plt.plot(np.arange(nb_epoch)+1, self.epoch_convergence_df['valid_loss'][0:nb_epoch])
        plt.xlim([1, nb_epoch+1])
        plt.xlabel('epoch')
        plt.ylabel('loss function')
        plt.legend(['train_loss', 'valid_loss'], loc='upper right')
        if save:
            if not os.path.exists("./convergence_figures/"):
                os.makedirs("./convergence_figures")
            plt.savefig("./convergence_figures/loss_function.png", format='png', dpi=200)
        else:
            plt.show()
            raw_input()

    def plt_train_valid_acc(self, nb_epoch=None, save=False):
        if nb_epoch == None:
            # draw full length time series, i.e. self.nb_total_epoch
            nb_epoch = self.nb_total_epoch
        assert nb_epoch <= self.nb_total_epoch, 'set nb_epoch exceeds nb_total_epoch = {}'.format(self.nb_total_epoch)
        # draw series with epoch limitation
        plt.figure()
        plt.plot(np.arange(nb_epoch)+1, self.epoch_convergence_df['train_acc'][0:nb_epoch])
        plt.plot(np.arange(nb_epoch)+1, self.epoch_convergence_df['valid_acc'][0:nb_epoch])
        plt.xlim([1, nb_epoch+1])
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['train_acc', 'valid_acc'], loc='lower right')
        if save:
            if not os.path.exists("./convergence_figures/"):
                os.makedirs("./convergence_figures")
            plt.savefig("./convergence_figures/accuracy.png", format='png', dpi=200)
        else:
            plt.show()
            raw_input()

    def plt_time_consumption(self, nb_epoch=None, save=False):
        if nb_epoch == None:
            # draw full length time series, i.e. self.nb_total_epoch
            nb_epoch = self.nb_total_epoch
        assert nb_epoch <= self.nb_total_epoch, 'set nb_epoch exceeds nb_total_epoch = {}'.format(self.nb_total_epoch)
        # draw series with epoch limitation
        plt.figure()
        plt.plot(np.arange(nb_epoch)+1, self.epoch_convergence_df['time_in_sec'][0:nb_epoch])
        plt.xlim([1, nb_epoch+1])
        plt.xlabel('epoch')
        plt.ylabel('time consumption (sec)')
        plt.legend(['time_in_sec'], loc='upper right')
        if save:
            if not os.path.exists("./convergence_figures/"):
                os.makedirs("./convergence_figures")
            plt.savefig("./convergence_figures/time_consumption.png", format='png', dpi=200)
        else:
            plt.show()
            raw_input()

if __name__ == '__main__':
    path_name = 'relocation/training_procedure/'
    file_name_smallset_2fc = 'convergence_smallset_vgg2fc1024_places_60epoch_sgdlr1e-05m50_reloc_model.csv'
    test_plot_smallset_2fc = ConvergencePlot(filename=path_name + file_name_smallset_2fc)
    file_name_smallset_3fc = 'convergence_smallset_vgg3fc1024_places_100epoch_sgdlr1e-05m50_reloc_model.csv'
    test_plot_smallset_3fc = ConvergencePlot(filename=path_name + file_name_smallset_3fc)
    plt.figure()
    plt.plot(np.arange(test_plot_smallset_2fc.nb_total_epoch) + 1, test_plot_smallset_2fc.epoch_convergence_df['train_loss'])
    plt.plot(np.arange(test_plot_smallset_3fc.nb_total_epoch) + 1, test_plot_smallset_3fc.epoch_convergence_df['train_loss'])
    plt.xlim([1, max(test_plot_smallset_2fc.nb_total_epoch, test_plot_smallset_3fc.nb_total_epoch) + 1])
    plt.xlabel('epoch')
    plt.ylabel('train loss')
    plt.legend(['2fc w Dropout 0.5', '3fc w Dropout 0.5'], loc='upper right')
    plt.title('relocation for smallset 5000 images')
    plt.show()
    raw_input()