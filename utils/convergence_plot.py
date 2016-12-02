__author__ = 'bsl'

import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import os


class ConvergencePlot():
    def __init__(self, filename=None,
                 column_list = ['epoch', 'train_loss', 'train_acc', 'valid_loss', 'valid_acc']):
        filetype = filename.split('.')[-1]
        if filetype == 'csv':
            epoch_convergence_array = np.loadtxt(filename, dtype='float32', delimiter=',')
            self.epoch_convergence_df = pd.DataFrame(epoch_convergence_array, columns=column_list)
        else:   # txt file will not be supported
            print 'file name does not end with .csv'

    def plt_train_valid_loss(self, save=False):
        if 'train_loss' in self.epoch_convergence_df.keys()\
                and 'valid_loss' in self.epoch_convergence_df.keys():
            # draw series with epoch limitation
            plt.figure()
            plt.plot(self.epoch_convergence_df['epoch'], self.epoch_convergence_df['train_loss'])
            plt.plot(self.epoch_convergence_df['epoch'], self.epoch_convergence_df['valid_loss'])
            plt.xlim([self.epoch_convergence_df['epoch'][0], self.epoch_convergence_df['epoch'].__len__()])
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
        else:
            print 'there is no column named as train_loss or valid_loss ..'

    def plt_train_valid_acc(self, save=False):
        if 'train_acc' in self.epoch_convergence_df.keys()\
                and 'valid_acc' in self.epoch_convergence_df.keys():
            plt.figure()
            plt.plot(self.epoch_convergence_df['epoch'], self.epoch_convergence_df['train_acc'])
            plt.plot(self.epoch_convergence_df['epoch'], self.epoch_convergence_df['valid_acc'])
            plt.xlim([self.epoch_convergence_df['epoch'][0], self.epoch_convergence_df['epoch'][-1]])
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
        else:
            print 'there is no column named as train_acc or valid_acc ..'


if __name__ == '__main__':
    path_name = 'relocation/training_procedure/'
    file_name_smallset_3fc = 'convergence_vggrr3fc1024_largeset_15fzlayer_ls100_40epoch_sgdlr1e-05m1_reloc_model.csv'
    columns = ['epoch', 'train_loss', 'valid_loss']
    test_plot_smallset_3fc = ConvergencePlot(filename=path_name + file_name_smallset_3fc,
                                             column_list=columns)
    test_plot_smallset_3fc.plt_train_valid_loss()
    test_plot_smallset_3fc.plt_train_valid_acc()
