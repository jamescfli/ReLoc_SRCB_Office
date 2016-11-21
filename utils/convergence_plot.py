import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import os


class ConvergencePlot():
    def __init__(self, filename=None, nb_line_skipped = 0):
        with open(filename, 'r') as epoch_file:
            lines = epoch_file.readlines()[nb_line_skipped:]
            lines = lines[1::2]     # skip line intermittently
        self.nb_total_epoch = lines.__len__()
        epoch_convergence_array = np.zeros((self.nb_total_epoch, 5), dtype='float32')
        for index, line in enumerate(lines):
            items = line.split(' ')
            select_items = items[3::3]
            select_items[0] = select_items[0].rstrip(string.ascii_letters)  # delete 'sec'
            select_items[4] = select_items[4].rstrip(string.whitespace)     # delete '\n'
            select_items_in_float = [float(item) for item in select_items]
            epoch_convergence_array[index, :] = select_items_in_float
        column_list = ['time_in_sec', 'train_loss', 'train_acc', 'valid_loss', 'valid_acc']
        self.epoch_convergence_df = pd.DataFrame(epoch_convergence_array, columns=column_list)

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
    filename_1e_3 = 'pretrain/training_procedure/train_vgg2fc_11fzlayer_125epoch_lr1e-3_2class_HomeOrOff_model.txt'
    filename_1e_4 = 'pretrain/training_procedure/train_vgg2fc_11fzlayer_25epcoh_lr1e-4_2class_HomeOrOff_model.txt'
    filename_1e_5 = 'pretrain/training_procedure/train_vgg2fc_11fzlayer_25epoch_lr1e-5_2class_HomeOrOff_model.txt'

    nb_line_skipped = 8

    test_plot_1e_3 = ConvergencePlot(filename=filename_1e_3, nb_line_skipped=nb_line_skipped)
    test_plot_1e_4 = ConvergencePlot(filename=filename_1e_4, nb_line_skipped=nb_line_skipped)
    test_plot_1e_5 = ConvergencePlot(filename=filename_1e_5, nb_line_skipped=nb_line_skipped)

    plt.figure()
    nb_epoch = 25
    plt.plot(np.arange(nb_epoch)+1, test_plot_1e_3.epoch_convergence_df['valid_loss'][0:nb_epoch])
    plt.plot(np.arange(nb_epoch)+1, test_plot_1e_4.epoch_convergence_df['valid_loss'][0:nb_epoch])
    plt.plot(np.arange(nb_epoch)+1, test_plot_1e_5.epoch_convergence_df['valid_loss'][0:nb_epoch])
    plt.xlim([1, nb_epoch])
    plt.xlabel('epoch')
    plt.ylabel('valid loss')
    plt.legend(['1e-3', '1e-4', '1e-5'], loc='upper left')
    plt.show()

    # test_plot = ConvergencePlot(filename=filename, nb_line_skipped=nb_line_skipped)
    # test_plot.plt_train_valid_loss()
    # test_plot.plt_train_valid_acc()
    # test_plot.plt_time_consumption()
