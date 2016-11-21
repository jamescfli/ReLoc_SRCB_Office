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
                epoch_convergence_array[index, :] = np.array(items[1:])     # discard epoch index
            column_list = ['train_loss', 'train_acc', 'valid_loss', 'valid_acc']
            self.epoch_convergence_df = pd.DataFrame(epoch_convergence_array, columns=column_list)
        elif filetype == 'txt':
            with open(filename, 'r') as epoch_file:
                lines = epoch_file.readlines()[nb_line_skipped:]
                lines = lines[1::2]  # skip line intermittently
            self.nb_total_epoch = lines.__len__()
            epoch_convergence_array = np.zeros((self.nb_total_epoch, 5), dtype='float32')
            for index, line in enumerate(lines):
                items = line.split(' ')
                select_items = items[3::3]
                select_items[0] = select_items[0].rstrip(string.ascii_letters)  # delete 'sec'
                select_items[4] = select_items[4].rstrip(string.whitespace)  # delete '\n'
                select_items_in_float = [float(item) for item in select_items]
                epoch_convergence_array[index, :] = select_items_in_float
            column_list = ['time_in_sec', 'train_loss', 'train_acc', 'valid_loss', 'valid_acc']
            self.epoch_convergence_df = pd.DataFrame(epoch_convergence_array, columns=column_list)
        else:
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
    # path_name = 'pretrain/training_procedure/'
    # file_name_1e3 = 'convergence_vgg2fc512_places_fineT_11fzlayer_25epoch_sgdlr1e-3_2class_HomeOrOff_model.csv'
    # file_name_1e4 = 'convergence_vgg2fc512_places_fineT_11fzlayer_50epoch_sgdlr1e-4_2class_HomeOrOff_model.csv'
    # file_name_1e5 = 'convergence_vgg2fc512_places_fineT_11fzlayer_50epoch_sgdlr1e-5_2class_HomeOrOff_model.csv'
    # file_name_1e6 = 'convergence_vgg2fc512_places_fineT_11fzlayer_100epoch_sgdlr1e-6_2class_HomeOrOff_model.csv'
    #
    # test_plot_1e3 = ConvergencePlot(filename=path_name+file_name_1e3)
    # test_plot_1e4 = ConvergencePlot(filename=path_name+file_name_1e4)
    # test_plot_1e5 = ConvergencePlot(filename=path_name+file_name_1e5)
    # test_plot_1e6 = ConvergencePlot(filename=path_name+file_name_1e6)

    # plt.figure()
    # plt.plot(np.arange(test_plot_1e3.nb_total_epoch)+1, test_plot_1e3.epoch_convergence_df['train_loss'])
    # plt.plot(np.arange(test_plot_1e4.nb_total_epoch)+1, test_plot_1e4.epoch_convergence_df['train_loss'])
    # plt.plot(np.arange(test_plot_1e5.nb_total_epoch)+1, test_plot_1e5.epoch_convergence_df['train_loss'])
    # plt.plot(np.arange(test_plot_1e6.nb_total_epoch)+1, test_plot_1e6.epoch_convergence_df['train_loss'])
    # plt.xlim([1, 100])
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.legend(['1e-3', '1e-4', '1e-5', '1e-6'], loc='upper right')
    # plt.show()
    # raw_input()

    # plt.figure()
    # plt.plot(np.arange(test_plot_1e5.nb_total_epoch)+1, test_plot_1e5.epoch_convergence_df['train_acc'])
    # plt.plot(np.arange(test_plot_1e5.nb_total_epoch)+1, test_plot_1e5.epoch_convergence_df['valid_acc'])
    # plt.xlim([1, test_plot_1e5.nb_total_epoch+1])
    # plt.xlabel('epoch')
    # plt.ylabel('acc')
    # plt.legend(['train_acc', 'valid_acc'], loc='upper left')
    # plt.title('accuracy lr=1e-5')
    # plt.show()
    # raw_input()

    # plt.figure()
    # plt.plot(np.arange(test_plot_1e6.nb_total_epoch) + 1, test_plot_1e6.epoch_convergence_df['train_loss'])
    # plt.plot(np.arange(test_plot_1e6.nb_total_epoch) + 1, test_plot_1e6.epoch_convergence_df['valid_loss'])
    # plt.xlim([1, test_plot_1e6.nb_total_epoch + 1])
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.legend(['train_loss', 'valid_loss'], loc='upper right')
    # plt.title('loss function with lr=1e-6')
    # plt.show()
    # raw_input()

    path_name = 'pretrain/training_procedure/'
    file_name_1e5 = 'convergence_vgg2fc256_imagenet_fineT_11fzlayer_25epoch_sgdlr1e-5_2class_HomeOrOff_model.txt'
    test_plot_1e5 = ConvergencePlot(filename=path_name + file_name_1e5, nb_line_skipped=8)
    plt.figure()
    plt.plot(np.arange(test_plot_1e5.nb_total_epoch) + 1, test_plot_1e5.epoch_convergence_df['train_loss'])
    plt.plot(np.arange(test_plot_1e5.nb_total_epoch) + 1, test_plot_1e5.epoch_convergence_df['valid_loss'])
    plt.xlim([1, test_plot_1e5.nb_total_epoch + 1])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train_loss', 'valid_loss'], loc='upper right')
    plt.title('loss function with lr=1e-5')
    plt.show()
    raw_input()