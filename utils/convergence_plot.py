import pandas as pd
import numpy as np
import string
# import matplotlib.pyplot as plt

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

    def plt_train_valid_loss(self):
        self.epoch_convergence_df.plot(x=np.arange(self.nb_total_epoch)+1, y=['train_loss', 'valid_loss'],
                                       kind='line', legend=True)

    def plt_train_valid_acc(self):
        self.epoch_convergence_df.plot(x=np.arange(self.nb_total_epoch)+1, y=['train_acc', 'valid_acc'],
                                       kind='line', legend=True)

    def plt_time_consumption(self):
        self.epoch_convergence_df.plot(x=np.arange(self.nb_total_epoch)+1, y=['time_in_sec'],
                                       kind='line', legend=True)

if __name__ == '__main__':
    filename = 'castrain_vgg_11-15-19fzlayer_10epoch_lr1e-05_2class_HomeOrOff_model.txt'
    nb_line_skipped = 6
    test_plot = ConvergencePlot(filename=filename, nb_line_skipped=nb_line_skipped)
    test_plot.plt_train_valid_loss()
    test_plot.plt_train_valid_acc()
    test_plot.plt_time_consumption()
