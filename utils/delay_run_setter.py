__author__ = 'bsl'

import time


def sleeper(time_len_in_secs=None):
    if time_len_in_secs is None:
        time_len_in_secs = raw_input('How long to wait: ')

    try:
        time_len_in_secs = float(time_len_in_secs)
    except ValueError:
        print('Pls input a number.\n')

    print('Before: %s' % time.ctime())
    time.sleep(time_len_in_secs)
    print('After: %s\n' % time.ctime())

if __name__ == '__main__':
    try:
        sleeper(10)
    except KeyboardInterrupt:
        print('\nKeyboard exception received. Exiting.')
        exit()