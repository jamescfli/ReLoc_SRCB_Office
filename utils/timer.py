__author__ = 'bsl'

import time

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.name:
            print '[%s] ' % self.name
        print 'Elapsed: %s' % (time.time() - self.tstart)

# # usage
# import
# with Timer('name of timer'):
#     # things to be timed here