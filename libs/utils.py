from __future__ import print_function

import time

class StopWatch():
    def __init__(self):
        self.start_time = time.time() 

    def reset(self):
        self.start_time = time.time()

    def elapsed(self):
        return time.time() - self.start_time
    
    def print_elapsed(self):
        time_diff = self.elapsed()

        print('Took {:.1f}s ({:.1f}m; {:.1f}h)'.format(time_diff, time_diff / 60, time_diff / 3600))
