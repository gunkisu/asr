from __future__ import print_function
import unittest
import numpy as np
from libs import utils

class UtilsTest(unittest.TestCase):
    def setUp(self):
        self.a_seq = np.array([ [[[i, i+1] for i in range(1,10,2)]]])

    def tearDown(self):
        pass

    def testSkipFames(self):
        every_n = 3
        print(self.a_seq, self.a_seq.shape)
        new_seq = utils.skip_frames(self.a_seq, every_n, True)
        print(new_seq)  

if __name__ == '__main__':
    unittest.main()



