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
        new_seq = utils.skip_frames(self.a_seq, every_n, False)
        print(new_seq)  

    def testSkipFamesRandom(self):
        every_n = 3
        print(self.a_seq, self.a_seq.shape)
        new_seq = utils.skip_frames(self.a_seq, every_n, True)
        print(new_seq)  

    def testRepeat(self):
        every_n = 3
        print(self.a_seq, self.a_seq.shape)
        new_seq = utils.skip_frames(self.a_seq, every_n, False)
        print(new_seq)
        a_new_seq = new_seq[0][0]
        expanded_seq = np.repeat(a_new_seq, every_n, axis=0)
        orig_len = len(self.a_seq[0][0])
        print(expanded_seq[:orig_len,:])
      


if __name__ == '__main__':
    unittest.main()



