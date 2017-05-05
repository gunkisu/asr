from __future__ import print_function
import unittest
import numpy as np
from libs import utils

import pprint

class UtilsTest(unittest.TestCase):
    def setUp(self):
        self.a_seq = np.array([ [[[i, i+1] for i in range(1,10,2)]]])

    def tearDown(self):
        pass

    def testSkipFames(self):
        every_n = 3
        new_seq = utils.skip_frames(self.a_seq, every_n, False)
        answer = [[np.array([[1, 2], [7, 8]])]]
        self.assertTrue(np.allclose(new_seq[0][0], answer[0][0]))

    def testSkipFamesRandom(self):
        every_n = 3
        new_seq = utils.skip_frames(self.a_seq, every_n, True)
#        self.assertTrue(np.allclose(new_seq[0][0], answer[0][0]))

    def testRepeat(self):
        every_n = 3
        new_seq = utils.skip_frames(self.a_seq, every_n, False)
        a_new_seq = new_seq[0][0]
        expanded_seq = np.repeat(a_new_seq, every_n, axis=0)
        orig_len = len(self.a_seq[0][0])

        answer = np.array([[1, 2], [1, 2], [1, 2], [7, 8], [7, 8]])
        self.assertTrue(np.allclose(expanded_seq[:orig_len,:], answer))
      
    def testCompressBatch(self):
        inp = np.arange(16).reshape((2,4,2))
        inp_mask = np.array([[0, 0, 1, 1], [1, 0, 0, 1]])
        compressed = utils.compress_batch(inp, inp_mask)
        answer = [np.array([[0,1], [4,5], [6,7]]), np.array([[8,9], [14,15]])]
        self.assertTrue(all([np.allclose(x, y) for x, y in zip(compressed, answer)]))



if __name__ == '__main__':
    unittest.main()



