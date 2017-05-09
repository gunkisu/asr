from __future__ import print_function
import unittest
import numpy as np
from libs import utils

import pprint

class UtilsTest(unittest.TestCase):
    def setUp(self):
        self.a_seq = np.array([ [[[i, i+1] for i in range(1,10,2)]]])

        self.inp = np.arange(16).reshape((2,4,2))
        
        self.inp_mask = np.array([[0, 0, 1, 1], 
                                    [1, 0, 0, 1]])

        self.inp_pad_mask = np.array([[1, 1, 1, 0], 
                                        [1, 1, 1, 1]])

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
      
#    def testCompressBatch(self):
#        compressed = utils.compress_batch(self.inp, self.inp_mask)
#        answer = [np.array([[0,1], [4,5], [6,7]]), np.array([[8,9], [14,15]])]
#        self.assertTrue(all([np.allclose(x, y) for x, y in zip(compressed, answer)]))

    def testCompressBatch(self):
        compressed = utils.compress_batch(self.inp, self.inp_mask)
#        padded = utils.pad_batch(compressed)
        answer = [np.array([[0,1], [4,5], [6,7]]), np.array([[8,9], [14,15], [0, 0]])]
        self.assertTrue(all([np.allclose(x, y) for x, y in zip(compressed, answer)]))

    def test_seg_len_info(self):
        lens = utils.seg_len_info(self.inp_mask)
        answer = [[2,1,1], [3,1]]
        self.assertEqual(lens, answer)
   

    def test_uncompress_batch(self):
        compressed = utils.compress_batch(self.inp, self.inp_mask)
        lens = utils.seg_len_info(self.inp_mask)

        answer = np.array([[[0, 1], [0,1], [4,5], [6,7]],
            [[8,9],[8,9],[8,9],[14,15]]])
        
        self.assertTrue(np.allclose(utils.uncompress_batch(compressed, lens), answer))


if __name__ == '__main__':
    unittest.main()



