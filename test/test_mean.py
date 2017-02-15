from __future__ import print_function
import unittest
import numpy

from theano import tensor as T
from theano import function

class MeanTest(unittest.TestCase):
    def setUp(self):
        self.x = T.dtensor3()
        self.mask = T.dmatrix()
        
        # n_seq, n_batch, n_dim
        self.x1 = self.x.dimshuffle(1,0,2)
        self.mask1 = self.mask.dimshuffle(1,0)

        # n_batch, n_seq, n_dim
        self.input = numpy.array(
            [
                [[1.,1.], [1.,1.], [1., 1.]],
                [[2.,2.], [2.,2.], [2., 2.]],
                [[3.,3.], [3.,3.], [3., 3.]],
                [[4.,4.], [4.,4.], [4., 4.]]
            ])

        # n_batch, n_seq, n_dim
        self.input2 = numpy.array(
            [
                [[1.,1.], [1.,1.], [1., 1.],[0.,0.]],
                [[2.,2.], [2.,2.], [2., 2.],[2.,2.]],
                [[3.,3.], [3.,3.], [3., 3.],[0.,0.]],
                [[4.,4.], [4.,4.], [4., 4.],[0.,0.]]
            ])

        self.input2_mask = numpy.array(
            [   
                [1, 1, 1, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 0],
                [1, 1, 1, 0]
                ])

    def tearDown(self):
        pass

    def testMean(self):
        out_fn = self.mean_fn()
        
        result = numpy.array(
            [[1.,1.], [2.,2.], [3.,3.], [4.,4.]])
        out = out_fn(self.input)

        self.assertTrue(numpy.allclose(result, out))

    def testMeanMask(self):
        fn1 = self.masked_mean_fn()
        out1 = fn1(self.input2, self.input2_mask)

        fn2 = self.mean_fn()
        out2 = fn2(self.input2)

        result = numpy.array(
            [[1.,1.], [2.,2.], [3.,3.], [4.,4.]])
      
        self.assertTrue(numpy.allclose(result, out1))
        self.assertFalse(numpy.allclose(result, out2))


    def mean_fn(self):
        return function([self.x], T.mean(self.x1, axis=0))

    def masked_mean_fn(self):
        
        seq_len = T.sum(self.mask1, axis=0)
        seq_len = seq_len[:,None]
        
        seq_sum = T.sum(self.x1, axis=0)


        mean = seq_sum/ seq_len

        return function([self.x, self.mask], mean)
       


if __name__ == '__main__':
    unittest.main()



