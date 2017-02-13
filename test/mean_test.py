from __future__ import print_function
import unittest
import numpy

from theano import tensor as T
from theano import function

class MeanTest(unittest.TestCase):
    def setUp(self):
        x = T.dtensor3()
        
        # n_seq, n_batch, n_dim
        x1 = x.dimshuffle(1,0,2)
        mean = T.mean(x1, axis=0)
        
        self.out_fn = function([x], mean)

    def tearDown(self):
        pass

    def testMean(self):
        # n_batch, n_seq, n_dim
        a = numpy.array(
            [
                [[1.,1.], [1.,1.], [1., 1.]],
                [[2.,2.], [2.,2.], [2., 2.]],
                [[3.,3.], [3.,3.], [3., 3.]],
                [[4.,4.], [4.,4.], [4., 4.]]
            ])
        
        result = numpy.array(
            [[1.,1.], [2.,2.], [3.,3.], [4.,4.]])
        out = self.out_fn(a)

        self.assertTrue(numpy.allclose(result, out))
        
if __name__ == '__main__':
    unittest.main()



