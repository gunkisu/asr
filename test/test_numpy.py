from __future__ import print_function
import unittest
import numpy

from theano import tensor as T
from theano import function

class NumpyTest(unittest.TestCase):
    def setUp(self):
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

    def tearDown(self):
        pass

    def testTakeFirstInSeq(self):
        
        answer = numpy.array([
                [1.,1.],
                [2.,2.],
                [3.,3.],
                [4.,4.]])

        input = numpy.transpose(self.input, (1,0,2))
        
        out = input[0]
        self.assertTrue(numpy.allclose(answer, out))
        

if __name__ == '__main__':
    unittest.main()



