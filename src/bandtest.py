'''
Created on Jul 30, 2012

@author: nullas
'''
from __future__ import division

import unittest
import exceptions

from surfaces import Sphere as surface
from band import Band
from mpi4py import MPI


try:
    import petsc4py
    import os
    from petsc4py import PETSc
    petsc4py.init(os.sys.argv)
    from numpy import array as a
    import numpy.testing as npt
except Exception as exp:
    print exp

class TestBand(unittest.TestCase):


    def setUp(self):
        self.bnd = Band(surface,MPI.COMM_WORLD)


    def tearDown(self):
        pass


    def testBlockInd2SubWithoutBand(self):
        '''test BlockInd2SubWithoutBand'''
        example = ( (5, a([5,0,0]) ),
                    (21,a([1,1,0]) ),
                    (a([401,402]),a([[1,2],[0,0],[1,1]])) )
        for ind,sub in example:
            result = self.bnd.BlockInd2SubWithoutBand(ind)
            npt.assert_array_equal(result,sub)
        ExceptionExample = (10000,8000)
        for i in ExceptionExample:
            self.assertRaises(exceptions.IndexError, self.bnd.BlockInd2SubWithoutBand,i)
            
    def testBlockSub2CenterCarWithoutBand(self):
        '''test BlockSub2CenterCarWithoutBand'''
        example = ( ( 0, a([-1.9,-1.9,-1.9])), 
                    ( 1, a([-1.7,-1.9,-1.9])) )
        for ind,right in example:
            sub = self.bnd.BlockInd2SubWithoutBand(ind)
            rslt = self.bnd.BlockSub2CenterCarWithoutBand(sub)
            npt.assert_array_equal(rslt, right)
            


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()