'''
Created on Jul 29, 2012

@author: nullas
'''
from __future__ import division
from petsc4py import PETSc
from mpi4py import MPI
import scipy as sp
from numpy import array as a
import exceptions


class Band(object):
    '''
    classdocs
    '''


    def __init__(self,surface,comm):
        '''
        Constructor
        '''
        self.comm = comm
        OptDB = PETSc.Options()
        self.M = OptDB.getInt('M',20)
        self.m = OptDB.getInt('m',10)
        self.StencilWidth = OptDB.getInt('p',2)
        self.Dim = OptDB.getInt('d',3)
        self.hBlock = 4/self.M
        self.hGrid = self.hBlock/self.m
        
        self.numTotalBlocks = self.M**self.Dim
        self.numBlocksAssigned = self.numTotalBlocks // comm.size + int(comm.rank < (self.numTotalBlocks % comm.size))
        self.BlockStart = comm.exscan(self.numBlocksAssigned)
        if comm.rank == 0:self.BlockStart = 0
        indBlocks = sp.arange(self.BlockStart,self.BlockStart+self.numBlocksAssigned)
        subBlocks = self.BlockInd2SubWithoutBand(indBlocks)
        CenterBlocks = self.BlockSub2CenterCarWithoutBand(subBlocks)
    
    def BlockInd2SubWithoutBand(self,ind):
        if sp.isscalar(ind):
            if ind < 0 or ind > self.numTotalBlocks:
                raise exceptions.IndexError('BlockInd2SubWithouBand')
        else:
            if ind.any() < 0 or ind.any() >self.numTotalBlocks:
                raise exceptions.IndexError('BlockInd2SubWithouBand')
        sub = []
        for i in range(0,self.Dim):
            sub.append( ind%self.M )
            ind //= self.M
        return a(sub,dtype='int')
    def BlockSub2CenterCarWithoutBand(self,sub):       
        return sub/self.M*4-2+self.hBlock/2
        
        
        