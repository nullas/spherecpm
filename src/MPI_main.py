'''
Created on Jul 29, 2012

@author: nullas
'''
from __future__ import division
from surfaces import Sphere as surface
from band import Band
from mpi4py import MPI


try:
    import petsc4py
    import os
    from petsc4py import PETSc
    petsc4py.init(os.sys.argv)
except Exception as exp:
    print exp
    


if __name__ == '__main__':

    bnd = Band(surface,MPI.COMM_WORLD)
    