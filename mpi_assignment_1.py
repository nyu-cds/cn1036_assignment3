"""
    Processes with even rank print “Hello” and the processes with odd rank will print “Goodbye”.
    
"""

from mpi4py import MPI

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # check rank #
    if rank%2 == 0:
        print('Hello from process %d'%(rank))
    else:
        print('Goodbye from process %d'%(rank))