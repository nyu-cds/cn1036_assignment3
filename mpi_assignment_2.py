from mpi4py import MPI
import sys
import numpy

def get_input():
    """
        verify the input value.
        if input is an integer less than 100, then return the value
        otherwise, send a notification and request another input
    """
    
    try:
        n = int(input("please enter an integer less than 100\n"))
        if n>=100:
            raise Exception("the number should be an integer less than 100.")
    except Exception as e:
        print("Error:", e)
        n = get_input()
    return n

if __name__ == '__main__':
    """
        process 0 receives the initial value from the user and send it to process 1
        process i gets the value from process (i-1), multiply the value by i and send the result to process (i+1)
        process 0 gets the value from the last process and outputs the final result
    """
    
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    buff = numpy.zeros(1)

    if rank == 0:
        buff[0] = get_input()
        comm.Send(buff, dest=1)
        comm.Recv(buff, source=size-1)
        print("Process 0 received %d" %(buff[0]))

    else:
        comm.Recv(buff, source=rank-1)
        buff[0] = buff[0]*rank
        comm.Send(buff, dest=(rank+1)%size)