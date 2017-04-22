"""
    The program sorts the data set in parallel and verifies the result when it's finished.
    The data set is randomly generated by process 0.
    
"""

from mpi4py import MPI
import numpy as np

# set global variable #
dataset_size = 10000

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # initialize bins #
    bins = list()
    for i in range(size):
        bin = list()
        bins.append(bin)

    if rank == 0:
        # generate unsorted dataset #
        dataset = np.random.randint(dataset_size, size=dataset_size)
        min = min(dataset)
        max = max(dataset)
        interval = (max - min)/size

        # distribute data points into bins #
        for i in dataset:
            if i == max:
                process = size-1
            else:
                process = int((i-min)/interval)
            
            bins[process].append(i)

    # send out data to each process #
    data = comm.scatter(bins, root=0)
    data.sort()

    # collect results #
    sorted_data = comm.gather(data, root=0)

    if rank == 0:
        # reshape the result #
        sorted_list = list()
        for i in range(len(sorted_data)):
            sorted_list.extend(sorted_data[i])

        # verify that the result is correct #
        try:
            assert sorted_list == sorted(dataset), "result is incorrect"
        except AssertionError as e:
            print("AssertionError:%s"%(e))


