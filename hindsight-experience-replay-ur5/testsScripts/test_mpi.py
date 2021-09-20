from mpi4py import MPI
import numpy as np
comm = MPI.COMM_WORLD
n_steps = 10
data = np.random.choice(a=[1,0],size = n_steps).astype(int)
print("this is our array: {}. \nPrinted by rank: ".format(data) + str(comm.rank))

avg = comm.allreduce(data,op=MPI.SUM) / (comm.size)
avg = np.mean(avg)
if comm.rank == 0:
    print("The average is: {}".format(avg)) 

data = np.array(comm.gather(data,root=0)).ravel()

if comm.rank==0:
    print("length of gathered data: {}".format(len(data)))
    print("The second entry of data is: {}".format(data[1]))
    print("This is the gathered data: {}".format(data))
if comm.rank == 0:
    #sqDiff = np.zeros(10)
    sqDiff = 0
    for i,_ in enumerate(data):
        sqDiff += np.square(np.array(data[i]).astype(np.float32)-avg)
        #sqDiff = (np.array(sqDiff) + (np.array(data[i])-avg)
    sqDiff /= (comm.size*n_steps)

    sqDiff = np.sqrt(sqDiff)

    print("The standard deviation of data is {}.".format(sqDiff))

