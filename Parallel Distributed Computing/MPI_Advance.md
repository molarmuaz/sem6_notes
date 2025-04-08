## MPI Advance
Processes may need to communicate with all the others

1) **Communicate**: Broadcast, Gather, Scatter
2) **Synchronization**: Barriers
3) **Reductions**: sum, product, etc.

#### MPI_Bcast:

```c
int MPI_Bcast
(
	void*buf, // Buffer
	int count, //Number of values sent
	MPI_Datatype dtype, //e.g MPI_CHAR
	int root, // Sender's id
	MPI_Comm comm
);
``` 

buf = Buffer
count = size of buffer 
dtype = Datatype (e.g MPI_CHAR)
root = sender's id

All Processes receive buffer including the sender. There is no need to listen for the broadcast or use a receiver function. MPI automatically broadcasts to every processor and changes variable value for example:

```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, number;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        number = 42; // The number to broadcast
    }

    // All processes call MPI_Bcast — this is how they receive the message
    MPI_Bcast(&number, 1, MPI_INT, 0, MPI_COMM_WORLD);

    printf("Process %d received number %d\n", rank, number);

    MPI_Finalize();
    return 0;
}
```
Number value gets changed for every processor 

#### MPI_Scatter: 

```c 
int MPI_Scatter
(
	void*sendbuf, 
	int sendcount,
	MPI_Datatype sendtype, 
	
	void* recvbuf, 
	int recvcount,
	MPI_Datatype recvtype, 
	
	int root, 
	MPI_Comm comm
);
```

It is also one to all like Bcast, but instead of sending the same entire data to every processor, Scatter divides the data into chunks and sends each chunk to a different process.

```c
int data[4] = {10, 20, 30, 40}; // Root process (rank 0) only
int recv;
MPI_Scatter(data, 1, MPI_INT, &recv, 1, MPI_INT, 0, MPI_COMM_WORLD);
// Now: rank 0 gets 10, rank 1 gets 20, etc.
```

The returned values do not necessarily have to be the same data type as the sent values that is why we need to specify datatype for both in the function.

#### MPI_Gather:

```c
int MPI_Gather
(
	void* sendbuf,
	int sendcount,
	MPI_Datatype sendtype,

	void* recvbuf,
	int recvcount,
	MPI_Datatype recvtype,

	int root,
	MPI_Comm comm
);
```

Inverse to Scatter. Called after scatter; It gathers elements from all processes and unifies them in one process (root) .

The recvbuf has the collected values.

```c
int send = rank + 1; // Each process has different data
int gathered[4];     // Root will fill this
MPI_Gather(&send, 1, MPI_INT, gathered, 1, MPI_INT, 0, MPI_COMM_WORLD);
// Root now has [1, 2, 3, 4]
```

#### MPI_Scatterv

Similar to scatter, but the chunks made of the original data are not necessarily equal in size. This is useful in a lot of the cases where not all the processors perform equally good and we don't have to wait for a slow processor to complete the same amount of processing as a faster processor. This way we can divide workload according to competency.

```c
int MPI_Scatterv
(
	const void* sendbuf,
	const int sendcounts[], //Sizes of chunks for e.g p1 gets 3 elements, p2 gets 5
	const int displs[], // displacements, index where each process' chunk starts at
	MPI_Datatype sendtype,

	void* recvbuf,
	int recvcount,
	MPI_Datatype recvtype,

	int root,
	MPI_Comm comm
);
```

Usually displs is based on sendcounts i.e. If chunk is 2 elements long the next displacement will be i+2, but sometimes it is not which is why MPI wants us to define it instead of assuming. The formula to find displs if we want to have it according to sendcount is:

```c
displs[0] = 0;
displs[i] = displs[i-1] + sendcounts[i-1];  // for i >= 1
```

#### MPI_Gatherv

```c
int MPI_Gatherv
(
	const void* sendbuf,     // Data to send from each process
	int sendcount,           // Number of elements each process sends
	MPI_Datatype sendtype,   // Datatype of send buffer

	void* recvbuf,           // Buffer to gather data into (on root only)
	const int recvcounts[],  // How many elements each process sends
	const int displs[],      // Where to place each process's data in recvbuf
	MPI_Datatype recvtype,   // Datatype of receive buffer

	int root,                // Rank of root process (where data is gathered)
	MPI_Comm comm            // Communicator
);
```

Inverse of scatterv, we gather all different sized elements from different processes and store them in recvbuf

#### Example of Scatterv and Gatherv

###### 🍝 **Goal:**

- Root process has this array: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
- It distributes variable chunks to 4 processes.
- Each process doubles its data.
- Root gathers it back into a result array.
```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Root variables
    int sendbuf[10] = {1,2,3,4,5,6,7,8,9,10};
    int sendcounts[4] = {2, 3, 1, 4};    // elements per process
    int displs[4]     = {0, 2, 5, 6};    // starting index in sendbuf

    // Each process gets up to 4 elements
    int recvcount = sendcounts[rank];
    int recvbuf[4]; // local buffer (size enough for max possible chunk)

    // Scatterv: uneven distribution
    MPI_Scatterv(sendbuf, sendcounts, displs, MPI_INT,
                 recvbuf, recvcount, MPI_INT,
                 0, MPI_COMM_WORLD);

    // Double the values
    for (int i = 0; i < recvcount; i++) {
        recvbuf[i] *= 2;
    }

    // Gather the results
    int recvcounts[4] = {2, 3, 1, 4};
    int gatherv_displs[4] = {0, 2, 5, 6};
    int result[10];

    MPI_Gatherv(recvbuf, recvcount, MPI_INT,
                result, recvcounts, gatherv_displs, MPI_INT,
                0, MPI_COMM_WORLD);

    // Print result on root
    if (rank == 0) {
        printf("Final gathered result:\n");
        for (int i = 0; i < 10; i++) {
            printf("%d ", result[i]);
        }
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
```


#### Some more functions

**MPI_Allgather:** Gather but result is available to all processes.
**MPI_Allgatherv:** Gatherv, but result is available to all processes.
**MPI_Alltoall:** Allgather but all processes perform scatter and then they all perform gather
**MPI_Alltoallv:** Alltoall but different sized chunks.

Since, in alltoall each process knows the buffer values of all other processes, we can perform operations like matrix transposition.


## Synchronization

#### MPI_Barrier()
To synchronize data/processes before using them, we use the MPI_Barrier function which waits till all processes are up to speed.

```c
int MPI_Barrier(MPI_Comm comm); //You can use MPI_COMM_WORLD to include all processes
```

##### Use case:
Situation: You want to measure how long each process takes to double elements in an array — **but** you want the timer to start **only after** all processes are ready.

```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int data = rank + 1; // Each process has its own data
    int result;

    // Sync all processes before timing
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Simulate a simple operation
    result = data * 2;

    double end_time = MPI_Wtime();

    printf("Process %d doubled %d to %d in %f seconds\n", 
           rank, data, result, end_time - start_time);

    MPI_Finalize();
    return 0;
}
```

## Reduction
A **reduction** combines values from **all processes** into **a single result** using an operation like: sum, max, min, product, AND/OR, etc.

Input can either be a single value at each process (Scalar) or an array.

We reduce in two ways; The result exists only at the root (MPI_Reduce) or the result exists for every process (MPI_Allreduce)

#### MPI_Reduce

```c
int MPI_Reduce(
    const void* sendbuf,   // What each process sends
    void* recvbuf,         // Where root gets the result
    int count,             // Number of elements in send buf
    MPI_Datatype datatype, // e.g., MPI_INT, MPI_FLOAT
    MPI_Op op,             // e.g., MPI_SUM, MPI_MAX
    int root,              // Who gets the final result
    MPI_Comm comm          // Usually MPI_COMM_WORLD
);

int MPI_Allreduce(
    const void* sendbuf,   // Input buffer (data from each process)
    void* recvbuf,         // Output buffer (result goes here)
    int count,             // Number of elements per process
    MPI_Datatype datatype, // Type of elements (e.g., MPI_INT, MPI_FLOAT)
    MPI_Op op,             // Operation (e.g., MPI_SUM, MPI_MAX)
    MPI_Comm comm          // Communicator (usually MPI_COMM_WORLD)
);

```

#### Use case:
Situation: Sum all process ranks

```c
int rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

int my_value = rank;
int total_sum = 0;

MPI_Reduce(&my_value, &total_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

if (rank == 0)
    printf("Total sum of all ranks = %d\n", total_sum);

```

#### MPI Operations

| MPI Operation | Description                         |
| ------------- | ----------------------------------- |
| `MPI_SUM`     | Sum of values                       |
| `MPI_PROD`    | Product of values                   |
| `MPI_MAX`     | Maximum value                       |
| `MPI_MIN`     | Minimum value                       |
| `MPI_LAND`    | Logical AND (bitwise)               |
| `MPI_BAND`    | Bitwise AND                         |
| `MPI_LOR`     | Logical OR (bitwise)                |
| `MPI_BOR`     | Bitwise OR                          |
| `MPI_LXOR`    | Logical XOR (bitwise)               |
| `MPI_BXOR`    | Bitwise XOR                         |
| `MPI_MAXLOC`  | Maximum value with location (index) |
| `MPI_MINLOC`  | Minimum value with location (index) |

