from mpi4py import MPI
import random
import math

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


if rank == 0:
    print("Enter the size of the matrices (N x N): ")
    N = int(input())
else:
    N = None

N = comm.bcast(N, root=0)

def generate_random_matrix():
    return [[random.uniform(0, 1) for _ in range(N)] for _ in range(N)]

def multiply_matrices_partial(A_rows, B, num_rows):
    result = [[0] * N for _ in range(num_rows)]
    for i in range(num_rows):
        for j in range(N):
            for k in range(N):
                result[i][j] += A_rows[i][k] * B[k][j]
    return result

if rank == 0:
    # Generate two random matrices on the root process
    A = generate_random_matrix()
    B = generate_random_matrix()
    C = [[0] * N for _ in range(N)]
    
    # Broadcast matrix B to all processes
    comm.bcast(B, root=0)
    
    # Calculate workload per process
    rows_per_process = math.ceil(N / size)
    
    # Send portions of A to each process
    for i in range(1, size):
        start_row = i * rows_per_process
        if start_row >= N:
            # No more rows to process
            comm.send(None, dest=i)
        else:
            end_row = min(start_row + rows_per_process, N)
            rows_to_send = A[start_row:end_row]
            comm.send((rows_to_send, start_row), dest=i)
    
    # Process 0 computes its portion
    start_row = 0
    end_row = min(rows_per_process, N)
    local_rows = A[start_row:end_row]
    local_result = multiply_matrices_partial(local_rows, B, end_row - start_row)
    
    # Copy local result to C
    for i in range(len(local_result)):
        C[start_row + i] = local_result[i]
    
    # Receive results from other processes
    for i in range(1, size):
        result = comm.recv(source=i)
        if result is not None:
            rows, start_row = result
            for j in range(len(rows)):
                C[start_row + j] = rows[j]
                
    # Uncomment if you want to print the entire matrix
    #print("Multiplication result:")
    #print(C)
    print("Multiplication completed.")
    
else:
    # Receive broadcast of matrix B
    B = comm.bcast(None, root=0)
    
    # Receive portion of matrix A
    data = comm.recv(source=0)
    
    if data is not None:
        A_rows, start_row = data
        local_result = multiply_matrices_partial(A_rows, B, len(A_rows))
        comm.send((local_result, start_row), dest=0)
    else:
        comm.send(None, dest=0)
