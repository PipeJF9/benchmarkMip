#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np
import math

#this function calculate base primes numbers using the simple sieve method
def simple_sieve(limit):
    
    if limit < 2:
        return []
    
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    
    for i in range(2, int(math.sqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    
    return [i for i in range(2, limit + 1) if sieve[i]]

#this function divides the range into segments and applies the segmented sieve method
def segmented_sieve_chunk(start, end, base_primes):

    if start < 2:
        start = 2
    
    segment_size = end - start + 1
    is_prime = [True] * segment_size
    
    for prime in base_primes:
        if prime * prime > end:
            break
            
        first_multiple = max(prime * prime, ((start + prime - 1) // prime) * prime)
        
        for multiple in range(first_multiple, end + 1, prime):
            is_prime[multiple - start] = False
    
    primes = []
    for i in range(segment_size):
        if is_prime[i]:
            num = start + i
            if num >= start: 
                primes.append(num)
    
    return primes

#this function distributes the range of numbers among the available processes
def distribute_range(start, end, rank, size):

    total_numbers = end - start + 1
    chunk_size = total_numbers // size
    remainder = total_numbers % size
    
    local_start = start + rank * chunk_size + min(rank, remainder)
    local_size = chunk_size + (1 if rank < remainder else 0)
    local_end = local_start + local_size - 1
    
    return local_start, local_end


#this function counts the number of prime numbers in the range [10^(D-1), 10^D - 1] using parallel processing
def count_primes_sieve_parallel(D):
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    initial = 10 ** (D - 1)
    final = 10 ** D - 1
    
    sqrt_final = int(math.sqrt(final)) + 1
    
    if rank == 0:
        base_primes = simple_sieve(sqrt_final)
    else:
        base_primes = None
    
    base_primes = comm.bcast(base_primes, root=0)
    
    local_start, local_end = distribute_range(initial, final, rank, size)
    local_primes = segmented_sieve_chunk(local_start, local_end, base_primes)
    local_count = len(local_primes)
    
    total_count = comm.reduce(local_count, op=MPI.SUM, root=0)

    if rank == 0:
        print(f"Number of primes in the range [10^{D-1}, 10^{D}-1]: {total_count}")


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("Enter the value of D (1 to 9): ")
        D = int(input())
    else:
        D = None
    D = comm.bcast(D, root=0)
    count_primes_sieve_parallel(D)

if __name__ == "__main__":
    main()