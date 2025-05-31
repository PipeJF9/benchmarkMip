import cupy as cp


# This code counts the number of prime numbers in the range [10^(D-1), 10^D - 1]
# The method used is the Sieve of Eratosthenes for efficiency, 
# implemented using CuPy for GPU acceleration.
def count_primes(D):
    
    initial = 10 ** (D - 1)
    final = 10 ** D - 1
    
    if initial < 2:
        initial = 2
        
    is_prime = cp.ones(final + 1, dtype=cp.bool_)
    is_prime[0] = is_prime[1] = False 
    
    for i in range(2, int(final**0.5) + 1):
        if is_prime[i]:
            is_prime[i * i:final + 1:i] = False

    count = cp.sum(is_prime[initial:final + 1])
    
    return int(count)
def main():
    D = int(input("Enter the value of D (1 to 9): "))
    if D < 1 or D > 9:
        print("D must be between 1 and 9.")
        return
    
    prime_count = count_primes(D)
    print(f"Number of primes in the range [10^{D-1}, 10^{D}-1]: {prime_count}")

if __name__ == "__main__":
    main()