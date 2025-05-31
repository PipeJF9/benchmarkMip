import cupy as cp

def multiply_matrices(A, B):
    # Ensure the input matrices are CuPy arrays
    A = cp.asarray(A)
    B = cp.asarray(B)
    
    # Perform matrix multiplication
    C = cp.matmul(A, B)
    
    return C

def generate_random_matrix(N):
    return cp.random.uniform(0, 1, (N, N))

def main():
    N = int(input("Enter the size of the matrices (N x N): "))
    A = generate_random_matrix(N)
    B = generate_random_matrix(N)
    C = multiply_matrices(A, B)
    #print(C)
if __name__ == "__main__":
    main()
