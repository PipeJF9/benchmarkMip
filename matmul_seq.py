import random
N = int(input("Enter the size of the matrices (N x N): "))

# This script performs matrix multiplication of two N x N matrices
# using a simple triple nested loop approach.
def multiply_matrices(A, B):
    C = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]
    return C

def generate_random_matrix():
    return [[random.uniform(0, 1) for _ in range(N)] for _ in range(N)]

def main():
    A = generate_random_matrix()
    B = generate_random_matrix()
    C = multiply_matrices(A, B)
    return C

if __name__ == "__main__":
    C = main()
    print("Resultant matrix C:")
    for row in C:
        print(row)
    print("Matrix multiplication completed.")