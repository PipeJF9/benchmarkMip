#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import json

class BenchmarkSystem:
    def __init__(self):
        self.results = {
            'prime_counting': {'sequential': {}, 'mpi': {}, 'gpu': {}},
            'matrix_multiplication': {'sequential': {}, 'mpi': {}, 'gpu': {}}
        }
        self.mpi_workers = [1, 2, 4, 8, 16]
        self.prime_D_values = [3, 4, 5, 6, 7]  # 10^2 to 10^6 range
        self.matrix_sizes = [100, 200, 400, 800, 1600]
        
    def run_sequential_prime(self, D):
        """Run sequential prime counting"""
        try:
            start_time = time.time()
            # Create temporary input file
            with open('temp_input.txt', 'w') as f:
                f.write(str(D))
            
            result = subprocess.run([sys.executable, 'cntprime_seq.py'], 
                                  stdin=open('temp_input.txt', 'r'),
                                  capture_output=True, text=True, timeout=300)
            end_time = time.time()
            
            os.remove('temp_input.txt')
            
            if result.returncode == 0:
                return end_time - start_time
            else:
                print(f"Error in sequential prime counting for D={D}: {result.stderr}")
                return None
        except Exception as e:
            print(f"Exception in sequential prime counting for D={D}: {e}")
            return None
    
    def run_mpi_prime(self, D, num_workers):
        """Run MPI prime counting"""
        try:
            start_time = time.time()
            # Create temporary input file
            with open('temp_input.txt', 'w') as f:
                f.write(str(D))
            
            # Add --oversubscribe when using more workers than available cores
            mpi_command = ['mpirun']
            if num_workers > 4:  # Tu computadora tiene 4 núcleos
                mpi_command.append('--oversubscribe')
            mpi_command.extend(['-n', str(num_workers), sys.executable, 'cntprime_mpi.py'])
            
            result = subprocess.run(mpi_command,
                                stdin=open('temp_input.txt', 'r'),
                                capture_output=True, text=True, timeout=300)
            end_time = time.time()
            
            os.remove('temp_input.txt')
            
            if result.returncode == 0:
                return end_time - start_time
            else:
                print(f"Error in MPI prime counting for D={D}, workers={num_workers}: {result.stderr}")
                return None
        except Exception as e:
            print(f"Exception in MPI prime counting for D={D}, workers={num_workers}: {e}")
            return None
    
    def run_gpu_prime(self, D):
        """Run GPU prime counting"""
        try:
            start_time = time.time()
            # Create temporary input file
            with open('temp_input.txt', 'w') as f:
                f.write(str(D))
            
            result = subprocess.run([sys.executable, 'cntprime_gpu.py'], 
                                  stdin=open('temp_input.txt', 'r'),
                                  capture_output=True, text=True, timeout=300)
            end_time = time.time()
            
            os.remove('temp_input.txt')
            
            if result.returncode == 0:
                return end_time - start_time
            else:
                print(f"Error in GPU prime counting for D={D}: {result.stderr}")
                return None
        except Exception as e:
            print(f"Exception in GPU prime counting for D={D}: {e}")
            return None
    
    def run_sequential_matrix(self, N):
        """Run sequential matrix multiplication"""
        try:
            start_time = time.time()
            # Create temporary input file
            with open('temp_input.txt', 'w') as f:
                f.write(str(N))
            
            result = subprocess.run([sys.executable, 'matmul_seq.py'], 
                                  stdin=open('temp_input.txt', 'r'),
                                  capture_output=True, text=True, timeout=600)
            end_time = time.time()
            
            os.remove('temp_input.txt')
            
            if result.returncode == 0:
                return end_time - start_time
            else:
                print(f"Error in sequential matrix multiplication for N={N}: {result.stderr}")
                return None
        except Exception as e:
            print(f"Exception in sequential matrix multiplication for N={N}: {e}")
            return None
    
    def run_mpi_matrix(self, N, num_workers):
        """Run MPI matrix multiplication"""
        try:
            start_time = time.time()
            # Create temporary input file
            with open('temp_input.txt', 'w') as f:
                f.write(str(N))
            
            # Add --oversubscribe when using more workers than available cores
            mpi_command = ['mpirun']
            if num_workers > 4:  # Tu computadora tiene 4 núcleos
                mpi_command.append('--oversubscribe')
            mpi_command.extend(['-n', str(num_workers), sys.executable, 'matmul_mpi.py'])
            
            result = subprocess.run(mpi_command,
                                stdin=open('temp_input.txt', 'r'),
                                capture_output=True, text=True, timeout=600)
            end_time = time.time()
            
            os.remove('temp_input.txt')
            
            if result.returncode == 0:
                return end_time - start_time
            else:
                print(f"Error in MPI matrix multiplication for N={N}, workers={num_workers}: {result.stderr}")
                return None
        except Exception as e:
            print(f"Exception in MPI matrix multiplication for N={N}, workers={num_workers}: {e}")
            return None
    
    def run_gpu_matrix(self, N):
        """Run GPU matrix multiplication"""
        try:
            start_time = time.time()
            # Create temporary input file
            with open('temp_input.txt', 'w') as f:
                f.write(str(N))
            
            result = subprocess.run([sys.executable, 'matmul_gpu.py'], 
                                  stdin=open('temp_input.txt', 'r'),
                                  capture_output=True, text=True, timeout=600)
            end_time = time.time()
            
            os.remove('temp_input.txt')
            
            if result.returncode == 0:
                return end_time - start_time
            else:
                print(f"Error in GPU matrix multiplication for N={N}: {result.stderr}")
                return None
        except Exception as e:
            print(f"Exception in GPU matrix multiplication for N={N}: {e}")
            return None
    
    def benchmark_prime_counting(self):
        """Benchmark all prime counting implementations"""
        print("=== BENCHMARKING PRIME COUNTING ===")
        
        # Sequential benchmarking
        print("Running sequential prime counting...")
        for D in self.prime_D_values:
            print(f"  Testing D={D}...")
            exec_time = self.run_sequential_prime(D)
            if exec_time is not None:
                self.results['prime_counting']['sequential'][D] = exec_time
                print(f"    Time: {exec_time:.4f} seconds")
        
        # MPI benchmarking
        print("Running MPI prime counting...")
        for num_workers in self.mpi_workers:
            print(f"  Testing with {num_workers} workers...")
            self.results['prime_counting']['mpi'][num_workers] = {}
            for D in self.prime_D_values:
                print(f"    Testing D={D}...")
                exec_time = self.run_mpi_prime(D, num_workers)
                if exec_time is not None:
                    self.results['prime_counting']['mpi'][num_workers][D] = exec_time
                    print(f"      Time: {exec_time:.4f} seconds")
        
        # GPU benchmarking
        print("Running GPU prime counting...")
        for D in self.prime_D_values:
            print(f"  Testing D={D}...")
            exec_time = self.run_gpu_prime(D)
            if exec_time is not None:
                self.results['prime_counting']['gpu'][D] = exec_time
                print(f"    Time: {exec_time:.4f} seconds")
    
    def benchmark_matrix_multiplication(self):
        """Benchmark all matrix multiplication implementations"""
        print("\n=== BENCHMARKING MATRIX MULTIPLICATION ===")
        
        # Sequential benchmarking
        print("Running sequential matrix multiplication...")
        for N in self.matrix_sizes:
            print(f"  Testing N={N}...")
            exec_time = self.run_sequential_matrix(N)
            if exec_time is not None:
                self.results['matrix_multiplication']['sequential'][N] = exec_time
                print(f"    Time: {exec_time:.4f} seconds")
        
        # MPI benchmarking
        print("Running MPI matrix multiplication...")
        for num_workers in self.mpi_workers:
            print(f"  Testing with {num_workers} workers...")
            self.results['matrix_multiplication']['mpi'][num_workers] = {}
            for N in self.matrix_sizes:
                print(f"    Testing N={N}...")
                exec_time = self.run_mpi_matrix(N, num_workers)
                if exec_time is not None:
                    self.results['matrix_multiplication']['mpi'][num_workers][N] = exec_time
                    print(f"      Time: {exec_time:.4f} seconds")
        
        # GPU benchmarking
        print("Running GPU matrix multiplication...")
        for N in self.matrix_sizes:
            print(f"  Testing N={N}...")
            exec_time = self.run_gpu_matrix(N)
            if exec_time is not None:
                self.results['matrix_multiplication']['gpu'][N] = exec_time
                print(f"    Time: {exec_time:.4f} seconds")
    
    def generate_mpi_tables(self):
        """Generate tables showing MPI performance vs number of workers"""
        print("\n=== GENERATING MPI PERFORMANCE TABLES ===")
        
        # Prime counting MPI table
        print("\nMPI Prime Counting Performance Table:")
        print("Workers\\D", end="")
        for D in self.prime_D_values:
            print(f"\t{D}", end="")
        print()
        
        for num_workers in self.mpi_workers:
            print(f"{num_workers}", end="")
            for D in self.prime_D_values:
                if num_workers in self.results['prime_counting']['mpi'] and \
                   D in self.results['prime_counting']['mpi'][num_workers]:
                    time_val = self.results['prime_counting']['mpi'][num_workers][D]
                    print(f"\t{time_val:.4f}", end="")
                else:
                    print(f"\tN/A", end="")
            print()
        
        # Matrix multiplication MPI table
        print("\nMPI Matrix Multiplication Performance Table:")
        print("Workers\\N", end="")
        for N in self.matrix_sizes:
            print(f"\t{N}", end="")
        print()
        
        for num_workers in self.mpi_workers:
            print(f"{num_workers}", end="")
            for N in self.matrix_sizes:
                if num_workers in self.results['matrix_multiplication']['mpi'] and \
                   N in self.results['matrix_multiplication']['mpi'][num_workers]:
                    time_val = self.results['matrix_multiplication']['mpi'][num_workers][N]
                    print(f"\t{time_val:.4f}", end="")
                else:
                    print(f"\tN/A", end="")
            print()
    
    def find_optimal_mpi_workers(self):
        """Find optimal number of MPI workers for each problem size"""
        optimal_prime = {}
        optimal_matrix = {}
        
        # Find optimal workers for prime counting
        for D in self.prime_D_values:
            best_time = float('inf')
            best_workers = 1
            for num_workers in self.mpi_workers:
                if num_workers in self.results['prime_counting']['mpi'] and \
                   D in self.results['prime_counting']['mpi'][num_workers]:
                    time_val = self.results['prime_counting']['mpi'][num_workers][D]
                    if time_val < best_time:
                        best_time = time_val
                        best_workers = num_workers
            optimal_prime[D] = best_workers
        
        # Find optimal workers for matrix multiplication
        for N in self.matrix_sizes:
            best_time = float('inf')
            best_workers = 1
            for num_workers in self.mpi_workers:
                if num_workers in self.results['matrix_multiplication']['mpi'] and \
                   N in self.results['matrix_multiplication']['mpi'][num_workers]:
                    time_val = self.results['matrix_multiplication']['mpi'][num_workers][N]
                    if time_val < best_time:
                        best_time = time_val
                        best_workers = num_workers
            optimal_matrix[N] = best_workers
        
        return optimal_prime, optimal_matrix
    
    def generate_comparison_plots(self):
        """Generate log-log plots comparing all three implementations"""
        print("\n=== GENERATING COMPARISON PLOTS ===")
        
        optimal_prime, optimal_matrix = self.find_optimal_mpi_workers()
        
        # Prime counting comparison plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        
        # Sequential times
        seq_D = []
        seq_times = []
        for D in self.prime_D_values:
            if D in self.results['prime_counting']['sequential']:
                seq_D.append(10**D - 10**(D-1))  # Range size
                seq_times.append(self.results['prime_counting']['sequential'][D])
        
        # MPI times (optimal workers)
        mpi_D = []
        mpi_times = []
        for D in self.prime_D_values:
            optimal_workers = optimal_prime.get(D, 1)
            if optimal_workers in self.results['prime_counting']['mpi'] and \
               D in self.results['prime_counting']['mpi'][optimal_workers]:
                mpi_D.append(10**D - 10**(D-1))  # Range size
                mpi_times.append(self.results['prime_counting']['mpi'][optimal_workers][D])
        
        # GPU times
        gpu_D = []
        gpu_times = []
        for D in self.prime_D_values:
            if D in self.results['prime_counting']['gpu']:
                gpu_D.append(10**D - 10**(D-1))  # Range size
                gpu_times.append(self.results['prime_counting']['gpu'][D])
        
        if seq_D and seq_times:
            plt.loglog(seq_D, seq_times, 'o-', label='Sequential', linewidth=2, markersize=8)
        if mpi_D and mpi_times:
            plt.loglog(mpi_D, mpi_times, 's-', label='MPI (Optimal)', linewidth=2, markersize=8)
        if gpu_D and gpu_times:
            plt.loglog(gpu_D, gpu_times, '^-', label='GPU', linewidth=2, markersize=8)
        
        plt.xlabel('Range Size (10^D - 10^(D-1))', fontsize=12)
        plt.ylabel('Execution Time (seconds)', fontsize=12)
        plt.title('Prime Counting Performance Comparison', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Matrix multiplication comparison plot
        plt.subplot(1, 2, 2)
        
        # Sequential times
        seq_N = []
        seq_times = []
        for N in self.matrix_sizes:
            if N in self.results['matrix_multiplication']['sequential']:
                seq_N.append(N)
                seq_times.append(self.results['matrix_multiplication']['sequential'][N])
        
        # MPI times (optimal workers)
        mpi_N = []
        mpi_times = []
        for N in self.matrix_sizes:
            optimal_workers = optimal_matrix.get(N, 1)
            if optimal_workers in self.results['matrix_multiplication']['mpi'] and \
               N in self.results['matrix_multiplication']['mpi'][optimal_workers]:
                mpi_N.append(N)
                mpi_times.append(self.results['matrix_multiplication']['mpi'][optimal_workers][N])
        
        # GPU times
        gpu_N = []
        gpu_times = []
        for N in self.matrix_sizes:
            if N in self.results['matrix_multiplication']['gpu']:
                gpu_N.append(N)
                gpu_times.append(self.results['matrix_multiplication']['gpu'][N])
        
        if seq_N and seq_times:
            plt.loglog(seq_N, seq_times, 'o-', label='Sequential', linewidth=2, markersize=8)
        if mpi_N and mpi_times:
            plt.loglog(mpi_N, mpi_times, 's-', label='MPI (Optimal)', linewidth=2, markersize=8)
        if gpu_N and gpu_times:
            plt.loglog(gpu_N, gpu_times, '^-', label='GPU', linewidth=2, markersize=8)
        
        plt.xlabel('Matrix Size (N x N)', fontsize=12)
        plt.ylabel('Execution Time (seconds)', fontsize=12)
        plt.title('Matrix Multiplication Performance Comparison', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Comparison plots saved as 'performance_comparison.png'")
    
    def save_results(self):
        """Save all results to JSON file"""
        with open('benchmark_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        print("Results saved to 'benchmark_results.json'")
    
    def run_full_benchmark(self):
        """Run the complete benchmark suite"""
        print("Starting comprehensive benchmark suite...")
        print("This may take several minutes to complete.\n")
        
        # Check if required files exist
        required_files = ['cntprime_seq.py', 'cntprime_mpi.py', 'cntprime_gpu.py',
                         'matmul_seq.py', 'matmul_mpi.py', 'matmul_gpu.py']
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            print(f"ERROR: Missing required files: {missing_files}")
            return
        
        try:
            # Run benchmarks
            self.benchmark_prime_counting()
            self.benchmark_matrix_multiplication()
            
            # Generate reports
            self.generate_mpi_tables()
            self.generate_comparison_plots()
            self.save_results()
            
            print("\n=== BENCHMARK COMPLETE ===")
            print("All benchmarks completed successfully!")
            print("Check 'performance_comparison.png' for visual results")
            print("Check 'benchmark_results.json' for detailed data")
            
        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user.")
        except Exception as e:
            print(f"\nError during benchmarking: {e}")

def main():
    """Main function with user interface"""
    print("=== HIGH PERFORMANCE COMPUTING BENCHMARK SUITE ===")
    print("This tool benchmarks sequential, MPI, and GPU implementations")
    print("for prime counting and matrix multiplication algorithms.\n")
    
    while True:
        print("Options:")
        print("1. Run full benchmark suite")
        print("2. Benchmark prime counting only")
        print("3. Benchmark matrix multiplication only")
        print("4. Generate MPI performance tables only")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            benchmark_system = BenchmarkSystem()
            benchmark_system.run_full_benchmark()
            break
        elif choice == '2':
            benchmark_system = BenchmarkSystem()
            benchmark_system.benchmark_prime_counting()
            benchmark_system.generate_mpi_tables()
        elif choice == '3':
            benchmark_system = BenchmarkSystem()
            benchmark_system.benchmark_matrix_multiplication()
            benchmark_system.generate_mpi_tables()
        elif choice == '4':
            benchmark_system = BenchmarkSystem()
            # Load existing results if available
            if os.path.exists('benchmark_results.json'):
                with open('benchmark_results.json', 'r') as f:
                    benchmark_system.results = json.load(f)
                benchmark_system.generate_mpi_tables()
            else:
                print("No benchmark results found. Run benchmarks first.")
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()