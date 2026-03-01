import threading
import multiprocessing
import time
import random

# Generate random N×N matrix
def generate_matrix(N, seed=None):
    if seed is not None:
        random.seed(seed)
    return [[random.randint(1, 10) for _ in range(N)] for _ in range(N)]

def print_matrix(M, name="Matrix"):
    print(f"\n{name}:")
    for row in M:
        print("  ", row)

# 1. SEQUENTIAL Matrix Multiplication (baseline)
def matrix_multiply_sequential(A, B):
    N = len(A)
    C = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]
    return C

# 2. PARALLEL Matrix Multiplication — Threading (Each thread computes one row of C)
def compute_row_thread(A, B, C, row):
    N = len(A)
    for j in range(N):
        total = 0
        for k in range(N):
            total += A[row][k] * B[k][j]
        C[row][j] = total

def matrix_multiply_threaded(A, B):
    N = len(A)
    C = [[0] * N for _ in range(N)]
    threads = []
    for i in range(N):
        t = threading.Thread(target=compute_row_thread, args=(A, B, C, i))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    return C

# 3. PARALLEL Matrix Multiplication — Multiprocessing (Uses Pool.starmap — safe & no Queue deadlock)
def compute_row_process(args):
    A, B, row = args
    N = len(A)
    return row, [sum(A[row][k] * B[k][j] for k in range(N)) for j in range(N)]

def matrix_multiply_multiprocess(A, B):
    N = len(A)
    C = [[0] * N for _ in range(N)]
    args = [(A, B, i) for i in range(N)]
    with multiprocessing.Pool(processes=min(N, multiprocessing.cpu_count())) as pool:
        results = pool.map(compute_row_process, args)
    for row_idx, row_data in results:
        C[row_idx] = row_data
    return C

# Benchmark all three methods
def run_benchmark(N=4, verbose=True):
    print("=" * 60)
    print(f"  Parallel Matrix Multiplication — N={N}×{N}")
    print("=" * 60)

    A = generate_matrix(N, seed=42)
    B = generate_matrix(N, seed=99)

    if verbose and N <= 6:
        print_matrix(A, "Matrix A")
        print_matrix(B, "Matrix B")

    # ── Sequential ──────────────────────────────
    t0 = time.perf_counter()
    C_seq = matrix_multiply_sequential(A, B)
    t_seq = time.perf_counter() - t0

    # ── Threaded ────────────────────────────────
    t0 = time.perf_counter()
    C_thr = matrix_multiply_threaded(A, B)
    t_thr = time.perf_counter() - t0

    # ── Multiprocessing ─────────────────────────
    t0 = time.perf_counter()
    C_mp = matrix_multiply_multiprocess(A, B)
    t_mp = time.perf_counter() - t0

    # ── Verify correctness ──────────────────────
    assert C_seq == C_thr, "Threaded result mismatch!"
    assert C_seq == C_mp,  "Multiprocess result mismatch!"

    if verbose and N <= 6:
        print_matrix(C_seq, "Result Matrix C = A × B")

    print(f"\n{'Method':<22} {'Time (s)':>12} {'Speedup':>10}")
    print("-" * 46)
    print(f"{'Sequential':<22} {t_seq:>12.6f} {'1.00×':>10}")
    speedup_thr = t_seq / t_thr if t_thr > 0 else float('inf')
    speedup_mp  = t_seq / t_mp  if t_mp  > 0 else float('inf')
    print(f"{'Threading':<22} {t_thr:>12.6f} {speedup_thr:>9.2f}×")
    print(f"{'Multiprocessing':<22} {t_mp:>12.6f} {speedup_mp:>9.2f}×")
    print()
    print("All results verified correct!")

if __name__ == "__main__":
    # Small matrix — show full output
    run_benchmark(N=4, verbose=True)

    # Larger matrix — timing only
    print("\n" + "=" * 60)
    print("  Scaling test: N=50×50")
    print("=" * 60)
    run_benchmark(N=50, verbose=False)