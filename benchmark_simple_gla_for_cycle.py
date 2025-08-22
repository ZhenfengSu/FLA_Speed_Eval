import torch
import time
import argparse
from fla.ops.simple_gla import chunk_simple_gla
from fla.ops.simple_gla.fused_chunk import fused_chunk_simple_gla
from fla.ops.simple_gla.parallel import parallel_simple_gla

def run_benchmark(version, B, H, T, D, warmup_runs, num_runs):
    """
    Tests the performance and memory usage of a specified version and shape of simple_gla.

    Args:
        version (str): The version of simple_gla to test ('chunk', 'fused_chunk', 'parallel').
        B (int): Batch size.
        H (int): Number of heads.
        T (int): Sequence length.
        D (int): Hidden dimension.
        warmup_runs (int): Number of warm-up runs.
        num_runs (int): Number of benchmark runs for timing.
    """
    # 1. Environment and Parameter Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Warning: CUDA not detected. Running on CPU. Performance data will not be representative.")

    dtype = torch.bfloat16
    final_state = False  # This parameter does not affect performance

    # Map the version string to the actual function
    func_map = {
        'chunk': chunk_simple_gla,
        'fused_chunk': fused_chunk_simple_gla,
        'parallel': parallel_simple_gla
    }
    if version not in func_map:
        raise ValueError(f"Unknown version: {version}. Available versions are: {list(func_map.keys())}")
    target_func = func_map[version]

    print("---- Configuration ----")
    print(f"Version to Test: {version}")
    print(f"Device: {device}")
    print(f"Data Type: {dtype}")
    print(f"Input Shape (q, k, v): [B, T, H, D] = [{B}, {T}, {H}, {D}]")
    print(f"Input Shape (g): [B, T, H] = [{B}, {T}, {H}]")
    print(f"Warm-up Runs: {warmup_runs}")
    print(f"Timed Runs: {num_runs}")
    print("--------------------")

    # 2. Prepare Input Tensors
    with torch.no_grad():
        q = torch.randn(B, T, H, D, dtype=dtype, device=device)
        k = torch.randn(B, T, H, D, dtype=dtype, device=device)
        v = torch.randn(B, T, H, D, dtype=dtype, device=device)
        # Note: g has a different shape layout
        g = torch.randn(B, T, H, dtype=dtype, device=device)

    # 3. Warm-up
    print("\nStarting warm-up...")
    for _ in range(warmup_runs):
        if version in ['chunk', 'fused_chunk']:
            _ = target_func(q, k, v, g, scale=1.0, output_final_state=final_state)
        else:
            _ = target_func(q, k, v, g)
    if device == 'cuda':
        torch.cuda.synchronize()
    print("Warm-up complete.")

    # 4. Performance Test and Timing
    print(f"\nStarting timer for {num_runs} runs...")
    start_time = time.perf_counter()
    for _ in range(num_runs):
        if version in ['chunk', 'fused_chunk']:
            _ = target_func(q, k, v, g, scale=1.0, output_final_state=final_state)
        else:
            _ = target_func(q, k, v, g)
    if device == 'cuda':
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    print("Timing complete.")

    total_time_s = end_time - start_time
    avg_time_us = (total_time_s / num_runs) * 1_000_000

    # 5. Memory Usage Test (only on CUDA)
    peak_memory_mb = 0
    if device == 'cuda':
        print("\nMeasuring peak memory usage...")
        # Reset peak memory stats before the run
        torch.cuda.reset_peak_memory_stats(device=device)
        
        # Run the function once to measure its peak memory
        if version in ['chunk', 'fused_chunk']:
            output = target_func(q, k, v, g, scale=1.0, output_final_state=final_state)
        else:
            output = target_func(q, k, v, g)
        
        # Synchronize to ensure the operation is complete
        torch.cuda.synchronize(device=device)
        
        # Get the peak memory allocated in bytes and convert to megabytes
        peak_memory_bytes = torch.cuda.max_memory_allocated(device=device)
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        print("Memory measurement complete.")
    else:
        # If not on CUDA, just run once to get the output shape
        if version in ['chunk', 'fused_chunk']:
            output = target_func(q, k, v, g, scale=1.0, output_final_state=final_state)
        else:
            output = target_func(q, k, v, g)

    # Handle tuple output to get the tensor for shape checking
    if isinstance(output, tuple):
        output_tensor = output[0]
    else:
        output_tensor = output

    # 6. Final Results Output
    print("\n---- Benchmark Results ----")
    print(f"Output Shape: {output_tensor.shape}")
    print(f"Average Execution Time: {avg_time_us:.2f} us")
    if device == 'cuda':
        # Add the new memory metric to the final output
        print(f"Peak Memory Usage: {peak_memory_mb:.2f} MB")
    print("--------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark different versions and shapes of simple_gla.")
    
    parser.add_argument('--version', type=str, default='chunk', 
                        choices=['chunk', 'fused_chunk', 'parallel'],
                        help='The version of simple_gla to run. (default: chunk)')
    
    parser.add_argument('-B', '--batch_size', type=int, default=1, help='Batch size. (default: 1)')
    parser.add_argument('-H', '--num_heads', type=int, default=16, help='Number of heads. (default: 16)')
    parser.add_argument('-T', '--seq_len', type=int, default=4 * 1024, help='Sequence length. (default: 4096)')
    parser.add_argument('-D', '--hidden_dim', type=int, default=128, help='Hidden dimension. (default: 128)')

    parser.add_argument('--warmup_runs', type=int, default=10, help='Number of warm-up runs. (default: 10)')
    parser.add_argument('--num_runs', type=int, default=50, help='Number of benchmark runs. (default: 50)')

    args = parser.parse_args()

    run_benchmark(
        version=args.version,
        B=args.batch_size,
        H=args.num_heads,
        T=args.seq_len,
        D=args.hidden_dim,
        warmup_runs=args.warmup_runs,
        num_runs=args.num_runs
    )