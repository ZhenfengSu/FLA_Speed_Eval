import torch
import time
import argparse
from fla.ops.simple_gla import chunk_simple_gla
from fla.ops.simple_gla.fused_chunk import fused_chunk_simple_gla
from fla.ops.simple_gla.parallel import parallel_simple_gla

def run_benchmark(version, B, H, T, D, warmup_runs, num_runs):
    """
    Tests the performance of a specified version and shape of simple_gla 
    using a for loop and warm-up, and calculates the average time.

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
    print(f"Input Shape (q, k, v): [B, H, T, D] = [{B}, {H}, {T}, {D}]")
    print(f"Input Shape (g): [B, T, H] = [{B}, {T}, {H}]") # Note the shape of g
    print(f"Warm-up Runs: {warmup_runs}")
    print(f"Timed Runs: {num_runs}")
    print("--------------------")

    # 2. Prepare Input Tensors
    with torch.no_grad():
        X_mamba = 0.1 * torch.randn(B, T, H, D, dtype=dtype, device=device)
        dt_mamba = torch.ones(B, T, H, dtype=dtype, device=device)
        A_mamba = -0.1 * torch.rand(H, dtype=dtype, device=device)
        B_mamba = 0.1 * torch.randn(B, T, H, D, dtype=dtype, device=device)
        C_mamba = 0.1 * torch.randn(B, T, H, D, dtype=dtype, device=device)

        q = C_mamba.contiguous()
        k = B_mamba.contiguous()
        v = X_mamba.contiguous()
        g = (A_mamba * dt_mamba).contiguous()
        
        # Note: parallel_simple_gla requires different input shapes
        # if version == 'parallel':
        #     q = q.transpose(1, 2) # [B, T, H, D] -> [B, H, T, D]
        #     k = k.transpose(1, 2) # [B, T, H, D] -> [B, H, T, D]
        #     v = v.transpose(1, 2) # [B, T, H, D] -> [B, H, T, D]
        #     # g for parallel is [B, H, T]
        #     g = g.transpose(1, 2)
        #     print("Note: Input tensors for 'parallel' version have been transposed to [B, H, T, D] layout.")


        # 3. Warm-up
        print("\nStarting warm-up...")
        for _ in range(warmup_runs):
            # Pass arguments according to the function signature
            if version in ['chunk', 'fused_chunk']:
                _ = target_func(q, k, v, g, scale=1.0, output_final_state=final_state)
            else: # parallel
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
            else: # parallel
                _ = target_func(q, k, v, g)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        print("Timing complete.")

        # 5. Result Calculation and Output
        total_time_s = end_time - start_time
        avg_time_s = total_time_s / num_runs
        avg_time_us = avg_time_s * 1_000_000

        # Run once to get the output shape
        if version in ['chunk', 'fused_chunk']:
            output = target_func(q, k, v, g, scale=1.0, output_final_state=final_state)
        else: # parallel
            output = target_func(q, k, v, g)
        
        # Handle the case where the return value might be a tuple
        if isinstance(output, tuple):
            output_tensor = output[0]
        else:
            output_tensor = output

        print("\n---- Benchmark Results ----")
        print(f"Output Shape: {output_tensor.shape}")
        print(f"Average Execution Time: {avg_time_us:.2f} us")
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