import torch
import time
import argparse
import os
import torch.profiler
from fla.ops.simple_gla import chunk_simple_gla
from fla.ops.simple_gla.fused_chunk import fused_chunk_simple_gla
from fla.ops.simple_gla.parallel import parallel_simple_gla

def run_benchmark(version, B, H, T, D, warmup_runs, num_runs):
    """
    Tests the performance and memory usage of the FORWARD + BACKWARD pass
    for a specified version and shape of simple_gla.

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
    final_state = False

    func_map = {
        'chunk': chunk_simple_gla,
        'fused_chunk': fused_chunk_simple_gla,
        'parallel': parallel_simple_gla
    }
    if version not in func_map:
        raise ValueError(f"Unknown version: {version}. Available versions are: {list(func_map.keys())}")
    target_func = func_map[version]

    print("---- Configuration (Backward Pass) ----")
    print(f"Version to Test: {version}")
    print(f"Device: {device}")
    print(f"Data Type: {dtype}")
    print(f"Input Shape (q, k, v): [B, T, H, D] = [{B}, {T}, {H}, {D}]")
    print(f"Input Shape (g): [B, T, H] = [{B}, {T}, {H}]")
    print(f"Warm-up Runs: {warmup_runs}")
    print(f"Timed Runs: {num_runs}")
    print("---------------------------------------")

    # 2. Prepare Input Tensors with Gradient Tracking
    q = torch.randn(B, T, H, D, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device, requires_grad=True)
    g = torch.randn(B, T, H, dtype=dtype, device=device, requires_grad=True)
    grad_output = torch.randn(B, T, H, D, dtype=dtype, device=device)
    # Define the core operation for benchmarking (forward + backward)
    forward_time_list = []
    backward_time_list = []
    def run_op():
        q.grad = k.grad = v.grad = g.grad = None
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        if version in ['chunk', 'fused_chunk']:
            output = target_func(q, k, v, g, scale=1.0, output_final_state=final_state)
        else:
            output = target_func(q, k, v, g)
        torch.cuda.synchronize()
        forward_end_time = time.perf_counter()
        if isinstance(output, tuple):
            output[0].backward(gradient=grad_output, retain_graph=False)
            
        else:
            output.backward(gradient=grad_output, retain_graph=False)
        torch.cuda.synchronize()
        backward_end_time = time.perf_counter()
        forward_time_list.append(forward_end_time-start_time)
        backward_time_list.append(backward_end_time-forward_end_time)
        
        
    def run_op_mem():
        q.grad = k.grad = v.grad = g.grad = None
        if version in ['chunk', 'fused_chunk']:
            output = target_func(q, k, v, g, scale=1.0, output_final_state=final_state)
        else:
            output = target_func(q, k, v, g)
        if isinstance(output, tuple):
            output[0].backward(gradient=grad_output, retain_graph=False)
            
        else:
            output.backward(gradient=grad_output, retain_graph=False)
    # 3. Warm-up (for manual timing)
    print("\nStarting warm-up (for manual timing)...")
    for _ in range(warmup_runs):
        run_op()
    if device == 'cuda':
        torch.cuda.synchronize()
    print("Warm-up complete.")

    # 4. Performance Test with Manual Timer
    print(f"\nStarting manual timer for {num_runs} runs...")
    # start_time = time.perf_counter()
    for _ in range(num_runs):
        run_op()
    if device == 'cuda':
        torch.cuda.synchronize()
    # end_time = time.perf_counter()
    print("Manual timing complete.")
    # total_time_s_manual = end_time - start_time
    # avg_time_us_manual = (total_time_s_manual / num_runs) * 1_000_000
    forward_time_list.pop(0)
    backward_time_list.pop(0)
    forward_avg_time_us = sum(forward_time_list)/len(forward_time_list) * 1_000_000
    backward_avg_time_us = sum(backward_time_list)/len(backward_time_list) * 1_000_000

    peak_memory_mb = 0
    if device == 'cuda':
        print("\nMeasuring peak memory usage...")
        torch.cuda.reset_peak_memory_stats(device=device)
        run_op_mem()
        torch.cuda.synchronize(device=device)
        peak_memory_bytes = torch.cuda.max_memory_allocated(device=device)
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        print("Memory measurement complete.")

    # 7. Final Results Output
    with torch.no_grad():
        if version in ['chunk', 'fused_chunk']:
            output = target_func(q, k, v, g, scale=1.0, output_final_state=final_state)
        else:
            output = target_func(q, k, v, g)
        output_tensor = output[0] if isinstance(output, tuple) else output

    print("\n---- Benchmark Results (Forward + Backward Pass) ----")
    print(f"Output Shape: {output_tensor.shape}")
    print(f"Forward Timer Avg Time:   {forward_avg_time_us:.2f} us")
    print(f"Backward Timer Avg Time:   {backward_avg_time_us:.2f} us")
    if device == 'cuda':
        # print(f"Profiler CUDA Avg Time:  {avg_time_us_profiler:.2f} us")
        print(f"Peak Memory Usage:       {peak_memory_mb:.2f} MB")
    print("--------------------------------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark the BACKWARD pass of simple_gla versions.")
    
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