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
    target_func = func_map[version]

    print("---- Configuration (Forward + Backward Pass) ----")
    print(f"Version to Test: {version}")
    print(f"Device: {device}")
    print(f"Data Type: {dtype}")
    print(f"Input Shape (q, k, v): [B, T, H, D] = [{B}, {T}, {H}, {D}]")
    print(f"Input Shape (g): [B, T, H] = [{B}, {T}, {H}]")
    print(f"Warm-up Runs: {warmup_runs}")
    print(f"Timed Runs: {num_runs}")
    print("---------------------------------------")

    # 2. Prepare Input Tensors
    q = torch.randn(B, T, H, D, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device, requires_grad=True)
    g = torch.randn(B, T, H, dtype=dtype, device=device, requires_grad=True)
    grad_output = torch.randn(B, T, H, D, dtype=dtype, device=device)

    # 3. Manual Timing Section
    # This section remains to provide a baseline wall-clock time measurement.
    forward_time_list = []
    backward_time_list = []
    def run_op_for_manual_timing():
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
        # Only append times if it's not a warmup run
        if len(forward_time_list) < num_runs:
            forward_time_list.append(forward_end_time - start_time)
            backward_time_list.append(backward_end_time - forward_end_time)
            
    print("\nStarting warm-up and manual timing runs...")
    # Combine warmup and timed runs
    for i in range(warmup_runs + num_runs):
        run_op_for_manual_timing()
    print("Manual timing complete.")
    
    # Calculate average times from the collected lists
    forward_avg_time_us = sum(forward_time_list[-num_runs:]) / num_runs * 1_000_000
    backward_avg_time_us = sum(backward_time_list[-num_runs:]) / num_runs * 1_000_000

    # --- MODIFIED: PROFILER SECTION FOR MEMORY ANALYSIS ---
    if device == 'cuda':
        print(f"\nStarting torch.profiler for memory analysis...")
        if not os.path.exists("./logs"):
            os.makedirs("./logs")
        
        # MODIFIED: Define a separate function for profiler to use record_function
        def run_op_for_profiler():
            q.grad = k.grad = v.grad = g.grad = None
            
            # NEW: Use record_function to label forward and backward passes
            with torch.profiler.record_function("forward_pass"):
                if version in ['chunk', 'fused_chunk']:
                    output = target_func(q, k, v, g, scale=1.0, output_final_state=final_state)
                else:
                    output = target_func(q, k, v, g)
            
            with torch.profiler.record_function("backward_pass"):
                if isinstance(output, tuple):
                    output[0].backward(gradient=grad_output, retain_graph=True) # retain_graph for multi-run profiling
                else:
                    output.backward(gradient=grad_output, retain_graph=True)

        # Use the profiler context manager for a single, detailed run
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            # MODIFIED: schedule is not ideal for memory summary, we do one clean run
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./logs/{version}_memory_profile'),
            record_shapes=True,
            profile_memory=True, # This is the key for memory profiling
            with_stack=True      # Get python call stack for allocations
        ) as prof:
            # Run one instrumented pass to collect memory data
            run_op_for_profiler()
        
        print("Profiler run complete.")
        print("\n--- Profiler Memory Analysis (Top 15 Operators by Self CUDA Memory Allocation) ---")
        # MODIFIED: Sort the table by CUDA memory usage
        print(prof.key_averages().table(sort_by="self_device_memory_usage", row_limit=15))

    # 5. Peak Memory Usage Test (High-Water Mark)
    peak_memory_mb = 0
    if device == 'cuda':
        print("\nMeasuring peak memory usage (High-Water Mark)...")
        torch.cuda.reset_peak_memory_stats(device=device)
        # Use the original op for a clean run
        run_op_for_manual_timing()
        peak_memory_bytes = torch.cuda.max_memory_allocated(device=device)
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        print("Peak memory measurement complete.")

    # 6. Final Results Output
    print("\n---- Benchmark Results ----")
    print(f"Manual Forward Timer Avg:   {forward_avg_time_us:.2f} us")
    print(f"Manual Backward Timer Avg:  {backward_avg_time_us:.2f} us")
    if device == 'cuda':
        print(f"Peak Memory (High-Water Mark): {peak_memory_mb:.2f} MB")
        print("\nNOTE: For detailed operator-level memory usage, see the profiler table above")
        print(f"      and view the trace file in TensorBoard: tensorboard --logdir ./logs/{version}_memory_profile")
    print("---------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark simple_gla versions with memory profiling.")
    
    # Arguments remain the same
    parser.add_argument('--version', type=str, default='fused_chunk', choices=['chunk', 'fused_chunk', 'parallel'], help='The version of simple_gla to run.')
    parser.add_argument('-B', '--batch_size', type=int, default=1, help='Batch size.')
    parser.add_argument('-H', '--num_heads', type=int, default=16, help='Number of heads.')
    parser.add_argument('-T', '--seq_len', type=int, default=4 * 1024, help='Sequence length.')
    parser.add_argument('-D', '--hidden_dim', type=int, default=128, help='Hidden dimension.')
    parser.add_argument('--warmup_runs', type=int, default=5, help='Number of warm-up runs.')
    parser.add_argument('--num_runs', type=int, default=10, help='Number of benchmark runs.')

    args = parser.parse_args()
    run_benchmark(
        version=args.version, B=args.batch_size, H=args.num_heads,
        T=args.seq_len, D=args.hidden_dim, warmup_runs=args.warmup_runs,
        num_runs=args.num_runs
    )
