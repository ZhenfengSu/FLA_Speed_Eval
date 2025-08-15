# FLA_Speed_Eval
An Speed Evaluation Script for FLA  Library
## Environment Setup
Please follow the steps of [FLA_Install](https://github.com/fla-org/flash-linear-attention)

## Usage for single shape
```bash
python benchmark_simple_gla_for_cycle.py -B batch_size -H num_heads -D dim_head -S seq_len
```

## Usage for bash script
You can choose the head_dim, num_heads, seq_len by modifying the three commands in the bash script.
```bash
DIMS=(64 96 128 160 192)

# Arrays of variables to test
# VERSIONS=("chunk" "fused_chunk" "parallel")
VERSIONS=("chunk" "fused_chunk")
# HEADS=(2 4 8 16 32)
HEADS=(2 4 8 12 16 32)
SEQ_LENS=(4096 8192 16384 32768 65536 131072) # 4k, 8k, 16k, 32k, 64k, 128k
```

After modifying the bash script, you can run it as follows:
```bash
bash benchmark_simple_gla_for_cycle.sh
```
Then the log files will be generated in directory.

## Output the results on Excel
After get the log files, you can run the following command to output the results on Excel.
```bash
python process_logs.py
```
