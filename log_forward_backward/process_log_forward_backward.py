import re
import pandas as pd
import os
from collections import defaultdict

def parse_log_file(file_path):
    """
    Parses a single log file to extract parameters, execution time and memory usage.
    Returns:
        dict[(B,H,T,D)] = {
            'forward_time': float,
            'backward_time': float,
            'memory': float
        }
    """
    param_regex = r"--- Test Parameters: B=(\d+), H=(\d+), T=(\d+), D=(\d+) ---"
    fwd_regex = r"Forward Timer Avg Time:\s*([\d.]+) us"
    bwd_regex = r"Backward Timer Avg Time:\s*([\d.]+) us"
    mem_regex = r"Peak Memory Usage:\s*([\d.]+) MB"

    data = {}
    if not os.path.exists(file_path):
        print(f"Warning: Log file not found: {file_path}")
        return data

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    test_blocks = content.split('--- Test Parameters:')[1:]
    for block in test_blocks:
        block = "--- Test Parameters:" + block
        param_match = re.search(param_regex, block)
        fwd_match = re.search(fwd_regex, block)
        bwd_match = re.search(bwd_regex, block)
        mem_match = re.search(mem_regex, block)

        if param_match and (fwd_match or bwd_match or mem_match):
            params = tuple(map(int, param_match.groups()))
            fwd = float(fwd_match.group(1)) if fwd_match else None
            bwd = float(bwd_match.group(1)) if bwd_match else None
            mem = float(mem_match.group(1)) if mem_match else None

            data[params] = {
                'forward_time': fwd,
                'backward_time': bwd,
                'memory': mem
            }
    return data


def main():
    log_configs = {
        'materialized': 'chunk.log',
        'non-materialized': 'fused_chunk.log',
        'left-product': 'parallel.log'
    }

    all_parsed_data = {}
    for name, log_file in log_configs.items():
        print(f"Parsing {log_file} as '{name}'...")
        all_parsed_data[name] = parse_log_file(log_file)

    combined_forward = defaultdict(dict)
    combined_backward = defaultdict(dict)

    for name, parsed_data in all_parsed_data.items():
        for params, metrics in parsed_data.items():
            B, H, T, D = params
            if 'B' not in combined_forward[params]:
                combined_forward[params].update({'B': B, 'H': H, 'T': T, 'D': D})
                combined_backward[params].update({'B': B, 'H': H, 'T': T, 'D': D})

            combined_forward[params][f'{name}_forward_us'] = metrics['forward_time']
            combined_backward[params][f'{name}_backward_us'] = metrics['backward_time']

    df_forward = pd.DataFrame(list(combined_forward.values()))
    df_backward = pd.DataFrame(list(combined_backward.values()))

    # 只保留 D ∈ {64,96,128}
    target_dims = {64, 96, 128}
    df_forward = df_forward[df_forward['D'].isin(target_dims)]
    df_backward = df_backward[df_backward['D'].isin(target_dims)]

    # === 写 forward 报告 ===
    with pd.ExcelWriter('forward_report.xlsx', engine='openpyxl') as writer:
        df_forward.to_excel(writer, sheet_name='Summary', index=False)
        for h in sorted(df_forward['H'].unique()):
            df_forward[df_forward['H'] == h].to_excel(writer, sheet_name=f'Head={h}', index=False)

    # === 写 backward 报告 ===
    with pd.ExcelWriter('backward_report.xlsx', engine='openpyxl') as writer:
        df_backward.to_excel(writer, sheet_name='Summary', index=False)
        for h in sorted(df_backward['H'].unique()):
            df_backward[df_backward['H'] == h].to_excel(writer, sheet_name=f'Head={h}', index=False)

    print("✅ 已生成 forward_report.xlsx 和 backward_report.xlsx，只包含 D=64,96,128 的数据！")


if __name__ == "__main__":
    main()