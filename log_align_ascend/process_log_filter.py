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
    # 日志文件和别名映射
    log_configs = {
        'materialized': 'chunk.log',
        'non-materialized': 'fused_chunk.log',
        'left-product': 'parallel.log'
    }

    all_parsed_data = {}
    for name, log_file in log_configs.items():
        print(f"Parsing {log_file} as '{name}'...")
        all_parsed_data[name] = parse_log_file(log_file)

    combined_time = defaultdict(dict)
    combined_mem = defaultdict(dict)

    for name, parsed_data in all_parsed_data.items():
        for params, metrics in parsed_data.items():
            B, H, T, D = params
            if 'B' not in combined_time[params]:
                combined_time[params].update({'B': B, 'H': H, 'T': T, 'D': D})
                combined_mem[params].update({'B': B, 'H': H, 'T': T, 'D': D})

            combined_time[params][f'{name}_forward_us'] = metrics['forward_time']
            combined_time[params][f'{name}_backward_us'] = metrics['backward_time']
            combined_mem[params][f'{name}_mem_mb'] = metrics['memory']

    # 转换为DataFrame
    df_time = pd.DataFrame(list(combined_time.values()))
    df_mem = pd.DataFrame(list(combined_mem.values()))

    # === 只保留 D ∈ {64, 96, 128} 的数据 ===
    target_dims = {64, 96, 128}
    df_time = df_time[df_time['D'].isin(target_dims)]
    df_mem = df_mem[df_mem['D'].isin(target_dims)]

    # === 写时间报告 ===
    with pd.ExcelWriter('time_report.xlsx', engine='openpyxl') as writer:
        # 总表
        df_time.to_excel(writer, sheet_name='Summary', index=False)
        # 按 HeadDim 分表
        for d in sorted(df_time['D'].unique()):
            df_time[df_time['D'] == d].to_excel(writer, sheet_name=f'Dim={d}', index=False)

    # === 写显存报告 ===
    with pd.ExcelWriter('memory_report.xlsx', engine='openpyxl') as writer:
        # 总表
        df_mem.to_excel(writer, sheet_name='Summary', index=False)
        # 按 HeadDim 分表
        for d in sorted(df_mem['D'].unique()):
            df_mem[df_mem['D'] == d].to_excel(writer, sheet_name=f'Dim={d}', index=False)

    print("✅ 已生成 time_report.xlsx 和 memory_report.xlsx，只包含 D=64,96,128 的数据！")


if __name__ == "__main__":
    main()