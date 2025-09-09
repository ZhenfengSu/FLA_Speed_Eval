import re
import pandas as pd
import os
from collections import defaultdict

# 日志文件到最终列名前的映射
LOG_NAME_MAP = {
    "chunk.log": "materialized",
    "fused_chunk.log": "non-materialized",
    "parallel.log": "left-product",
    "lo_mat_mem.log": "lo_mat",
    "lo_cmm_mem.log": "lo_cmm",
}

# 只保留这些D
VALID_D = {64, 96, 128}


def parse_log_file(file_path):
    """
    解析单个日志文件，提取参数和显存数据

    Returns:
        dict: key=(B,H,T,D), value = memory (MB)
    """
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: file not found {file_path}")
        return {}

    param_regex = r"--- Test Parameters: B=(\d+), H=(\d+), T=(\d+), D=(\d+) ---"
    mem_regex = r"Peak Memory Usage:\s*([\d.]+) MB"

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.split("--- Test Parameters:")[1:]
    results = {}
    for block in blocks:
        block = "--- Test Parameters:" + block
        param_match = re.search(param_regex, block)
        mem_match = re.search(mem_regex, block)

        if param_match and mem_match:
            B, H, T, D = map(int, param_match.groups())
            if D not in VALID_D:  # 过滤D
                continue
            memory = float(mem_match.group(1))
            results[(B, H, T, D)] = memory
    return results


def main():
    all_data = defaultdict(dict)

    # 逐个解析日志
    for filename, alias in LOG_NAME_MAP.items():
        print(f"Parsing {filename} as {alias} ...")
        parsed = parse_log_file(filename)
        for params, mem in parsed.items():
            B, H, T, D = params
            if "B" not in all_data[params]:
                all_data[params].update({"B": B, "H": H, "T": T, "D": D})
            all_data[params][alias] = mem

    if not all_data:
        print("❌ No valid data parsed! Exiting.")
        return

    df = pd.DataFrame(list(all_data.values()))

    # 按照H生成不同sheet
    output_excel = "memory_report.xlsx"
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        for h in sorted(df["H"].unique()):
            sheet_name = f"Head={h}"
            df_sheet = df[df["H"] == h].copy()
            # 按照 (B, T, D) 排序更好看
            df_sheet = df_sheet.sort_values(by=["B", "T", "D"])
            df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"✅ Excel saved to {output_excel}")


if __name__ == "__main__":
    main()