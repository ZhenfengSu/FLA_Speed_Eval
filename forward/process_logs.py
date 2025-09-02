import re
import pandas as pd
import os
from collections import defaultdict


def parse_log_file(file_path):
    """
    Parses a single log file to extract parameters, execution time, and memory usage.
    
    Args:
        file_path (str): The path to the log file.

    Returns:
        dict: A dictionary where keys are parameter tuples (B, H, T, D) 
              and values are dictionaries containing 'time' and 'memory'.
    """
    # Define regex to match parameters, time, and memory.
    param_regex = r"--- Test Parameters: B=(\d+), H=(\d+), T=(\d+), D=(\d+) ---"
    time_regex = r"Average Execution Time:\s*([\d.]+) us"
    mem_regex = r"Peak Memory Usage:\s*([\d.]+) MB"
    
    data = {}
    if not os.path.exists(file_path):
        print(f"Warning: Log file not found: {file_path}")
        return data

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Split the content into test blocks.
    test_blocks = content.split('--- Test Parameters:')[1:]
    
    for block in test_blocks:
        full_block_text = "--- Test Parameters:" + block
        param_match = re.search(param_regex, full_block_text)
        time_match = re.search(time_regex, full_block_text)
        mem_match = re.search(mem_regex, full_block_text) # Try to find memory usage
        
        if param_match and time_match:
            params = tuple(map(int, param_match.groups()))
            time = float(time_match.group(1))
            # Handle cases where memory usage might not be in the log
            memory = float(mem_match.group(1)) if mem_match else None
            
            data[params] = {'time': time, 'memory': memory}
            
    return data


def main():
    """
    Main function to execute log parsing and generate a comprehensive Excel report.
    """
    # --- Uncomment the line below to generate dummy files for testing ---
    # create_dummy_logs()

    # Configuration for log files and their corresponding column names in the report
    log_configs = {
        'materialized': 'chunk.log',
        'non-materialized': 'fused_chunk.log',
        'left-product': 'parallel.log'
    }
    output_excel_file = 'performance_report.xlsx'

    # 1. Parse all configured log files
    all_parsed_data = {}
    for name, log_file in log_configs.items():
        print(f"Parsing {log_file} for '{name}' data...")
        all_parsed_data[name] = parse_log_file(log_file)

    # 2. Merge data from all sources
    combined_data = defaultdict(dict)
    
    for name, parsed_data in all_parsed_data.items():
        if not parsed_data:
            print(f"No data found for '{name}'.")
            continue
            
        for params, metrics in parsed_data.items():
            B, H, T, D = params
            # Ensure the base parameter info is set
            if 'B' not in combined_data[params]:
                combined_data[params].update({'B': B, 'H': H, 'S': T, 'D': D})
            
            # Add the time and memory metrics with descriptive names
            combined_data[params][f'{name}_us'] = metrics.get('time')
            combined_data[params][f'{name}_mem_mb'] = metrics.get('memory')
            
    # 3. Convert the merged data into a Pandas DataFrame
    if not combined_data:
        print("Error: Could not parse any valid data from the log files. Exiting.")
        return
        
    df = pd.DataFrame(list(combined_data.values()))
    
    # Define the final column order dynamically
    base_columns = ['B', 'H', 'S', 'D']
    metric_columns = []
    for name in log_configs.keys():
        metric_columns.append(f'{name}_us')
        metric_columns.append(f'{name}_mem_mb')
        
    final_columns = base_columns + metric_columns
    df = df.reindex(columns=final_columns)

    # 4. Write to different Excel sheets based on the 'D' (Head Dimension) value
    print(f"\nGenerating Excel report: {output_excel_file}...")
    try:
        with pd.ExcelWriter(output_excel_file, engine='openpyxl') as writer:
            head_dims = sorted(df['D'].unique())
            
            for dim in head_dims:
                sheet_name = f'HeadDim={dim}'
                print(f"  - Writing to Sheet: {sheet_name}")
                
                df_sheet = df[df['D'] == dim].copy()
                df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"\nReport generated successfully! File saved as: {output_excel_file}")

    except Exception as e:
        print(f"\nAn error occurred while generating the Excel file: {e}")


if __name__ == "__main__":
    main()