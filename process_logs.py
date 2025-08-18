import re
import pandas as pd
import os
from collections import defaultdict

def parse_log_file(file_path):
    """
    Parses a single log file to extract parameters and execution time.
    
    Args:
        file_path (str): The path to the log file.

    Returns:
        dict: A dictionary where keys are parameter tuples (B, H, T, D) and values are the execution times.
    """
    # Define regex to match parameters and execution time.
    param_regex = r"--- Test Parameters: B=(\d+), H=(\d+), T=(\d+), D=(\d+) ---"
    # This regex matches the English output format.
    time_regex = r"Average Execution Time:\s*([\d.]+) us"
    
    data = {}
    if not os.path.exists(file_path):
        print(f"Warning: Log file not found: {file_path}")
        return data

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Split the content into test blocks based on the '--- Test Parameters:' separator.
    test_blocks = content.split('--- Test Parameters:')[1:]
    
    for block in test_blocks:
        # Re-assemble the full block text for matching.
        full_block_text = "--- Test Parameters:" + block
        param_match = re.search(param_regex, full_block_text)
        time_match = re.search(time_regex, full_block_text)
        
        if param_match and time_match:
            # Extract parameters and convert to integers.
            params = tuple(map(int, param_match.groups()))
            # Extract time and convert to a float.
            time = float(time_match.group(1))
            data[params] = time
            
    return data


def main():
    """
    Main function to execute log parsing and Excel report generation.
    """

    # Define file names for all three versions
    materialized_log = 'chunk.log'
    non_materialized_log = 'fused_chunk.log'
    parallel_log = 'parallel.log' # <-- Added parallel log file
    output_excel_file = 'performance_report.xlsx'

    # 1. Parse the three log files
    print(f"Parsing {materialized_log}...")
    materialized_data = parse_log_file(materialized_log)
    
    print(f"Parsing {non_materialized_log}...")
    non_materialized_data = parse_log_file(non_materialized_log)
    
    print(f"Parsing {parallel_log}...") # <-- Added parsing for parallel
    parallel_data = parse_log_file(parallel_log)

    if not all([materialized_data, non_materialized_data, parallel_data]):
        print("Warning: One or more log files might be empty or could not be parsed.")
    if not any([materialized_data, non_materialized_data, parallel_data]):
        print("Error: All log files are empty or could not be parsed. Exiting.")
        return

    # 2. Merge the data
    # Using defaultdict simplifies the code; it returns an empty dictionary for a new key.
    combined_data = defaultdict(dict)

    # Populate data for materialized (chunk)
    for params, time in materialized_data.items():
        B, H, T, D = params
        combined_data[params].update({
            'B': B, 'H': H, 'S': T, 'D': D, # Note: T is renamed to S
            'materialized': time
        })

    # Populate data for non-materialized (fused_chunk)
    for params, time in non_materialized_data.items():
        B, H, T, D = params
        # If this configuration doesn't exist, create an entry for it.
        if 'B' not in combined_data[params]:
             combined_data[params].update({
                'B': B, 'H': H, 'S': T, 'D': D
            })
        combined_data[params]['non-materialized'] = time
        
    # Populate data for parallel, naming the column 'left-product' <-- MODIFIED SECTION
    for params, time in parallel_data.items():
        B, H, T, D = params
        # If this configuration doesn't exist, create an entry for it.
        if 'B' not in combined_data[params]:
             combined_data[params].update({
                'B': B, 'H': H, 'S': T, 'D': D
            })
        # Use the requested column name 'left-product'
        combined_data[params]['left-product'] = time
        
    # 3. Convert the merged data into a Pandas DataFrame
    if not combined_data:
        print("Could not parse any valid data from the log files.")
        return
        
    df = pd.DataFrame(list(combined_data.values()))
    
    # Ensure the column order is correct and handle any missing data (fill with NaN).
    # <-- Added 'left-product' to the column list
    final_columns = ['B', 'H', 'S', 'D', 'materialized', 'non-materialized', 'left-product']
    df = df.reindex(columns=final_columns)

    # 4. Write to different Excel sheets based on the 'D' (Head Dimension) value
    print(f"\nGenerating Excel report: {output_excel_file}...")
    try:
        with pd.ExcelWriter(output_excel_file, engine='openpyxl') as writer:
            # Get all unique Head Dimension values
            head_dims = sorted(df['D'].unique())
            
            for dim in head_dims:
                sheet_name = f'HeadDim={dim}'
                print(f"  - Writing to Sheet: {sheet_name}")
                
                # Filter the data for the current dimension
                df_sheet = df[df['D'] == dim].copy()
                
                # Write the data to the corresponding sheet without the pandas index column
                df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"\nReport generated successfully! File saved as: {output_excel_file}")

    except Exception as e:
        print(f"\nAn error occurred while generating the Excel file: {e}")


if __name__ == "__main__":
    main()