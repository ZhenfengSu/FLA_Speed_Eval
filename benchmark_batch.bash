#!/bin/bash

# --- Configuration ---
# Name of the Python script
PYTHON_SCRIPT="benchmark_simple_gla_for_cycle.py"
# PYTHON_SCRIPT="benchmark_simple_gla_profile.py"

# Fixed parameters
B=1
# D=128
DIMS=(64 96 128 160 192)

# Arrays of variables to test
# VERSIONS=("chunk" "fused_chunk" "parallel")
VERSIONS=("chunk" "fused_chunk")
# HEADS=(2 4 8 16 32)
HEADS=(2 4 8 12 16 32)
SEQ_LENS=(4096 8192 16384 32768 65536 131072) # 4k, 8k, 16k, 32k, 64k, 128k

# --- Script Start ---
echo "Starting the benchmark script..."
echo "A .log file will be generated for each version."

# Iterate over each version
for VERSION in "${VERSIONS[@]}"; do
    # Define the log file name for the current version
    LOG_FILE="${VERSION}.log"
    
    echo "===================================================="
    echo "Starting tests for version: ${VERSION}"
    echo "All output will be saved to: ${LOG_FILE}"
    echo "===================================================="

    # Create/clear the log file for the current version and write a header
    echo "Benchmark results for version: ${VERSION}" > "$LOG_FILE"
    echo "Test run on: $(date)" >> "$LOG_FILE"
    echo "----------------------------------------------------" >> "$LOG_FILE"

    # Iterate over each head configuration
    for H in "${HEADS[@]}"; do
        # Iterate over each sequence length configuration
        for T in "${SEQ_LENS[@]}"; do
            # Iterate over each dimension configuration
            for D in "${DIMS[@]}"; do
                # Display the currently running test in the terminal
                echo "Running: Version=${VERSION}, H=${H}, T=${T}, D=${D}"

                # Write a separator header for this test in the log file
                echo "" >> "$LOG_FILE"
                echo "--- Test Parameters: B=${B}, H=${H}, T=${T}, D=${D} ---" >> "$LOG_FILE"

                # Execute the Python script and append (>>) standard output to the log file
                python "$PYTHON_SCRIPT" \
                    --version "$VERSION" \
                    -B "$B" \
                    -H "$H" \
                    -T "$T" \
                    -D "$D" \
                    >> "$LOG_FILE"
            done
        done
    done
    
    echo "All tests for version ${VERSION} are complete. Results are saved in ${LOG_FILE}."
done

echo ""
echo "All benchmarks have been completed!"