#!/bin/bash

# Function to check GPU memory usage
check_gpu_memory() {
    # Get the memory usage of GPUs 4, 5, 6, 7
    memory_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
    echo "$memory_usage"
}

# Function to check if all GPUs have memory usage below 1GB
all_gpus_below_threshold() {
    threshold=1000  # 1GB in MB
    for memory in $1; do
        if (( memory >= threshold )); then
            return 1
        fi
    done
    return 0
}

# Initialize the counter
counter=0
required_minutes=10
interval=60  # Check every minute

while true; do
    memory_usage=$(check_gpu_memory)
    if all_gpus_below_threshold "$memory_usage"; then
        ((counter++))
        echo "All GPUs are below 1GB for $counter minutes."
        if (( counter >= required_minutes )); then
            echo "Executing the script as all GPUs have been below 1GB for $required_minutes minutes."
            bash ./train.sh
            exit 0
        fi
    else
        counter=0
        echo "Memory usage is above threshold. Resetting counter."
    fi
    sleep $interval
done