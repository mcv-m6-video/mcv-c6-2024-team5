#!/bin/bash

CONFIG_FILE="scripts/configurations.json"
PYTHON_SCRIPT="main.py"
PYTHON_ENV="mcvc6"

# Activate conda environment
conda activate $PYTHON_ENV

# Read configurations and run them
jq -c '.[]' $CONFIG_FILE | while read -r config; do
  # Construct command line arguments from JSON, correctly handling boolean flags
  ARGS=$(echo $config | jq -r 'to_entries|map(if .value == true then "--\(.key)" elif .value == false then empty else "--\(.key) \(.value|tostring)" end)|.[]')
  
  echo "Running configuration: $config"
  python $PYTHON_SCRIPT $ARGS
done