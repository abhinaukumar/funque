#!/bin/bash

model_name=$1
max_processes=$2

if [[ -z "$model_name" ]]
then
    model_name="funque"
elif [[ "$model_name" != "funque" ]] && [[ "$model_name" != "pyvmaf" ]]
then
    echo "Invalid model name \"${}\""
    exit 1
fi

if [ -z "$max_processes" ]
then
    max_processes="1"
fi

source .venv/bin/activate

date

PYTHONPATH=python ./run_multi_testing.py \
            ./resource/dataset/multi_test_datasets.py \
            ./model/${model_name}_release.json \
            --cache-result \
            --parallelize \
            --processes ${max_processes} \
            --suppress-plot --print-result

deactivate
