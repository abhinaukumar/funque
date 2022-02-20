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

feature_file="./resource/feature_param/${model_name}_features.py"

PYTHONPATH=python ./run_training.py \
            ./resource/dataset/CC_HDDO_dataset.py \
            ${feature_file} \
            ./resource/model_param/libsvmnusvr_v2.py \
            ./model/${model_name}_model.json \
            --cache-result \
            --parallelize \
            --processes ${max_processes} \
            --suppress-plot

PYTHONPATH=python ./run_multi_testing.py \
            ./resource/dataset/multi_test_datasets.py \
            ./model/${model_name}_model.json \
            --cache-result \
            --parallelize \
            --processes ${max_processes} \
            --suppress-plot --print-result

deactivate
