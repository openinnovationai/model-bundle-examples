#!/bin/bash

set -e

readonly EXAMPLES=("fastai" "pytorch" "sklearn" "statsmodels" "tensorflow" "transformers" "xgboost")

for dir in "${EXAMPLES[@]}"
do
    echo "Testing $dir"
    cd $dir

    if [[ -d ".venv" ]]; then
        rm -rf .venv
    fi
    python -m venv .venv
    source .venv/bin/activate
    pip install --quiet -r requirements.txt

    python create_bundle.py

    cp ../tests/load_predict.py .
    python load_predict.py

    echo "$dir passed"
    rm load_predict.py
    cd ..
done

echo "Tests completed"
