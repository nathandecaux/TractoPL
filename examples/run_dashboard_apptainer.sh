#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <image.sif> <tractopl-checkout> <bids-root>" >&2
    exit 2
fi

image=$1
tractopl_dir=$2
dataset_root=$3

apptainer exec \
    --cleanenv \
    --writable-tmpfs \
    --bind "$tractopl_dir:/opt/tractopl" \
    --bind "$dataset_root:$dataset_root" \
    --bind "$dataset_root:$dataset_root" \
    --env "PYTHONPATH=/opt/tractopl" \
    "$image" \
    bash -c '
        set -euo pipefail
        pip install -e /opt/tractopl --no-deps
        streamlit run /opt/tractopl/dashboard/streamlit_dashboard.py --server.headless true --server.port 8501 --server.address 0.0.0.0' _ "$dataset_root"