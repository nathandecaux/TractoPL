#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "Usage: $0 <image.sif> <tractopl-checkout> <bids-root> <atlas-manifest> <subject-id>" >&2
    exit 2
fi

image=$1
checkout=$2
dataset_root=$3
atlas_manifest=$4
subject_id=$5
atlas_dir=$(dirname "$atlas_manifest")

for path in "$image" "$checkout" "$dataset_root" "$atlas_manifest"; do
    [[ -e "$path" ]] || {
        echo "Required path does not exist: $path" >&2
        exit 2
    }
done

[[ -f "$checkout/examples/run_workflow.sh" ]] || {
    echo "TractoPL checkout does not contain examples/run_workflow.sh: $checkout" >&2
    exit 2
}

command -v apptainer >/dev/null || {
    echo "Apptainer is not installed or is not on PATH." >&2
    exit 127
}

apptainer exec \
    --cleanenv \
    --writable-tmpfs \
    --bind "$checkout:/opt/tractopl" \
    --bind "$dataset_root:$dataset_root" \
    --bind "$atlas_dir:$atlas_dir" \
    --env "PYTHONPATH=/opt/tractopl" \
    "$image" \
    bash -c '
        set -euo pipefail
        /usr/bin/python3 -m pip install --no-cache-dir --break-system-packages --quiet -e /opt/tractopl
        exec bash /opt/tractopl/examples/run_workflow.sh "$@"
    ' _ "$dataset_root" "$atlas_manifest" "$subject_id"