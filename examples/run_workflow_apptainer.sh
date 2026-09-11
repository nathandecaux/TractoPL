#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 <image.sif> <bids-root> <atlas-manifest> <subject-id>" >&2
    exit 2
fi

image=$1
dataset_root=$2
atlas_manifest=$3
subject_id=$4
atlas_dir=$(dirname "$atlas_manifest")

for path in "$image" "$dataset_root" "$atlas_manifest"; do
    [[ -e "$path" ]] || {
        echo "Required path does not exist: $path" >&2
        exit 2
    }
done

command -v apptainer >/dev/null || {
    echo "Apptainer is not installed or is not on PATH." >&2
    exit 127
}

apptainer exec \
    --cleanenv \
    --bind "$dataset_root:$dataset_root" \
    --bind "$atlas_dir:$atlas_dir" \
    "$image" \
    bash /opt/tractopl/examples/run_workflow.sh \
        "$dataset_root" "$atlas_manifest" "$subject_id"