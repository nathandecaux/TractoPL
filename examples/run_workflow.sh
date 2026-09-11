#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <bids-root> <atlas-manifest> <subject-id>" >&2
    exit 2
fi

dataset_root=$1
atlas_manifest=$2
subject_id=$3

for command in \
    tractopl-preprocessing \
    tractopl-msmt-csd \
    tractopl-mcm \
    tractopl-apply-trans-to-vtk \
    tractopl-bundle-seg \
    tractopl-tractometry; do
    command -v "$command" >/dev/null || {
        echo "Required command is not installed: $command" >&2
        exit 127
    }
done

tractopl-preprocessing \
    --subject "$subject_id" \
    --db-root "$dataset_root"

tractopl-msmt-csd \
    --subject "$subject_id" \
    --db-root "$dataset_root" \
    --step response \
    --step fod \
    --step fixels \
    --step fixels2peaks \
    --step fixel_density \
    --step ifod2_tracto \
    --n-streamlines 1000000

tractopl-mcm estimation \
    --subject "$subject_id" \
    --db-root "$dataset_root"

tractopl-bundle-seg segmentation \
    --subject "$subject_id" \
    --db-root "$dataset_root" \
    --atlas "$atlas_manifest"

tractopl-mcm projection \
    --subject "$subject_id" \
    --db-root "$dataset_root" \
    --bundle-pipeline bundle_seg

tractopl-tractometry association \
    --subject "$subject_id" \
    --db-root "$dataset_root" \
    --atlas "$atlas_manifest" \
    --mcm-pipeline mcm_tensors \
    --bundle-pipeline bundle_seg