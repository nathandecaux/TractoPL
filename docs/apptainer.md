# Apptainer

[apptainer/TractoPL.def](../apptainer/TractoPL.def) builds the runtime of the external tools:
Anima 4.2 (official Ubuntu binaries, public scripts and scripts data), MRtrix3 and ANTs. It does
not contain TractoPL itself: the checkout is mounted at runtime, so Python changes are used
immediately without rebuilding the image. The image does not include any dataset, atlas, bundle
templates or centroids.

## Build

The build needs a privileged Apptainer build service or a configured `--fakeroot` setup, and
network access to GitHub. Apptainer stages its temporary sandbox under `$APPTAINER_TMPDIR`
(`/tmp` by default); on hosts where `/tmp` is a small or quota-limited tmpfs, point it at a
directory with a few GB free to avoid a `Disk quota exceeded` error mid-build:

```shell
export APPTAINER_TMPDIR=/path/to/scratch/dir     # e.g. /home/$USER/apptainer-tmp
mkdir -p "$APPTAINER_TMPDIR"
apptainer build --fakeroot tractopl.sif apptainer/TractoPL.def
```

## Run the workflow

```shell
bash examples/run_workflow_apptainer.sh \
  tractopl.sif \
  "$PWD" \
  /data/my-bids-dataset \
  /data/my-atlas/atlas.json \
  01
```

The launcher mounts the checkout at `/opt/tractopl`, sets `PYTHONPATH=/opt/tractopl` and runs the
mounted [examples/run_workflow.sh](../examples/run_workflow.sh). Every supplied directory is bound
at the same absolute path inside the container, so relative paths in an atlas manifest stay valid
and derivatives are written directly to the host BIDS root.

The image provides `tractopl-*` wrappers (`tractopl-preprocessing`, `tractopl-msmt-csd`,
`tractopl-mcm`, `tractopl-bundle-seg`, `tractopl-tractometry`, `tractopl-connectome`,
`tractopl-generate-centroid`, `tractopl-frechet-clustering`, `tractopl-apply-trans-to-vtk`,
`tractopl-convert-tractogram`, `tractopl-population-analysis`) that load the modules of the
mounted checkout. Changes to packaging metadata or Python dependencies still require rebuilding
the image.

The [pipeline launcher](applications.md#pipeline-launcher) can also run every step through the
image, and [examples/run_dashboard_apptainer.sh](../examples/run_dashboard_apptainer.sh) starts the
dashboard inside it.
