# Processing workflow

## Prepare a dataset

Pass the BIDS root explicitly to every command with `--db-root`, or set `TRACTOPL_DATASET_ROOT`
for the Python API. TractoPL never selects a local dataset automatically.

The initial input of one participant is a BIDS DWI image with its `.bval` and `.bvec` files.
Existing BIDS derivatives can stay where they are: `Dataset` indexes raw and derivative files
together, `Subject` selects files by their entities, and `BIDSFile` keeps the entities needed to
name new outputs.

```python
from TractoPL.data.loader import Dataset

dataset = Dataset("/data/my-bids-dataset")
print(dataset.describe())          # participants, files and derivative pipelines found
subject = dataset.get_subject("01")
```

## Prepare an atlas

Bundle segmentation and tractometry need an atlas manifest. Copy
[atlas_manifest.example.json](../TractoPL/config_profiles/atlas_manifest.example.json) next to the
atlas and update its paths:

- required: `name`, `reference`, `bundles_dir`;
- used only by the steps that need them: `centroids_dir`, `parcellations_dir`, `bundle_mapping`.

Paths relative to the manifest are supported. Custom bundle names are supported and must not
contain underscores.

To use HCP105, download the atlas from [Zenodo](https://zenodo.org/records/1477956), apply the
[templating procedure](https://zenodo.org/records/22641462), and write a manifest pointing to the
reference image, bundle templates and centroids. The optional
[hcp105_legacy.json](../TractoPL/config_profiles/hcp105_legacy.json) profile can be used as
`bundle_mapping` to keep the historical HCP names.

## Complete workflow

[examples/run_workflow.sh](../examples/run_workflow.sh) runs every step for one participant and
stops at the first failure (details in [examples/README.md](../examples/README.md)):

```shell
bash examples/run_workflow.sh /data/my-bids-dataset /data/my-atlas/atlas.json 01
```

It runs preprocessing, creates FODs/fixels and an iFOD2 tractogram, fits MCM, segments the bundles
from the whole-brain tractogram, projects the MCM metrics onto those bundles and writes the
tractometry CSV files.

The same steps can be run from the browser with the [pipeline launcher](applications.md#pipeline-launcher).

## Command-line tools

| Command | Purpose |
|---|---|
| `tractopl-preprocessing` | Preprocess DWI, fit DTI, compute FA. |
| `tractopl-msmt-csd` | Responses, FODs, fixels and tractography. |
| `tractopl-mcm` | Fit MCM, or project MCM metrics onto bundle tractograms. |
| `tractopl-bundle-seg` | Register the atlas and segment bundles. |
| `tractopl-tractometry` | Associate metrics to centroids, or combine CSV files. |
| `tractopl-connectome` | Bundle density metrics. |
| `tractopl-generate-centroid` | Centroid VTK of one tractogram. |
| `tractopl-frechet-clustering` | Centroid clusters with Fréchet distances. |
| `tractopl-quickbundle` | QuickBundles clustering of a tractogram. |
| `tractopl-apply-trans-to-vtk` | Apply a transform to a VTK tractogram. |
| `tractopl-convert-tractogram` | Convert a tractogram format. |
| `tractopl-population-analysis` | Population-level analysis of tractometry data. |
| `tractopl-report-analysis` | PDF report of a population analysis. |
| `tractopl-gui`, `tractopl-gui-*` | [Web applications](applications.md). |

Typical commands for one participant:

```shell
tractopl-preprocessing --subject 01 --db-root /data/my-bids-dataset
tractopl-msmt-csd --subject 01 --db-root /data/my-bids-dataset \
  --step response --step fod --step fixels --step fixels2peaks --step ifod2_tracto
tractopl-mcm estimation --subject 01 --db-root /data/my-bids-dataset
tractopl-bundle-seg segmentation --subject 01 --db-root /data/my-bids-dataset \
  --atlas /data/my-atlas/atlas.json
tractopl-mcm projection --subject 01 --db-root /data/my-bids-dataset \
  --bundle-pipeline bundle_seg
tractopl-tractometry association --subject 01 --db-root /data/my-bids-dataset \
  --atlas /data/my-atlas/atlas.json --mcm-pipeline mcm_tensors --bundle-pipeline bundle_seg
```

MCM uses the peak/fixel density when it is available. `--n-comparts` fits a fixed number of
anisotropic compartments, `--model-selection` selects it with AIC. The historical
maximum-likelihood mode and Levenberg optimizer remain the defaults.

Tractometry reads the bundle names from the BIDS derivative entities instead of imposing an HCP
list. `tractopl-tractometry combine-csv` creates TractSeg-style metric CSV files from an existing
tractometry derivative.

## Population analysis

The AFQ / population analysis is configured in JSON. Start from
[population_analysis_example_config.json](../TractoPL/analysis/population_analysis_example_config.json),
set `dataset` to the BIDS root and `hcp_asso_pipeline` to the tractometry derivative name:

```shell
tractopl-population-analysis -c analysis_config.json \
  --subjects-table /data/my-bids-dataset/participants.xlsx \
  --output-dir /data/my-bids-dataset/derivatives/population_analysis
```

The participant table and output directory are optional. A missing or invalid configuration, or
the `/PATH/TO/DATASET` placeholder, is rejected before the analysis starts. The
[population analysis application](applications.md) builds this configuration interactively.
