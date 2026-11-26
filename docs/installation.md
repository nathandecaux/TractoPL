# Installation

## pixi (recommended)

[pixi](https://pixi.sh) installs Python and every dependency from conda-forge, in an environment
local to the checkout and pinned by `pixi.lock`:

```shell
git clone https://github.com/nathandecaux/TractoPL.git
cd TractoPL
pixi install            # default environment: TractoPL (editable) + web applications
pixi run gui            # start the web applications
pixi shell              # shell with the tractopl-* commands on the PATH
```

| Environment | Content | Example |
|---|---|---|
| `default` | TractoPL (editable install of the checkout) and the web applications | `pixi run tractopl-mcm --help` |
| `test` | `default` + pytest | `pixi run test` |
| `build` | tools to build the wheel | `pixi run -e build wheel` |

## Conda package

Each release publishes a `noarch` conda package `tractopl` on the
[prefix.dev channel](https://prefix.dev/channels/tractopl) (the `.conda` file is also attached to
the GitHub release). Its dependencies come from conda-forge:

```shell
pixi add --channel https://prefix.dev/tractopl tractopl        # in a pixi workspace
conda install -c https://repo.prefix.dev/tractopl -c conda-forge tractopl
```

The web applications need `streamlit plotly panel hvplot bokeh` in addition.

The conda package installs on Linux x86_64 and Intel macOS only: its dependency `antspyx` is not
built by conda-forge for Apple Silicon. On Apple Silicon, use the pixi workspace (it takes
`antspyx` from PyPI, macOS ≥ 14) or pip.

## pip

```shell
pip install "TractoPL[dashboard] @ git+https://github.com/nathandecaux/TractoPL.git"
# a given version:
pip install "TractoPL[dashboard] @ git+https://github.com/nathandecaux/TractoPL.git@v0.1.0"
# or the wheel attached to a release:
pip install "tractopl-0.1.0-py3-none-any.whl[dashboard]"
```

Optional extras:

| Extra | Content |
|---|---|
| `dashboard` | web applications (`tractopl-gui*`) |
| `tractosearch` | `lpqtree` and `tractosearch`, for `tractosearch_nearest_in_radius` and the TractoSearch based steps |
| `dev` | pytest |

## External tools

TractoPL calls neuroimaging programs that are not Python packages. Install the ones needed by the
steps you run, or use the [Apptainer image](apptainer.md) that contains them:

| Tool | Used by |
|---|---|
| [Anima](https://github.com/Inria-Empenn/Anima-Public) (binaries, scripts and scripts data) | preprocessing, MCM |
| [MRtrix3](https://www.mrtrix.org/) | FODs, fixels, tractography |
| [ANTs](https://github.com/ANTsX/ANTs) | registration (bundle segmentation) |
| [TractoSearch](https://github.com/StongeEtienne/tractosearch) | bundle segmentation |
| TractSeg, FSL | legacy TractSeg steps |

Their locations are set with environment variables:

| Variable | Value |
|---|---|
| `TRACTOPL_ANIMA_DIR` | Anima binaries |
| `TRACTOPL_ANIMA_DATA_DIR` | Anima scripts data |
| `TRACTOPL_ANIMA_SCRIPTS_DIR` | Anima public scripts |
| `TRACTOPL_ANIMA_PRIVATE_SCRIPTS_DIR` | Anima private scripts |
| `TRACTOPL_TRACTSEG_DIR` | TractSeg |
| `TRACTOPL_MRTRIX_DIR` | MRtrix3 binaries |
| `TRACTOPL_DATASET_ROOT` | default BIDS root for the Python API and the dashboard |

Legacy Anima and TractSeg configuration files are still recognized as a fallback.
