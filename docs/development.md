# Development

```shell
git clone https://github.com/nathandecaux/TractoPL.git
cd TractoPL
pixi install                 # editable install of the checkout
pixi run test
```

Without pixi: `pip install -e '.[dashboard,dev]' && python -m pytest`.

Dependencies are declared twice in [pyproject.toml](../pyproject.toml): `[project]` (PyPI names,
for pip) and `[tool.pixi.*]` (conda-forge names, for the pixi environments and the conda package).
Update both when adding a dependency, then run `pixi install` and commit `pixi.lock`.

## Releases

The `Release` workflow ([.github/workflows/release.yml](../.github/workflows/release.yml)) runs
the tests, builds the wheel, the sdist and the conda package (`pixi publish`, pixi-build-python
backend, `noarch`), creates the GitHub release and uploads the conda package to prefix.dev. The
version lives in `TractoPL/_version.py`; [packaging/bump_version.py](../packaging/bump_version.py)
also updates `[tool.pixi.package].version` in `pyproject.toml`.

| Action | Result |
|---|---|
| `git push` (fix, any commit) | tests only, nothing published |
| `pixi run release patch` (or `packaging/release.sh patch\|minor\|major\|X.Y.Z`) | bump + commit + tag `vX.Y.Z` + push → packages + GitHub release + prefix.dev |
| Actions › Release › *Run workflow*, `release = patch/minor/major` | same, from the GitHub page |
| Actions › Release › *Run workflow*, `release = none` | test packages, downloadable from the run page |

A tag with a suffix (`pixi run release 1.2.0-rc1`) makes a GitHub pre-release.

`pixi run version` prints the current version.

### prefix.dev setup (once)

1. Create the channel (e.g. `tractopl`) on [prefix.dev](https://prefix.dev).
2. In the channel settings, *Trusted Publishers*: add the GitHub repository
   `nathandecaux/TractoPL` and the workflow `release.yml`. No secret is needed then; otherwise
   create an API key and store it as the `PREFIX_API_KEY` repository secret.
3. In the GitHub repository, *Settings › Secrets and variables › Actions › Variables*: set
   `PREFIX_CHANNEL` to the channel name. Without this variable the upload is skipped (the `.conda`
   file is still attached to the GitHub release).

### Local builds

```shell
pixi run -e build wheel                   # dist/*.whl and dist/*.tar.gz
pixi publish --target-dir dist/conda      # dist/conda/tractopl-<version>-*.conda
```

## Code layout

- `data/`: BIDS indexing (`loader.py`: `Dataset`, `Subject`, `BIDSFile`), tractogram and MCM I/O
- `pipeline/`: processing steps and their command-line entry points
- `analysis/`: tractometry and population analyses, reports
- `atlasing/`: centroids and clustering
- `dashboard/`: Streamlit applications and their launcher (`apps.py`)
- `scripts/`: standalone utilities (tractogram conversion, transforms, QuickBundles)
- `configuration.py`: external tools and atlas manifests
- `config_profiles/`: atlas manifest example and HCP105 bundle mapping
- `_version.py`: version (single source)
