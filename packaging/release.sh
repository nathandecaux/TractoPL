#!/usr/bin/env bash
# Publish a release: bump the version, commit, tag and push. The CI (Release workflow)
# then runs the tests, builds the wheel and the conda package, creates the GitHub release
# and uploads the conda package to prefix.dev.
#   packaging/release.sh patch|minor|major|X.Y.Z[-rc1]      (or: pixi run release patch)
# A regular `git push` (fix) publishes nothing.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ $# -eq 1 ]] || { echo "usage: $0 patch|minor|major|X.Y.Z[-suffix]" >&2; exit 2; }
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Uncommitted changes: commit or stash them first." >&2; exit 1
fi
branch=$(git rev-parse --abbrev-ref HEAD)
suffix=""
arg=$1
if [[ "$arg" =~ ^v?([0-9]+\.[0-9]+\.[0-9]+)(-.+)$ ]]; then arg=${BASH_REMATCH[1]}; suffix=${BASH_REMATCH[2]}; fi
version=$(python3 packaging/bump_version.py "$arg")
tag="v$version$suffix"
files=(TractoPL/_version.py pyproject.toml)
if git rev-parse -q --verify "refs/tags/$tag" >/dev/null; then
  git checkout -- "${files[@]}"; echo "Tag $tag already exists." >&2; exit 1
fi
read -r -p "Release $tag from '$branch'? [y/N] " ok
[[ "$ok" == [yY]* ]] || { git checkout -- "${files[@]}"; echo "Cancelled."; exit 1; }
git commit -m "Release $tag" -- "${files[@]}"
git tag -a "$tag" -m "TractoPL $version$suffix"
git push origin "$branch" "$tag"
echo "Pushed $tag: https://github.com/nathandecaux/TractoPL/actions"
