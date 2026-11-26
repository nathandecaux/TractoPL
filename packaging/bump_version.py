"""Print or change the version.

The single source is TractoPL/_version.py; the version of the conda package
([tool.pixi.package] in pyproject.toml) is kept in sync by this script.

    python packaging/bump_version.py              -> print the current version
    python packaging/bump_version.py patch        -> 0.1.0 -> 0.1.1
    python packaging/bump_version.py minor|major
    python packaging/bump_version.py 1.4.0        -> explicit version
    python packaging/bump_version.py --check      -> fail if the two files disagree
Prints the new version on stdout.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = ROOT / "TractoPL" / "_version.py"
PYPROJECT = ROOT / "pyproject.toml"
VERSION_RE = re.compile(r'__version__ = "(\d+)\.(\d+)\.(\d+)"')
# `version = "X.Y.Z"` line of the [tool.pixi.package] table
PIXI_RE = re.compile(r'(\[tool\.pixi\.package\]\s*\n(?:[^\[\n].*\n|\n)*?version = ")(\d+\.\d+\.\d+)(")')


def current():
    m = VERSION_RE.search(VERSION_FILE.read_text())
    if not m:
        raise SystemExit(f'no __version__ = "X.Y.Z" in {VERSION_FILE}')
    return tuple(map(int, m.groups()))


def pixi_version():
    m = PIXI_RE.search(PYPROJECT.read_text())
    if not m:
        raise SystemExit(f'no version = "X.Y.Z" in [tool.pixi.package] of {PYPROJECT}')
    return m.group(2)


def write(new):
    VERSION_FILE.write_text(VERSION_RE.sub(f'__version__ = "{new}"', VERSION_FILE.read_text(), count=1))
    PYPROJECT.write_text(PIXI_RE.sub(rf"\g<1>{new}\g<3>", PYPROJECT.read_text(), count=1))


def main(argv):
    major, minor, patch = current()
    version = f"{major}.{minor}.{patch}"
    if not argv:
        print(version)
        return
    arg = argv[0].lstrip("v")
    if arg == "--check":
        if pixi_version() != version:
            raise SystemExit(f"TractoPL/_version.py ({version}) != [tool.pixi.package] ({pixi_version()})")
        print(version)
        return
    if arg == "major":
        new = f"{major + 1}.0.0"
    elif arg == "minor":
        new = f"{major}.{minor + 1}.0"
    elif arg == "patch":
        new = f"{major}.{minor}.{patch + 1}"
    elif re.fullmatch(r"\d+\.\d+\.\d+", arg):
        new = arg
    else:
        raise SystemExit(f"expected patch, minor, major, X.Y.Z or --check, got {argv[0]!r}")
    write(new)
    print(new)


if __name__ == "__main__":
    main(sys.argv[1:])
