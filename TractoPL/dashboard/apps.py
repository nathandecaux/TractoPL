"""
Command-line entry points for the TractoPL web applications.

Each application has its own command (tractopl-gui-<name>), and `tractopl-gui`
starts a home page that lists them and starts each one on demand, in its own
process and on its own port.

Usage:
    tractopl-gui [--port 8500] [--no-browser]
    tractopl-gui-pipeline [--port N] [--no-browser] [streamlit options...]
    tractopl-gui-report [summary_results.csv] [--port N] [--no-browser]
"""

import argparse
import importlib.util
import os
import socket
import sys
from typing import List, NamedTuple, Optional


class App(NamedTuple):
    module: str
    kind: str  # 'streamlit' or 'panel'
    title: str
    icon: str
    description: str


APPS = {
    'pipeline': App(
        'TractoPL.dashboard.streamlit_launcher', 'streamlit', 'Pipeline launcher', '🚀',
        'Run the processing workflow step by step on a BIDS dataset, locally or in Apptainer, '
        'and check which outputs each participant has.'),
    'analysis': App(
        'TractoPL.analysis.streamlit_analysis', 'streamlit', 'Population analysis', '📊',
        'Build the population_analysis configuration (groups, correlations, confounds, '
        'ComBat...) and run it with live logs.'),
    'dashboard': App(
        'TractoPL.dashboard.streamlit_dashboard', 'streamlit', 'Tractometry dashboard', '🧠',
        'Explore bundle profiles and statistics of the tractometry derivatives of a dataset.'),
    'report': App(
        'TractoPL.analysis.report_analysis', 'panel', 'Results browser', '📋',
        'Filter and inspect the summary_results.csv written by the population analysis.'),
}


def command_name(name: str) -> str:
    return f'tractopl-gui-{name}'


def script_path(module: str) -> str:
    """Locate an app's source file without importing (and so running) it."""
    return importlib.util.find_spec(module).origin


def free_port(start: int) -> int:
    port = start
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(('0.0.0.0', port))
                return port
            except OSError:
                port += 1


def app_command(name: str, port: Optional[int] = None, browser: bool = True,
                extra: Optional[List[str]] = None) -> List[str]:
    app = APPS[name]
    extra = list(extra or [])
    if app.kind == 'streamlit':
        cmd = [sys.executable, '-m', 'streamlit', 'run', script_path(app.module),
               '--server.headless', 'false' if browser else 'true']
        if port is not None:
            cmd += ['--server.port', str(port)]
        return cmd + extra
    cmd = [sys.executable, '-m', app.module] + extra
    if port is not None:
        cmd += ['--port', str(port)]
    if not browser:
        cmd.append('--no-browser')
    return cmd


def _parse(description: str, argv: Optional[List[str]]):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--port', type=int, default=None, help='Port to serve on.')
    parser.add_argument('--no-browser', action='store_true',
                        help='Do not open a browser tab.')
    # Unknown arguments go to the application (streamlit options, CSV path...).
    return parser.parse_known_args(argv)


def run_app(name: str, argv: Optional[List[str]] = None):
    app = APPS[name]
    args, extra = _parse(f'{app.title}: {app.description}', argv)
    cmd = app_command(name, args.port, not args.no_browser, extra)
    os.execv(cmd[0], cmd)


def pipeline():
    run_app('pipeline')


def analysis():
    run_app('analysis')


def dashboard():
    run_app('dashboard')


def report():
    run_app('report')


def home(argv: Optional[List[str]] = None):
    args, extra = _parse('Home page giving access to all TractoPL applications.', argv)
    cmd = [sys.executable, '-m', 'streamlit', 'run',
           script_path('TractoPL.dashboard.streamlit_home'),
           '--server.port', str(args.port or free_port(8500)),
           '--server.headless', 'true' if args.no_browser else 'false'] + extra
    os.execv(cmd[0], cmd)


if __name__ == '__main__':
    home()
