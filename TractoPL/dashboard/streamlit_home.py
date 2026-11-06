"""
Home page of the TractoPL web applications.

Lists the applications declared in apps.py; each one is started on demand in
its own process and port, and stopped with this page.

Usage:
    tractopl-gui
"""

import atexit
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import streamlit as st

from TractoPL.dashboard.apps import APPS, app_command, command_name, free_port

LOG_DIR = Path(tempfile.gettempdir())
STARTUP_TIMEOUT = 60


@st.cache_resource
def processes():
    """name -> (Popen, port), shared by every session of this server."""
    procs = {}

    def stop_all():
        for proc, _ in procs.values():
            if proc.poll() is None:
                proc.terminate()

    atexit.register(stop_all)
    return procs


def log_path(name: str) -> Path:
    return LOG_DIR / f'{command_name(name)}.log'


def is_listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(('127.0.0.1', port)) == 0


def start(name: str):
    port = free_port(8501)
    with open(log_path(name), 'w') as log:
        proc = subprocess.Popen(app_command(name, port, browser=False),
                                stdout=log, stderr=subprocess.STDOUT)
    processes()[name] = (proc, port)
    deadline = time.time() + STARTUP_TIMEOUT
    while proc.poll() is None and not is_listening(port) and time.time() < deadline:
        time.sleep(0.5)


def stop(name: str):
    proc, _ = processes().pop(name)
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def app_url(port: int) -> str:
    """Same host as this page, so links also work from a remote browser."""
    host = st.context.headers.get('Host', 'localhost')
    if not host.endswith(']'):  # keep IPv6 literals such as [::1]
        host = host.rsplit(':', 1)[0]
    return f'http://{host}:{port}'


def log_tail(name: str, lines: int = 40) -> str:
    try:
        return '\n'.join(log_path(name).read_text(errors='replace').splitlines()[-lines:])
    except OSError:
        return ''


def app_card(name: str):
    app = APPS[name]
    entry = processes().get(name)
    with st.container(border=True):
        st.subheader(f'{app.icon} {app.title}')
        st.write(app.description)
        st.caption(f'Standalone: `{command_name(name)}`')

        if entry is None:
            if st.button('Start', key=f'start_{name}', type='primary'):
                with st.spinner(f'Starting {app.title}...'):
                    start(name)
                st.rerun()
            return

        proc, port = entry
        if proc.poll() is not None:
            st.error(f'Stopped (exit code {proc.returncode}).')
            with st.expander('Log'):
                st.code(log_tail(name) or '(empty)')
            if st.button('Restart', key=f'restart_{name}'):
                processes().pop(name)
                with st.spinner(f'Starting {app.title}...'):
                    start(name)
                st.rerun()
            return

        url = app_url(port)
        c1, c2 = st.columns(2)
        c1.link_button('Open', url, type='primary', width='stretch')
        if c2.button('Stop', key=f'stop_{name}', width='stretch'):
            stop(name)
            st.rerun()
        st.caption(f'Running on port {port} — {url}')
        with st.expander('Log'):
            st.code(log_tail(name) or '(empty)')


st.set_page_config(page_title='TractoPL', page_icon='🧠', layout='wide')
st.title('🧠 TractoPL')
st.write('Start an application, then open it in a new tab. '
         'Applications stop with this page.')

names = list(APPS)
for row in range(0, len(names), 2):
    for col, name in zip(st.columns(2), names[row:row + 2]):
        with col:
            app_card(name)

st.caption('Over SSH, forward the port of each application you open, '
           'in addition to the port of this page.')
