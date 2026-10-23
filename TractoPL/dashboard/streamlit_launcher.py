"""
Streamlit launcher for the TractoPL command-line workflow.

One block per workflow step: its options, the inputs it needs and the outputs
it writes (checked in the dataset), the commands it will run, and a button to
run it. Commands run locally or inside an Apptainer image. See
examples/README.md for what each step reads and writes.

Usage:
    streamlit run TractoPL/dashboard/streamlit_launcher.py
"""

import copy
import glob
import json
import os
import signal
import socket
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import pandas as pd
import streamlit as st

from TractoPL.dashboard import job_runner
from TractoPL.data.loader import Dataset

CHECKOUT = Path(__file__).resolve().parents[2]
CONTAINER_CHECKOUT = '/opt/tractopl'
TRACTOPL_COMMANDS = [
    'tractopl-preprocessing',
    'tractopl-msmt-csd',
    'tractopl-mcm',
    'tractopl-bundle-seg',
    'tractopl-tractometry',
]

PREPROCESSING_STEPS = {
    'anima': 'animaPreprocessing (correction, brain mask)',
    'dti': 'DTI estimation',
    'fa': 'FA map',
}
MSMT_STEPS = [
    'response',
    'fod',
    'fixels',
    'fixels2peaks',
    'fixel_density',
    'ifod2_tracto',
]
COMPARTMENT_TYPES = {1: 'Stick', 2: 'Zeppelin', 3: 'Tensor', 4: 'NODDI', 5: 'DDI'}
ML_MODES = {0: 'Marginal likelihood', 1: 'Profile likelihood', 2: 'Variable projection'}
OPTIMIZERS = ['levenberg', 'bobyqa', 'ccsaq', 'bfgs']

# Local workdir where the launcher remembers the settings of previous sessions.
WORKDIR = Path(os.environ.get('TRACTOPL_LAUNCHER_WORKDIR', Path(__file__).resolve().parent / 'launcher_workdir'))
SETTINGS_FILE = WORKDIR / 'settings.json'


# =============================================================================
# Dataset
# =============================================================================

@st.cache_resource(show_spinner='Indexing dataset...')
def load_dataset(db_root: str) -> Dataset:
    """Index a BIDS dataset once per root; cleared after each run."""
    return Dataset(db_root)


class Check(NamedTuple):
    """A file expected in the dataset for one participant."""
    label: str
    filters: Optional[Dict[str, Any]] = None
    min_count: int = 1
    pattern: Optional[str] = None  # glob for files the index skips, e.g. sourcedata


@st.cache_data(show_spinner=False, max_entries=512)
def _cached_counts(_dataset: Dataset, index_id: int, check: Check, subjects: Tuple[str, ...]) -> Dict[str, int]:
    """Counts for one dataset index; a new index (Refresh dataset) gets a new id."""
    return count_matches(_dataset, check, list(subjects))


def count_matches(dataset: Dataset, check: Check, subjects: List[str]) -> Dict[str, int]:
    """Count the files matching a check for each participant."""
    if check.pattern:
        return {
            subject: len(glob.glob(check.pattern.format(root=dataset.db_root, sub=subject), recursive=True))
            for subject in subjects
        }
    matches = dataset.get_global(to_dataframe=True, **check.filters)
    if len(matches) == 0:
        return {subject: 0 for subject in subjects}
    counts = matches['subject'].value_counts()
    return {subject: int(counts.get(subject, 0)) for subject in subjects}


def check_table(dataset: Dataset, checks: List[Check], subjects: List[str]) -> pd.DataFrame:
    """Return one row per participant and one boolean column per check."""
    table = pd.DataFrame(index=subjects)
    for check in checks:
        counts = _cached_counts(dataset, id(dataset), check, tuple(subjects))
        table[check.label] = [counts[subject] >= check.min_count for subject in subjects]
    return table


def complete_subjects(table: pd.DataFrame) -> List[str]:
    return table.index[table.all(axis=1)].tolist() if len(table.columns) else list(table.index)


# Files written by one step and read by the next ones. `!*` means "entity absent".

def preproc_dwi():
    return Check('Preprocessed DWI', dict(pipeline='preprocessing', suffix='dwi', desc='preproc',
                                          extension='nii.gz', model='!*'))


def preproc_bvec():
    return Check('Corrected bvec', dict(pipeline='preprocessing', desc='preproc', extension='bvec'))


def brain_mask():
    return Check('Brain mask', dict(pipeline='preprocessing', suffix='mask', label='brain', space='B0'))


def dti_tensors():
    return Check('DTI', dict(pipeline='preprocessing', model='DTI', metric='!*', extension='nii.gz'))


def fa_map():
    return Check('FA', dict(pipeline='preprocessing', metric='FA', extension='nii.gz'))


def raw_bval():
    return Check('Raw bval', dict(scope='raw', extension='bval'))


def fixel_density(pipeline):
    return Check('Fixel density', dict(pipeline=pipeline, suffix='density', label='WM',
                                       desc='fixels2peaks', extension='nii.gz'))


def whole_brain_tractogram(pipeline='msmt_csd'):
    return Check('Whole-brain tractogram', dict(pipeline=pipeline, suffix='tracto', algo='ifod2',
                                                label='brain', extension='tck'))


def mcm_archive(pipeline):
    return Check('MCM archive', dict(pipeline=pipeline, extension='mcmx'))


def segmented_bundles(pipeline):
    return Check('Segmented bundles', dict(pipeline=pipeline, datatype='tracto', suffix='tracto',
                                           extension='trk'))


def subject_centroids(pipeline, clustering='centroid'):
    return Check('Centroids', dict(pipeline=pipeline, datatype='atlas', suffix='centroids',
                                   clustering=clustering, extension='vtk'))


def metric_bundles(pipeline):
    return Check('MCM bundles (VTK)', dict(pipeline=pipeline, datatype='tracto', suffix='tracto',
                                           extension='vtk'))


def step_io(step: str, params: Dict[str, Any]):
    """Return the (inputs, outputs) checks of a step with the current options."""
    if step == 'preprocessing':
        steps = params['preprocessing_steps']
        inputs: List[Check] = []
        outputs: List[Check] = []
        # Inputs of a sub-step that an earlier selected sub-step writes are not checked.
        if 'anima' in steps:
            inputs += [
                Check('Raw DWI', dict(scope='raw', suffix='dwi', extension='nii.gz', desc='!*')),
                Check('Raw bval/bvec', dict(scope='raw', suffix='dwi', extension=['.bval', '.bvec']), 2),
                Check('T1w', dict(scope='raw', suffix='T1w', extension='nii.gz')),
                Check('DICOMs', pattern='{root}/sub-{sub}/sourcedata/sub-{sub}_dicoms/**/*'),
            ]
            if params['with_reversed_b0']:
                inputs.append(Check('Reversed B0', dict(scope='raw', desc='b0reversed', extension='nii.gz')))

            outputs += [preproc_dwi(), preproc_bvec(), brain_mask()]
        if 'dti' in steps:
            if 'anima' not in steps:
                inputs += [preproc_dwi(), preproc_bvec(), raw_bval()]
            outputs.append(dti_tensors())
        if 'fa' in steps:
            if 'dti' not in steps:
                inputs.append(dti_tensors())
            outputs.append(fa_map())
        return inputs, outputs

    if step == 'msmt_csd':
        pipeline = params['msmt_pipeline']
        tissues = ['WM', 'GM', 'CSF']
        by_step = {
            'response': Check('Responses', dict(pipeline=pipeline, suffix='response', label=tissues), 3),
            'fod': Check('FODs', dict(pipeline=pipeline, suffix='fod', label='WM')),
            'fixels': Check('Fixels', dict(pipeline=pipeline, extension='fixel', label='WM')),
            'fixels2peaks': Check('Fixel peaks', dict(pipeline=pipeline, suffix='peaks', label='WM',
                                                      desc='fixels2peaks')),
            'fixel_density': fixel_density(pipeline),
            'ifod2_tracto': whole_brain_tractogram(pipeline),
        }
        inputs = [preproc_dwi(), preproc_bvec(), brain_mask(), raw_bval()]
        return inputs, [by_step[name] for name in params['msmt_steps']]

    if step == 'mcm_estimation':
        inputs = [preproc_dwi(), preproc_bvec(), brain_mask(), raw_bval()]
        if not params['model_selection']:
            inputs.append(fixel_density('*'))
        return inputs, [mcm_archive(params['mcm_pipeline'])]

    if step == 'bundle_seg':
        inputs = [fa_map(), whole_brain_tractogram()]
        outputs = [
            Check('Transforms', dict(pipeline='bundle_seg', suffix='xfm',
                                     desc=['0GenericAffine', '1InverseWarp']), 2),
            subject_centroids('bundle_seg'),
            segmented_bundles('bundle_seg'),
        ]
        return inputs, outputs

    if step == 'mcm_projection':
        inputs = [
            mcm_archive(params['projection_mcm_pipeline']),
            fa_map(),
            segmented_bundles(params['projection_bundle_pipeline']),
        ]
        return inputs, [metric_bundles(params['projection_mcm_pipeline'])]

    if step == 'tractometry':
        inputs = [
            metric_bundles(params['tractometry_mcm_pipeline']),
            subject_centroids(params['tractometry_bundle_pipeline'], params['clustering_method']),
            fa_map(),
        ]
        outputs = [Check('Metric CSV', dict(pipeline=params['tractometry_pipeline'],
                                            datatype='metric', suffix='mean'))]
        return inputs, outputs

    raise ValueError(f"Unknown step: {step}")


# =============================================================================
# Commands
# =============================================================================

def build_command(step: str, subject: str, params: Dict[str, Any]) -> List[str]:
    """Return the tractopl argv of one workflow step for one participant."""
    common = ['--subject', subject, '--db-root', params['db_root']]

    if step == 'preprocessing':
        command = ['tractopl-preprocessing'] + common
        if 'anima' in params['preprocessing_steps'] and not params['with_reversed_b0']:
            command.append('--without-reversed-b0')
        if not params['flip_bvec_checkbox']:
            command.append('--no-flip-bvec')
        else:
            command += ['--flip-bvec', str(params['flip_bvec'])]
        for preprocessing_step in params['preprocessing_steps']:
            command += ['--step', preprocessing_step]
        return command

    if step == 'msmt_csd':
        command = ['tractopl-msmt-csd'] + common + ['--pipeline', params['msmt_pipeline']]
        for msmt_step in params['msmt_steps']:
            command += ['--step', msmt_step]
        command += ['--n-streamlines', str(params['n_streamlines'])]
        if params['force']:
            command.append('--force')
        return command

    if step == 'mcm_estimation':
        command = ['tractopl-mcm', 'estimation'] + common + [
            '--pipeline', params['mcm_pipeline'],
            '--tensor-model', str(params['tensor_model']),
            '--free-water' if params['free_water'] else '--no-free-water',
            '--restricted-water' if params['restricted_water'] else '--no-restricted-water',
            '--ml-mode', str(params['ml_mode']),
            '--optimizer', params['optimizer'],
        ]
        if params['stanisz']:
            command.append('--stanisz')
        if params['stationary_water']:
            command.append('--stationary-water')
        if params['model_selection']:
            command.append('--model-selection')
        else:
            command += ['--n-comparts', str(params['n_comparts'])]
        if params['n_threads']:
            command += ['--n-threads', str(params['n_threads'])]
        return command

    if step == 'bundle_seg':
        command = ['tractopl-bundle-seg', 'segmentation'] + common + ['--atlas', params['atlas']]
        if params['keep_intermediate']:
            command.append('--keep-intermediate')
        return command

    if step == 'mcm_projection':
        return ['tractopl-mcm', 'projection'] + common + [
            '--pipeline', params['projection_mcm_pipeline'],
            '--bundle-pipeline', params['projection_bundle_pipeline'],
        ]

    if step == 'tractometry':
        return ['tractopl-tractometry', 'association'] + common + [
            '--atlas', params['atlas'],
            '--pipeline', params['tractometry_pipeline'],
            '--mcm-pipeline', params['tractometry_mcm_pipeline'],
            '--bundle-pipeline', params['tractometry_bundle_pipeline'],
            '--clustering-method', params['clustering_method'],
            '--n-pts', str(params['n_pts']),
        ]

    raise ValueError(f"Unknown step: {step}")
from pathlib import Path
# home = Path.home()

def apptainer_prefix(execution: Dict[str, Any]) -> List[str]:
    """Return `apptainer exec` with the dataset, atlas, and checkout mounted."""
    # Host tempdir (honours $TMPDIR) is bound and exported, since --cleanenv drops it.
    tmpdir = tempfile.gettempdir()
    binds = [execution['db_root'], tmpdir]
    if execution['atlas']:
        binds.append(os.path.dirname(os.path.abspath(execution['atlas'])))
    # --pwd keeps the host working directory from shadowing the mounted checkout
    # on sys.path (the image wrappers use `python -m`).
    prefix = ['apptainer', 'exec', '--cleanenv', '--writable-tmpfs', '--pwd', tmpdir, '--env', f'TMPDIR={tmpdir}', '--bind', Path.home().as_posix()+':'+Path.home().as_posix(),'--bind','/home/ndecaux:/home/ndecaux']
    for path in dict.fromkeys(binds):
        prefix += ['--bind', f'{path}:{path}']
    prefix += [
        '--bind', f"{execution['checkout']}:{CONTAINER_CHECKOUT}",
        '--env', f'PYTHONPATH={CONTAINER_CHECKOUT}',
        execution['image'],
    ]
    return prefix


def wrap_command(command: List[str], execution: Dict[str, Any]) -> List[str]:
    """Run a tractopl command locally or inside the Apptainer image."""
    if execution['mode'] == 'local':
        return command
    prefix = apptainer_prefix(execution)
    if execution['pip_install']:
        # Installs every tractopl-* entry point; bundle-seg calls some that the
        # image wrappers do not provide.
        install = f'pip install --quiet -e {CONTAINER_CHECKOUT} --no-deps >/dev/null && exec "$@"'
        return prefix + ['bash', '-c', install, '_'] + command
    return prefix + command


def probe_image(execution: Dict[str, Any]) -> Dict[str, Any]:
    """Load the image: check the checkout imports and list tractopl commands."""
    script = (
        'python -c "import TractoPL; print(TractoPL.__file__)" || exit 1; '
        f'for c in {" ".join(TRACTOPL_COMMANDS)}; do command -v "$c" >/dev/null && echo "CMD $c"; done'
    )
    result = subprocess.run(
        apptainer_prefix(execution) + ['bash', '-c', script],
        capture_output=True, text=True, timeout=300,
    )
    lines = result.stdout.splitlines()
    return {
        'key': (execution['image'], execution['checkout']),
        'ok': result.returncode == 0,
        'module': next((line for line in lines if line.endswith('.py')), ''),
        'commands': [line[4:] for line in lines if line.startswith('CMD ')],
        'error': result.stderr.strip()[-2000:],
    }


def submit_job(log_root: str, workers: int, tasks: List[Dict[str, Any]]) -> str:
    """Start a detached runner; it keeps going if the dashboard is closed."""
    job_id = time.strftime('%Y%m%d-%H%M%S')
    log_root_path = Path(log_root).expanduser().resolve()
    job_dir = log_root_path / job_id
    suffix = 1
    while job_dir.exists():
        suffix += 1
        job_dir = log_root_path / f'{job_id}-{suffix}'
    job_dir.mkdir(parents=True)
    (job_dir / 'job.json').write_text(json.dumps({'workers': workers, 'tasks': tasks}, indent=1))
    # Locate the runner through the imported package, not the launcher's own path,
    # so it works wherever streamlit was started from.
    runner = Path(job_runner.__file__).resolve()
    with open(job_dir / 'runner.log', 'w') as runner_log:
        process = subprocess.Popen(
            [sys.executable, str(runner), str(job_dir)],
            stdout=runner_log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True, cwd=str(CHECKOUT),
        )
    # Reap the runner when it exits, otherwise it stays a zombie that looks alive.
    threading.Thread(target=process.wait, daemon=True).start()
    return job_dir.name


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def runner_alive(state: Dict[str, Any]) -> Optional[bool]:
    """Whether the runner of a job still runs; None when it cannot be told from here.

    The pid is checked only on the host that started the runner, and must still
    be a job runner (not a zombie or a recycled pid). Elsewhere the heartbeat
    the runner refreshes every few seconds is used.
    """
    pid = state.get('pid')
    if pid and state.get('host', socket.gethostname()) == socket.gethostname():
        try:
            fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            cmdline = Path(f'/proc/{pid}/cmdline').read_bytes()
        except OSError:
            return False
        return fields[0] != 'Z' and b'job_runner' in cmdline
    heartbeat = state.get('heartbeat')
    if heartbeat is None:
        return None
    return time.time() - heartbeat < 6 * job_runner.HEARTBEAT_SECONDS


def tail(path: Path, max_lines: int = 10) -> str:
    """Last lines of a log; reread only when the file changed."""
    try:
        stat = path.stat()
    except OSError:
        return '(no log yet)'
    return _tail(str(path), stat.st_size, stat.st_mtime_ns, max_lines)


@st.cache_data(show_spinner=False, max_entries=256)
def _tail(path: str, size: int, mtime_ns: int, max_lines: int, max_bytes: int = 256 * 1024) -> str:
    """Read at most `max_bytes` from the end of a log."""
    path = Path(path)
    try:
        with path.open('rb') as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            data = handle.read()
    except OSError:
        return '(no log yet)'
    lines = data.decode(errors='replace').splitlines()
    if size > max_bytes:
        lines = lines[1:]  # first line is probably cut
    return '\n'.join(lines[-max_lines:])


STATUS_ICONS = {'pending': '⏳', 'running': '🔄', 'success': '✅', 'failed': '❌',
                'skipped': '⏭️', 'cancelled': '🛑', 'interrupted': '⚠️', 'starting': '🔄',
                'cancelling': '🛑'}
ACTIVE = ('starting', 'running', 'cancelling')
# Statuses removed by "Delete failed jobs" and relaunched by "Relaunch".
FINISHED_BAD = ('failed', 'cancelled', 'interrupted')
RETRY_TASKS = ('failed', 'skipped', 'cancelled', 'interrupted')
STARTUP_GRACE_SECONDS = 60


class Job(NamedTuple):
    path: Path
    status: str
    state: Dict[str, Any]
    tasks: List[Dict[str, Any]]


def load_job(job_dir: Path) -> Job:
    """Read a job directory and work out its real status.

    A runner killed without cleaning up (reboot, kill -9) leaves "running" in
    state.json: such jobs and their unfinished tasks are shown as interrupted.
    """
    state = read_json(job_dir / 'state.json', None)
    if state is None:
        try:
            age = time.time() - (job_dir / 'job.json').stat().st_mtime
        except OSError:
            age = 0
        status = 'starting' if age < STARTUP_GRACE_SECONDS else 'interrupted'
        spec = read_json(job_dir / 'job.json', {})
        tasks = [{'subject': t['subject'], 'step': t['step'], 'status': 'pending' if status == 'starting'
                  else 'interrupted', 'log': job_runner.task_log_name(t)} for t in spec.get('tasks', [])]
        return Job(job_dir, status, {}, tasks)
    status = state.get('status', 'running')
    tasks = [dict(task) for task in state.get('tasks', [])]
    if status == 'running':
        if runner_alive(state) is False:
            status = 'interrupted'
            for task in tasks:
                if task['status'] in ('pending', 'running'):
                    task['status'] = 'interrupted'
        elif job_runner.cancel_file(job_dir).exists():
            status = 'cancelling'
    return Job(job_dir, status, state, tasks)


def list_jobs(log_root: str) -> List[Job]:
    """All jobs of a log directory; shared by the job watcher and the Jobs page."""
    return [Job(Path(path), status, state, tasks) for path, status, state, tasks in _list_jobs(log_root)]


@st.cache_data(ttl=2, show_spinner=False)
def _list_jobs(log_root: str) -> List[tuple]:
    # Plain tuples: the script runs as __main__, so Job itself cannot be pickled.
    root = Path(log_root)
    if not root.is_dir():
        return []
    dirs = sorted((d for d in root.iterdir() if (d / 'job.json').is_file()), reverse=True)
    return [(str(job.path), job.status, job.state, job.tasks) for job in map(load_job, dirs)]


def request_cancel(job: Job, subject: Optional[str] = None) -> None:
    """Ask the runner to stop the job (or one participant).

    The cancel file works whatever host runs the job; the signal is a faster
    path when the runner runs on this host.
    """
    job_runner.cancel_file(job.path, subject).touch()
    if subject is None and job.state.get('host', socket.gethostname()) == socket.gethostname() \
            and runner_alive(job.state):
        try:
            os.kill(job.state['pid'], signal.SIGTERM)
        except (OSError, KeyError):
            pass


def delete_job(job: Job) -> None:
    shutil.rmtree(job.path, ignore_errors=True)


def relaunch_job(job: Job, log_root: str) -> Optional[str]:
    """Start a new job with the tasks that did not succeed, in their order."""
    spec = read_json(job.path / 'job.json', {})
    retry = {(t['subject'], t['step']) for t in job.tasks if t['status'] in RETRY_TASKS}
    tasks = [t for t in spec.get('tasks', []) if (t['subject'], t['step']) in retry]
    if not tasks:
        return None
    return submit_job(log_root, int(spec.get('workers', 1)), tasks)


def format_duration(task: Dict[str, Any]) -> str:
    """Duration of a finished task; start time of a running one (it does not go stale)."""
    started = task.get('started')
    if not started:
        return ''
    if not task.get('ended'):
        return f" · since {time.strftime('%H:%M', time.localtime(started))}"
    seconds = int(task['ended'] - started)
    hours, rest = divmod(seconds, 3600)
    return f' · {hours}h{rest // 60:02d}' if hours else f' · {rest // 60}m{rest % 60:02d}s'


# =============================================================================
# Notifications
# =============================================================================

NOTIFICATION_HISTORY = 50
TERMINAL_TASK = {'success', 'failed', 'cancelled', 'interrupted'}
# Job changes older than this are not announced (e.g. jobs from before the event store).
EVENT_MAX_AGE_SECONDS = 15 * 60
EVENT_KEEP_SECONDS = 7 * 24 * 3600


def notify(message: str, icon: str = 'ℹ️', duration: str = 'short') -> None:
    """Queue a toast; it survives the st.rerun() that usually follows an action."""
    st.session_state.setdefault('pending_toasts', []).append((message, icon, duration))
    history = st.session_state.setdefault('notification_history', [])
    history.insert(0, (time.strftime('%H:%M:%S'), icon, message))
    del history[NOTIFICATION_HISTORY:]


def flush_notifications() -> None:
    """Show the queued notifications, grouped in one toast when there are several."""
    pending = st.session_state.pop('pending_toasts', [])
    if len(pending) == 1:
        st.toast(pending[0][0], icon=pending[0][1], duration=pending[0][2])
    elif pending:
        important = [icon for _, icon, duration in pending if duration == 'long']
        st.toast(f'**{len(pending)} notifications**  \n'
                 + '  \n'.join(f'{icon} {message}' for message, icon, _ in pending),
                 icon=important[0] if important else pending[0][1],
                 duration='long' if important else 'short')


def jobs_changed() -> None:
    """Forget the cached job list after an action on the log directory."""
    _list_jobs.clear()


def job_summary(job: Job) -> str:
    counts = {status: sum(t['status'] == status for t in job.tasks) for status in STATUS_ICONS}
    parts = [f"{counts['success']}/{len(job.tasks)} succeeded"]
    parts += [f'{counts[status]} {status}' for status in ('failed', 'cancelled', 'interrupted') if counts[status]]
    return ', '.join(parts)


# Event store: one small JSON file per job change in <log_root>/.notifications.
# Several dashboards (possibly on other hosts sharing the directory) look at the
# same jobs: an event is created with os.link, which fails if it already exists,
# so each change is recorded exactly once whoever sees it first. Each session
# then shows the events it has not shown yet.

def events_dir(log_root: str) -> Path:
    return Path(log_root) / '.notifications'


def record_event(log_root: str, key: str, message: str, icon: str, duration: str) -> bool:
    """Record an event once; False when it already exists (or cannot be written)."""
    directory = events_dir(log_root)
    path = directory / f'{key}.json'
    if path.exists():
        return False
    tmp = directory / f'.{key}.{socket.gethostname()}.{os.getpid()}.{threading.get_ident()}.tmp'
    try:
        directory.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps({'time': time.time(), 'message': message, 'icon': icon,
                                   'duration': duration}))
        os.link(tmp, path)  # atomic, and fails if another dashboard was first
        return True
    except OSError:
        return False
    finally:
        tmp.unlink(missing_ok=True)


def record_job_events(log_root: str, jobs: List[Job], known: set) -> None:
    """Turn the recent changes of the jobs into events; `known` ones are skipped."""
    now = time.time()

    def record(key: str, *event: str) -> None:
        if key not in known:
            record_event(log_root, key, *event)

    for job in jobs:
        name = job.path.name
        try:
            changed = max(path.stat().st_mtime for path in (job.path / 'state.json', job.path / 'job.json')
                          if path.exists())
        except (OSError, ValueError):
            continue
        if now - changed > EVENT_MAX_AGE_SECONDS:
            continue
        record(f'{name}__started', f'Job {name} started ({len(job.tasks)} task(s)).', '🚀', 'short')
        for task in job.tasks:
            status = task['status']
            if status in TERMINAL_TASK:
                record(f"{name}__{task['subject']}__{task['step']}__{status}",
                       f"sub-{task['subject']} · {task['step']}: {status}", STATUS_ICONS[status],
                       'short' if status == 'success' else 'long')
        if job.status not in ACTIVE:
            record(f'{name}__job__{job.status}', f'Job {name} {job.status}: {job_summary(job)}.',
                   STATUS_ICONS.get(job.status, '⚠️'), 'long')


def prune_events(log_root: str) -> None:
    now = time.time()
    for path in events_dir(log_root).glob('*.json'):
        try:
            if now - path.stat().st_mtime > EVENT_KEEP_SECONDS:
                path.unlink(missing_ok=True)
        except OSError:
            pass


def seen_events(log_root: str) -> set:
    """Events this session has already shown (or chose not to show)."""
    root, seen = st.session_state.get('seen_events', (None, None))
    if root != log_root:
        # A new session (or log directory) starts from now: past events are not replayed.
        prune_events(log_root)
        seen = {path.stem for path in events_dir(log_root).glob('*.json')}
        st.session_state['seen_events'] = (log_root, seen)
    return seen


def announce(log_root: str, key: str, message: str, icon: str = '🚀', duration: str = 'short') -> None:
    """Record an event caused by this session and show it here right away."""
    record_event(log_root, key, message, icon, duration)
    seen_events(log_root).add(key)
    notify(message, icon, duration)


def show_new_events(log_root: str) -> None:
    seen = seen_events(log_root)
    events = []
    for path in events_dir(log_root).glob('*.json'):
        if path.stem in seen:
            continue
        event = read_json(path, None)
        seen.add(path.stem)
        if isinstance(event, dict):
            events.append(event)
    for event in sorted(events, key=lambda event: event.get('time', 0)):
        notify(event.get('message', ''), event.get('icon', 'ℹ️'), event.get('duration', 'short'))


def job_watcher(log_root: str) -> None:
    """Sidebar summary of the jobs; notifies their changes on every page.

    Runs every 5 s: everything that happened in between comes as one toast.
    """
    jobs = list_jobs(log_root)
    root, seen = st.session_state.get('seen_events', (None, None))
    record_job_events(log_root, jobs, seen if root == log_root else set())
    show_new_events(log_root)
    flush_notifications()
    active = [job for job in jobs if job.status in ACTIVE]
    running_tasks = sum(t['status'] == 'running' for job in active for t in job.tasks)
    st.caption(f'🔄 {len(active)} active job(s), {running_tasks} running task(s)' if active
               else 'No active job.')
    history = st.session_state.get('notification_history', [])
    with st.expander(f'Notifications ({len(history)})'):
        if not history:
            st.caption('Nothing yet.')
        for stamp, icon, message in history:
            st.caption(f'{stamp} {icon} {message}')
        if history and st.button('Clear notifications', key='clear_notifications'):
            st.session_state['notification_history'] = []
            st.rerun()


def jobs_signature(jobs: List[Job]) -> tuple:
    return tuple((job.path.name, job.status, tuple(t['status'] for t in job.tasks)) for job in jobs)


def watch_jobs(log_root: str) -> None:
    """Redraw the Jobs page only when a job or task status changed.

    The logs of running tasks refresh on their own (see render_task).
    """
    if jobs_signature(list_jobs(log_root)) != st.session_state.get('jobs_rendered'):
        st.rerun()


def render_jobs(log_root: str) -> None:
    """One card per participant/step, reloaded from the log directory."""
    jobs = list_jobs(log_root)
    st.session_state['jobs_rendered'] = jobs_signature(jobs)
    bad_jobs = [job for job in jobs if job.status in FINISHED_BAD]

    toolbar = st.columns([2, 2, 3])
    show_closed = toolbar[0].checkbox('Show closed cards', value=False, key='show_closed')
    hide_success = toolbar[1].checkbox('Hide successful jobs', value=False, key='hide_success')
    with toolbar[2].popover(f'🗑️ Delete failed / interrupted jobs ({len(bad_jobs)})',
                            disabled=not bad_jobs):
        st.write(f'Delete {len(bad_jobs)} job folder(s) and their logs from {log_root}?')
        st.caption(', '.join(job.path.name for job in bad_jobs))
        if st.button('Delete', type='primary', key='delete_bad_jobs'):
            for job in bad_jobs:
                delete_job(job)
            jobs_changed()
            notify(f'Deleted {len(bad_jobs)} failed / interrupted job(s).', '🗑️')
            st.rerun()

    shown = [job for job in jobs if not (hide_success and job.status == 'success')]
    if not shown:
        st.caption(f'No job found in {log_root}.')
        return
    job = select_job(shown)
    if job is not None:
        render_job(job, log_root, show_closed)


def select_job(jobs: List[Job]) -> Optional[Job]:
    """One compact row per job; the clicked one is detailed below the table.

    The selection is remembered by job name: a new job shifts the rows, so the
    table gets a new key (its row selection is reset) whenever its rows change.
    """
    names = [job.path.name for job in jobs]
    rows = []
    for job in jobs:
        counts = {status: sum(t['status'] == status for t in job.tasks) for status in STATUS_ICONS}
        rows.append({
            'Status': f"{STATUS_ICONS.get(job.status, '⚠️')} {job.status}",
            'Job': job.path.name,
            'Done': f"{counts['success']}/{len(job.tasks)}",
            'Running': counts['running'],
            'Failed': counts['failed'] + counts['interrupted'],
            'Participants': ', '.join(dict.fromkeys(f"sub-{t['subject']}" for t in job.tasks)),
        })

    key = f'jobs_table_{abs(hash(tuple(names)))}'

    def remember_selection() -> None:
        selected = st.session_state[key].selection.rows
        if selected:
            st.session_state['selected_job'] = names[selected[0]]

    st.dataframe(pd.DataFrame(rows), hide_index=True, on_select=remember_selection,
                 selection_mode='single-row', key=key, height=min(38 + 35 * len(rows), 300))
    selected = st.session_state.get('selected_job')
    if selected not in names:
        # Default to the most recent job.
        selected = names[0]
    return jobs[names.index(selected)]


def lazy_expander(label: str, key: str, expanded: bool):
    """Expander whose content runs only while it is open.

    Its label and initial state must stay the same across reruns, otherwise
    Streamlit sees a new widget and collapses it: the initial state is frozen
    on first display.
    """
    expanded = st.session_state.setdefault(f'{key}_initial', expanded)
    return st.expander(label, expanded=expanded, key=key, on_change='rerun')


def render_job(job: Job, log_root: str, show_closed: bool) -> None:
    name = job.path.name
    counts = {status: sum(t['status'] == status for t in job.tasks) for status in STATUS_ICONS}
    st.markdown(
        f"#### {STATUS_ICONS.get(job.status, '⚠️')} Job {name} — {job.status} "
        f"({counts['success']}/{len(job.tasks)} done, {counts['running']} running, "
        f"{counts['failed']} failed)"
    )
    closed = set(read_json(job.path / 'closed.json', []))
    st.caption(f"{job.path}" + (f" · host {job.state['host']}, pid {job.state.get('pid')}"
                                 if job.state.get('host') else ''))
    actions = st.columns(4)
    if job.status in ('running', 'starting') and actions[0].button('Cancel job', key=f'cancel_{name}'):
        request_cancel(job)
        jobs_changed()
        notify(f'Cancellation requested for job {name}.', '🛑')
        st.rerun()
    if job.status not in ACTIVE:
        if any(t['status'] in RETRY_TASKS for t in job.tasks) and \
                actions[1].button('Relaunch unfinished tasks', key=f'relaunch_{name}'):
            new_job = relaunch_job(job, log_root)
            if new_job:
                announce(log_root, f'{new_job}__started', f'Job {new_job} started: unfinished tasks of {name}.')
            jobs_changed()
            st.rerun()
        if actions[2].button('Delete job', key=f'delete_{name}'):
            delete_job(job)
            jobs_changed()
            notify(f'Job {name} deleted.', '🗑️')
            st.rerun()
    if job.state.get('error'):
        st.error('The runner crashed.')
        st.code(job.state['error'], language='text')
    if job.status == 'interrupted':
        st.warning('The runner stopped without finishing (killed or machine restarted).')
    if not job.tasks or job.status == 'interrupted' and not job.state:
        runner_log = lazy_expander('Runner log', f'runner_log_{name}', False)
        with runner_log:
            if runner_log.open:
                st.code(tail(job.path / 'runner.log'), language='text')

    for task in job.tasks:
        card_id = f"{task['subject']}_{task['step']}"
        if card_id in closed and not show_closed:
            continue
        render_task(job, task, card_id, closed)


def render_task(job: Job, task: Dict[str, Any], card_id: str, closed: set) -> None:
    name = job.path.name
    with st.container(border=True):
        title, cancel, close = st.columns([6, 1, 1])
        code = task.get('returncode')
        title.markdown(
            f"{STATUS_ICONS.get(task['status'], '⚠️')} **sub-{task['subject']}** · {task['step']} · "
            f"{task['status']}" + (f' (exit {code})' if code not in (None, 0) else '')
            + format_duration(task)
        )
        if job.status == 'running' and task['status'] in ('pending', 'running'):
            if job_runner.cancel_file(job.path, task['subject']).exists():
                cancel.caption('Cancelling…')
            elif cancel.button('Cancel', key=f'cancel_{name}_{card_id}',
                               help='Stops this participant (this step and the next ones).'):
                request_cancel(job, task['subject'])
                notify(f"Cancellation requested for sub-{task['subject']} in job {name}.", '🛑')
                st.rerun()
        if card_id in closed:
            if close.button('Reopen', key=f'open_{name}_{card_id}'):
                write_closed(job.path, closed - {card_id})
                notify(f"Card sub-{task['subject']} · {task['step']} reopened.", '📂')
                st.rerun()
        elif close.button('Close', key=f'close_{name}_{card_id}'):
            write_closed(job.path, closed | {card_id})
            notify(f"Card sub-{task['subject']} · {task['step']} closed.", '📁')
            st.rerun()
        log_path = job.path / task['log']
        if task['status'] in ('pending', 'skipped') and not log_path.is_file():
            return
        logs = lazy_expander('Logs', f'logs_{name}_{card_id}', task['status'] in ('running', 'failed'))
        with logs:
            if not logs.open:
                return
            if task['status'] == 'running':
                st.fragment(run_every=5)(show_log)(log_path)
            else:
                show_log(log_path)
            if log_path.is_file():
                st.download_button('Download full log', log_path.read_bytes,
                                   file_name=f'{name}_{task["log"]}',
                                   key=f'dl_{name}_{card_id}', on_click='ignore')


def show_log(log_path: Path) -> None:
    # tail() rereads the file only when its size or mtime changed.
    st.code(tail(log_path), language='text')


def write_closed(job_dir: Path, closed: set) -> None:
    (job_dir / 'closed.json').write_text(json.dumps(sorted(closed)))


def pipeline_selector(label: str, default: str, key: str) -> str:
    """Select a derivative pipeline, allowing one that does not exist yet."""
    preferred = st.session_state.get(key) or default
    options = sorted(set(DERIVATIVES) | {default, preferred})
    st.session_state[key] = preferred
    return st.selectbox(label, options, key=key)


# =============================================================================
# Remembered settings
# =============================================================================

def _text(value: Any) -> bool:
    return isinstance(value, str)


def _flag(value: Any) -> bool:
    return isinstance(value, bool)


def _choice(options) -> Any:
    options = list(options)
    return lambda value: value in options and not isinstance(value, bool)


def _subset(options) -> Any:
    options = list(options)
    return lambda value: isinstance(value, list) and all(item in options for item in value)


def _number(low: int, high: int) -> Any:
    return lambda value: isinstance(value, int) and not isinstance(value, bool) and low <= value <= high


# Widget key -> validator; a saved value is only restored when it is still valid.
COMMON_SETTINGS = {
    'db_root_input': _text,
    'atlas_input': _text,
    'n_workers': _number(1, 64),
    'mode': _choice(['Local', 'Apptainer']),
    'image_path': _text,
    'checkout_path': _text,
    'pip_install': _flag,
    **{f'include_{step}': _flag for step in (
        'preprocessing', 'msmt_csd', 'mcm_estimation', 'bundle_seg', 'mcm_projection', 'tractometry')},
    'preprocessing_steps': _subset(PREPROCESSING_STEPS),
    'with_reversed_b0': _flag,
    'flip_bvec_checkbox': _flag,
    'flip_bvec': _number(0, 2),
    'msmt_pipeline': _text,
    'n_streamlines': _number(1000, 10_000_000),
    'msmt_steps': _subset(MSMT_STEPS),
    'force': _flag,
    'mcm_pipeline': _text,
    'tensor_model': _choice(COMPARTMENT_TYPES),
    'model_selection': _flag,
    'n_comparts': _number(1, 5),
    'free_water': _flag,
    'restricted_water': _flag,
    'stanisz': _flag,
    'stationary_water': _flag,
    'ml_mode': _choice(ML_MODES),
    'optimizer': _choice(OPTIMIZERS),
    'n_threads': _number(0, 256),
    'keep_intermediate': _flag,
    'projection_mcm_pipeline': _text,
    'projection_bundle_pipeline': _text,
    'tractometry_pipeline': _text,
    'clustering_method': _choice(['centroid', 'parcellation']),
    'tractometry_mcm_pipeline': _text,
    'tractometry_bundle_pipeline': _text,
    'n_pts': _number(10, 500),
}
DEFAULTS: Dict[str, Any] = {
    'n_workers': 1,
    'mode': 'Local',
    'pip_install': True,
    **{key: True for key in COMMON_SETTINGS if key.startswith('include_')},
    'preprocessing_steps': list(PREPROCESSING_STEPS),
    'with_reversed_b0': True,
    'flip_bvec_checkbox': True,
    'flip_bvec': 1,
    'n_streamlines': 1_000_000,
    'msmt_steps': list(MSMT_STEPS),
    'force': False,
    'tensor_model': 3,
    'model_selection': False,
    'n_comparts': 3,
    'free_water': True,
    'restricted_water': True,
    'stanisz': False,
    'stationary_water': False,
    'ml_mode': 2,
    'n_threads': 0,
    'keep_intermediate': False,
    'n_pts': 100,
}
# Settings that only make sense for one dataset.
DATASET_SETTINGS = {'log_root': _text, 'subjects': None}


def read_settings() -> Dict[str, Any]:
    try:
        data = json.loads(SETTINGS_FILE.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_settings(settings: Dict[str, Any]) -> None:
    """Best effort: a read-only workdir must not break the launcher."""
    try:
        WORKDIR.mkdir(parents=True, exist_ok=True)
        tmp = SETTINGS_FILE.with_suffix('.tmp')
        tmp.write_text(json.dumps(settings, indent=1, sort_keys=True))
        os.replace(tmp, SETTINGS_FILE)
    except OSError:
        pass


def restore_common_settings() -> None:
    """Seed missing widget values from the saved settings, then from the defaults.

    Widgets that are not drawn on the current page lose their value, so this
    runs on every script run. Environment variables win on the first run.
    """
    if not st.session_state.get('environment_applied'):
        st.session_state['environment_applied'] = True
        for key, variable in (('db_root_input', 'TRACTOPL_DATASET_ROOT'), ('atlas_input', 'TRACTOPL_ATLAS')):
            if os.environ.get(variable):
                st.session_state[key] = os.environ[variable]
    saved = read_settings().get('common', {})
    for key, is_valid in COMMON_SETTINGS.items():
        if key in st.session_state:
            continue
        if key in saved and is_valid(saved[key]):
            st.session_state[key] = saved[key]
        elif key in DEFAULTS:
            st.session_state[key] = copy.copy(DEFAULTS[key])


def restore_dataset_settings(db_root: str, subjects: List[str]) -> None:
    """Seed the participant selection and log directory saved for this dataset."""
    if st.session_state.get('settings_restored_for') != db_root:
        st.session_state['settings_restored_for'] = db_root
        saved = read_settings().get('datasets', {}).get(db_root, {})
        chosen = saved.get('subjects')
        chosen = [item for item in chosen if item in subjects] if isinstance(chosen, list) else []
        st.session_state['subjects'] = chosen or subjects[:1]
        log_root = saved.get('log_root')
        st.session_state['log_root'] = (
            log_root if _text(log_root) and log_root else os.path.join(db_root, 'tractopl_logs')
        )


def save_settings(db_root: str) -> None:
    """Merge the widgets currently on screen into the saved settings."""
    settings = read_settings()
    common = settings.get('common') if isinstance(settings.get('common'), dict) else {}
    datasets = settings.get('datasets') if isinstance(settings.get('datasets'), dict) else {}
    updated_common = dict(common)
    for key, is_valid in COMMON_SETTINGS.items():
        if key in st.session_state and is_valid(st.session_state[key]):
            updated_common[key] = st.session_state[key]
    updated_dataset = dict(datasets.get(db_root, {}))
    for key in DATASET_SETTINGS:
        if key in st.session_state:
            updated_dataset[key] = st.session_state[key]
    updated = {'common': updated_common, 'datasets': {**datasets, db_root: updated_dataset}}
    if updated != {'common': common, 'datasets': datasets}:
        write_settings(updated)


def forget_settings() -> None:
    try:
        SETTINGS_FILE.unlink()
    except OSError:
        pass
    for key in [*COMMON_SETTINGS, *DATASET_SETTINGS, 'environment_applied', 'settings_restored_for']:
        st.session_state.pop(key, None)


# =============================================================================
# Page
# =============================================================================

st.set_page_config(page_title='TractoPL', layout='wide')

st.markdown(
    """
    <style>
    header[data-testid="stHeader"] {
        background-color: #4a4f57;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
    }
    [data-testid="stToolbar"]::before {
        content: "TractoPL";
        color: #ffffff;
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        padding: 0 1.5rem 0 0.5rem;
        margin-right: 0.5rem;
        border-right: 1px solid rgba(255, 255, 255, 0.35);
        white-space: nowrap;
    }
    [data-testid="stTopNavLink"],
    [data-testid="stTopNavLink"] * {
        color: #ffffff !important;
    }
    [data-testid="stTopNavLink"] {
        border-radius: 0.375rem;
        font-size: 1.15rem;
        padding: 0.4rem 1rem;
    }
    [data-testid="stTopNavLink"] * {
        font-size: 1.15rem !important;
    }
    [data-testid="stTopNavLink"]:hover {
        background-color: rgba(255, 255, 255, 0.15);
    }
    [data-testid="stTopNavLink"][aria-current="page"] {
        background-color: #6b7280;
        font-weight: 600;
    }
    [data-testid="stToolbarActions"] *,
    [data-testid="stMainMenu"] * {
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

restore_common_settings()

with st.sidebar:
    st.header('Dataset')
    db_root = st.text_input('BIDS root', key='db_root_input')
    atlas = st.text_input('Atlas manifest (JSON)', key='atlas_input')
    if st.button('Refresh dataset'):
        load_dataset.clear()
        _cached_counts.clear()

if not db_root or not os.path.isdir(db_root):
    st.info('Enter an existing BIDS root in the sidebar to start.')
    st.stop()
db_root = os.path.abspath(db_root)

try:
    dataset = load_dataset(db_root)
    description = dataset.describe()
except Exception as error:  # noqa: BLE001 - surfaced to the user
    st.error(f"Could not index the dataset: {error}")
    st.stop()

SUBJECTS: List[str] = description['subjects']
DERIVATIVES: List[str] = description['derivatives']
restore_dataset_settings(db_root, SUBJECTS)

with st.sidebar:
    st.caption(f"{len(SUBJECTS)} participants, {description['file_count']} files")
    subjects = st.multiselect('Participants', SUBJECTS, key='subjects')
    st.caption('Existing derivatives: ' + (', '.join(DERIVATIVES) if DERIVATIVES else 'none'))

    st.header('Execution')
    n_workers = st.number_input(
        'Parallel participants', min_value=1, max_value=64, key='n_workers',
        help='Participants processed at the same time. Steps of one participant stay sequential.',
    )
    log_root = st.text_input(
        'Log directory', key='log_root',
        help='Jobs keep running without the dashboard; their logs are stored here.',
    )
    mode = st.radio('Run commands', ['Local', 'Apptainer'], horizontal=True, key='mode')
    execution: Dict[str, Any] = {'mode': mode.lower(), 'db_root': db_root, 'atlas': atlas}
    available_commands: Optional[List[str]] = None
    execution_problem = None

    if mode == 'Apptainer':
        images = sorted(str(path) for path in CHECKOUT.glob('*.sif'))
        st.session_state.setdefault('image_path', images[0] if images else '')
        st.session_state.setdefault('checkout_path', str(CHECKOUT))
        execution['image'] = os.path.abspath(st.text_input('Image (.sif)', key='image_path'))
        execution['checkout'] = os.path.abspath(st.text_input('TractoPL checkout', key='checkout_path'))
        execution['pip_install'] = st.checkbox(
            'Install the checkout in the image', key='pip_install',
            help='Runs `pip install -e` before each command (in a writable tmpfs). '
                 'Bundle segmentation needs it for tractopl-generate-centroid and '
                 'tractopl-apply-trans-to-vtk.',
        )
        key = (execution['image'], execution['checkout'])
        probe = st.session_state.get('probe')

        if not shutil.which('apptainer'):
            execution_problem = 'apptainer is not on PATH.'
        elif not os.path.isfile(execution['image']):
            execution_problem = 'Select an existing .sif image.'
        elif st.button('Load image'):
            with st.spinner('Starting the image...'):
                try:
                    probe = probe_image(execution)
                except subprocess.TimeoutExpired:
                    probe = {'key': key, 'ok': False, 'module': '', 'commands': [],
                             'error': 'The image did not answer within 5 minutes.'}
            st.session_state['probe'] = probe

        if execution_problem is None:
            if not probe or probe['key'] != key:
                execution_problem = 'Load the Apptainer image before running.'
            elif not probe['ok']:
                execution_problem = 'The image could not import TractoPL from the checkout.'
                st.error(probe['error'] or 'Unknown error.')
            else:
                available_commands = probe['commands']
                st.success(f"Image loaded: {len(probe['commands'])}/{len(TRACTOPL_COMMANDS)} commands")
                st.caption(f"TractoPL from {probe['module']}")

    if execution_problem:
        st.warning(execution_problem)

    st.divider()
    st.caption(f'Settings are remembered in {WORKDIR}')
    if st.button('Forget saved settings'):
        forget_settings()
        st.rerun()

if not SUBJECTS:
    st.warning('No participant was found in this dataset.')
    st.stop()

params: Dict[str, Any] = {'db_root': db_root, 'atlas': atlas}


def command_available(name: str) -> bool:
    if available_commands is None:
        return shutil.which(name) is not None
    return name in available_commands


def render_checks(title: str, checks: List[Check]) -> pd.DataFrame:
    """Show which selected participants have each expected file."""
    table = check_table(dataset, checks, subjects)
    done = len(complete_subjects(table))
    st.markdown(f"**{title}** — {done}/{len(subjects)} participants complete")
    if len(table.columns):
        icons = table.apply(lambda column: column.map({True: '✅', False: '❌'}))
        st.dataframe(icons.rename(index=lambda subject: f'sub-{subject}'))
    return table


def step_block(step: str, title: str, summary: str, options, needs_atlas: bool = False):
    """Render one step: options, input/output checks, commands, and a run button."""
    with st.container(border=True):
        header, toggle = st.columns([5, 1])
        header.markdown(f"#### {title}")
        header.caption(summary)
        included = toggle.checkbox('Include', key=f'include_{step}')

        options()

        inputs, outputs = step_io(step, params)
        missing_inputs: List[str] = []
        if subjects:
            with st.expander('Inputs and outputs', expanded=False):
                inputs_column, outputs_column = st.columns(2)
                with inputs_column:
                    input_table = render_checks('Inputs', inputs)
                with outputs_column:
                    output_table = render_checks('Outputs', outputs)
            ready = set(complete_subjects(input_table))
            missing_inputs = [subject for subject in subjects if subject not in ready]
            done = len(complete_subjects(output_table))
            header.caption(f"Outputs complete for {done}/{len(subjects)} selected participants")

        problems = []
        if not subjects:
            problems.append('Select at least one participant in the sidebar.')
        if step == 'preprocessing' and not params['preprocessing_steps']:
            problems.append('Select at least one preprocessing step.')
        if needs_atlas and not os.path.isfile(atlas):
            problems.append('This step needs an existing atlas manifest.')
        if execution_problem:
            problems.append(execution_problem)

        commands = [wrap_command(build_command(step, subject, params), execution) for subject in subjects]
        if commands:
            with st.expander('Commands', expanded=False):
                st.code('\n'.join(shlex.join(command) for command in commands), language='shell')
            tool = build_command(step, subjects[0], params)[0]
            if not execution_problem and not command_available(tool):
                problems.append(f"{tool} is not available.")

        for problem in problems:
            st.warning(problem)
        if missing_inputs and not problems:
            st.info(
                'Inputs missing for ' + ', '.join(f'sub-{subject}' for subject in missing_inputs)
                + '. Earlier steps in the same run may produce them.'
            )

        run_now = st.button('Run this step', key=f'run_{step}', disabled=bool(problems))

    return step, included, run_now, bool(problems)


# =============================================================================
# Workflow steps
# =============================================================================

def preprocessing_options():
    left, right = st.columns(2)
    with left:
        selected = st.multiselect(
            'Steps', list(PREPROCESSING_STEPS),
            format_func=PREPROCESSING_STEPS.get, key='preprocessing_steps',
        )
        # Keep the pipeline order whatever the selection order.
        params['preprocessing_steps'] = [name for name in PREPROCESSING_STEPS if name in selected]
    with right:
        params['with_reversed_b0'] = st.checkbox(
            'Use a reversed B0 for distortion correction', key='with_reversed_b0',
            disabled='anima' not in params['preprocessing_steps'],
        )

    with right:
        params['flip_bvec_checkbox'] = st.checkbox(
            'Flip the b-vectors', key='flip_bvec_checkbox',
            disabled='dti' not in params['preprocessing_steps'],
        )
        if params['flip_bvec_checkbox']:
            params['flip_bvec'] = st.slider(
                'B-vector flip axis', min_value=0, max_value=2, step=1,
                key='flip_bvec',
                disabled='dti' not in params['preprocessing_steps'],
            )


def msmt_options():
    left, right = st.columns(2)
    with left:
        params['msmt_pipeline'] = pipeline_selector('Output pipeline', 'msmt_csd', 'msmt_pipeline')
        params['n_streamlines'] = st.number_input(
            'Streamlines', min_value=1000, max_value=10_000_000, step=100_000,
            key='n_streamlines',
        )
    with right:
        params['msmt_steps'] = st.multiselect('Steps', MSMT_STEPS, key='msmt_steps')
        params['force'] = st.checkbox('Recompute existing outputs', key='force')


def mcm_estimation_options():
    left, middle, right = st.columns(3)
    with left:
        st.markdown('**Model**')
        params['mcm_pipeline'] = pipeline_selector('Output pipeline', 'mcm_tensors', 'mcm_pipeline')
        params['tensor_model'] = st.selectbox(
            'Anisotropic compartment type', list(COMPARTMENT_TYPES),
            format_func=lambda value: f'{value} — {COMPARTMENT_TYPES[value]}', key='tensor_model',
        )
        params['model_selection'] = st.checkbox(
            'Select the number of compartments with AIC', key='model_selection'
        )
        params['n_comparts'] = st.number_input(
            'Anisotropic compartments', min_value=1, max_value=5,
            disabled=params['model_selection'], key='n_comparts',
            help='Without AIC, the fixel density map from MSMT-CSD sets the count per voxel.',
        )
    with middle:
        st.markdown('**Isotropic compartments**')
        params['free_water'] = st.checkbox('Free water', key='free_water')
        params['restricted_water'] = st.checkbox('Restricted water', key='restricted_water')
        params['stanisz'] = st.checkbox('Stanisz', key='stanisz')
        params['stationary_water'] = st.checkbox('Stationary water', key='stationary_water')
    with right:
        st.markdown('**Optimization**')
        params['ml_mode'] = st.selectbox(
            'Maximum-likelihood mode', list(ML_MODES),
            format_func=lambda value: f'{value} — {ML_MODES[value]}', key='ml_mode',
        )
        params['optimizer'] = st.selectbox('Optimizer', OPTIMIZERS, key='optimizer')
        params['n_threads'] = st.number_input(
            'Threads (0 = all cores)', min_value=0, max_value=256, key='n_threads'
        )


def bundle_seg_options():
    params['keep_intermediate'] = st.checkbox(
        'Keep intermediate files', key='keep_intermediate'
    )


def mcm_projection_options():
    left, right = st.columns(2)
    with left:
        params['projection_mcm_pipeline'] = pipeline_selector(
            'MCM pipeline', params.get('mcm_pipeline', 'mcm_tensors'), 'projection_mcm_pipeline'
        )
    with right:
        params['projection_bundle_pipeline'] = pipeline_selector(
            'Bundle pipeline', 'bundle_seg', 'projection_bundle_pipeline'
        )


def tractometry_options():
    left, middle, right = st.columns(3)
    with left:
        params['tractometry_pipeline'] = pipeline_selector(
            'Output pipeline', 'tractometry', 'tractometry_pipeline'
        )
        params['clustering_method'] = st.selectbox(
            'Clustering method', ['centroid', 'parcellation'], key='clustering_method'
        )
    with middle:
        params['tractometry_mcm_pipeline'] = pipeline_selector(
            'MCM pipeline', params.get('projection_mcm_pipeline', 'mcm_tensors'),
            'tractometry_mcm_pipeline',
        )
        params['n_pts'] = st.number_input(
            'Points per bundle', min_value=10, max_value=500, step=10, key='n_pts'
        )
    with right:
        params['tractometry_bundle_pipeline'] = pipeline_selector(
            'Bundle pipeline', params.get('projection_bundle_pipeline', 'bundle_seg'),
            'tractometry_bundle_pipeline',
        )


def pipeline_page() -> None:
    blocks = [
        step_block(
            'preprocessing',
            '1. DWI preprocessing',
            'Corrects the DWI and writes a brain mask, fits DTI, and writes FA. Each step can run alone.',
            preprocessing_options,
        ),
        step_block(
            'msmt_csd',
            '2. MSMT-CSD, fixels, and tractography',
            'Writes responses, FODs, fixels, peaks, density, and a whole-brain tractogram.',
            msmt_options,
        ),
        step_block(
            'mcm_estimation',
            '3. MCM estimation',
            'Fits the multi-compartment model and writes the .mcmx archive.',
            mcm_estimation_options,
        ),
        step_block(
            'bundle_seg',
            '4. Atlas registration and bundle segmentation',
            'Registers the atlas and segments bundles from the whole-brain tractogram.',
            bundle_seg_options,
            needs_atlas=True,
        ),
        step_block(
            'mcm_projection',
            '5. MCM projection onto bundles',
            'Writes one metric-bearing VTK tractogram per segmented bundle.',
            mcm_projection_options,
        ),
        step_block(
            'tractometry',
            '6. Tractometry association',
            'Writes association VTK files and per-bundle metric CSV files.',
            tractometry_options,
            needs_atlas=True,
        ),
    ]

    # =============================================================================
    # Execution
    # =============================================================================

    included = [step for step, include, _, blocked in blocks if include and not blocked]
    blocked_included = [step for step, include, _, blocked in blocks if include and blocked]

    st.divider()
    run_all = st.button(
        f"Run the {len(included)} included steps",
        type='primary',
        disabled=not included or bool(blocked_included),
    )
    if blocked_included:
        st.caption('Fix the warnings above to run every included step: ' + ', '.join(blocked_included))

    steps_to_run: List[str] = [step for step, _, run_now, _ in blocks if run_now]
    if run_all:
        steps_to_run = included

    if steps_to_run:
        tasks = [
            {'subject': subject, 'step': step,
             'command': wrap_command(build_command(step, subject, params), execution)}
            for subject in subjects for step in steps_to_run
        ]
        job_name = submit_job(log_root, int(n_workers), tasks)
        jobs_changed()
        announce(log_root, f'{job_name}__started',
                 f'Job {job_name} started: {len(tasks)} task(s), {int(n_workers)} parallel participant(s).')
        flush_notifications()
        st.success(f'Job {job_name} started: {len(tasks)} task(s), {int(n_workers)} parallel participant(s). '
                   'Follow it on the Jobs page.')


def jobs_page() -> None:
    render_jobs(log_root)
    st.fragment(run_every=5)(watch_jobs)(log_root)


with st.sidebar:
    st.header('Jobs')
    if hasattr(st, 'fragment'):
        st.fragment(run_every=5)(job_watcher)(log_root)
    else:
        job_watcher(log_root)

navigation = st.navigation(
    [
        st.Page(pipeline_page, title='Pipeline', default=True),
        st.Page(jobs_page, title='Jobs'),
    ],
    position='top',
)
navigation.run()
save_settings(db_root)
