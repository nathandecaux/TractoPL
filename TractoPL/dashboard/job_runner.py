"""Detached job runner for the launcher.

Usage: python -m TractoPL.dashboard.job_runner JOB_DIR

JOB_DIR holds ``job.json`` (``{"workers": int, "tasks": [{"subject", "step", "command"}]}``).
The runner executes the tasks of one participant sequentially (in the given
order) and up to ``workers`` participants in parallel. It writes ``state.json``
and one ``sub-<subject>_<step>.log`` per task in JOB_DIR. It does not depend on
the Streamlit server, so closing the dashboard does not stop the job.

Cancellation: SIGTERM/SIGINT, or a file in JOB_DIR (works from any host that
sees the directory):
- ``cancel`` cancels the whole job;
- ``cancel_sub-<subject>`` cancels one participant.
Each task runs in its own process group, which is terminated as a whole so the
tools started by the task (ANTs, MRtrix, anima...) stop too.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List

POLL_SECONDS = 2
HEARTBEAT_SECONDS = 10
KILL_AFTER_SECONDS = 20
LOG_LIMIT_BYTES = 10 * 1024 * 1024
LOG_TAIL_FLUSH_SECONDS = 10

_lock = threading.Lock()
_children: Dict[str, subprocess.Popen] = {}  # subject -> running task
_cancelled = threading.Event()
_cancelled_subjects: set = set()
_done = threading.Event()


class CappedLog:
    """Binary log file kept under `limit` bytes.

    Writes go straight to the file until it reaches the limit. Past it, the
    first half stays on disk and only the most recent output is kept after a
    marker; that tail is rewritten every few seconds and on close.
    """

    def __init__(self, path: Path, limit: int = LOG_LIMIT_BYTES):
        self.file = open(path, 'wb')
        self.head_limit = limit // 2
        self.tail_limit = limit - self.head_limit - 200  # room for the marker
        self.size = 0
        self.tail = None  # bytearray once the limit was reached
        self.dropped = 0
        self.last_flush = 0.0
        self.limit = limit

    def write(self, data: bytes) -> None:
        if self.tail is None:
            if self.size + len(data) <= self.limit:
                self.file.write(data)
                self.size += len(data)
                return
            # Limit reached: everything after the head becomes the tail buffer.
            self.file.flush()
            with open(self.file.name, 'rb') as reader:
                reader.seek(self.head_limit)
                self.tail = bytearray(reader.read())
        self.tail += data
        excess = len(self.tail) - self.tail_limit
        if excess > 0:
            del self.tail[:excess]
            self.dropped += excess
        if time.time() - self.last_flush >= LOG_TAIL_FLUSH_SECONDS:
            self.flush_tail()

    def flush_tail(self) -> None:
        if self.tail is None:
            return
        self.last_flush = time.time()
        marker = (f'\n\n[runner] ... {self.dropped / 1024 / 1024:.1f} MB of output dropped '
                  f'(log limited to {self.limit // (1024 * 1024)} MB) ...\n\n').encode()
        self.file.seek(self.head_limit)
        self.file.write(marker)
        self.file.write(self.tail)
        self.file.truncate()

    def text(self, message: str) -> None:
        self.write(message.encode())
        self.file.flush()

    def flush(self) -> None:
        self.file.flush()

    def close(self) -> None:
        self.flush_tail()
        self.file.close()


def copy_output(process: subprocess.Popen, log: CappedLog) -> None:
    # read1 returns what is available, so the log stays live.
    for chunk in iter(lambda: process.stdout.read1(65536), b''):
        log.write(chunk)
        log.flush()


def task_log_name(task: Dict[str, Any]) -> str:
    return f"sub-{task['subject']}_{task['step']}.log"


def cancel_file(job_dir: Path, subject: str = None) -> Path:
    return job_dir / ('cancel' if subject is None else f'cancel_sub-{subject}')


def write_state(job_dir: Path, state: Dict[str, Any]) -> None:
    tmp = job_dir / 'state.json.tmp'
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, job_dir / 'state.json')


def update(job_dir: Path, state: Dict[str, Any], index: int, **fields: Any) -> None:
    with _lock:
        state['tasks'][index].update(fields)
        write_state(job_dir, state)


def is_cancelled(subject: str) -> bool:
    return _cancelled.is_set() or subject in _cancelled_subjects


def stop_process(process: subprocess.Popen) -> None:
    """SIGTERM the task's process group, then SIGKILL it if it does not stop."""
    def signal_group(sig: int) -> None:
        try:
            os.killpg(process.pid, sig)
        except OSError:
            pass

    signal_group(signal.SIGTERM)

    def escalate() -> None:
        try:
            process.wait(timeout=KILL_AFTER_SECONDS)
        except subprocess.TimeoutExpired:
            signal_group(signal.SIGKILL)

    threading.Thread(target=escalate, daemon=True).start()


def cancel_subject(subject: str) -> None:
    with _lock:
        _cancelled_subjects.add(subject)
        process = _children.get(subject)
    if process is not None:
        stop_process(process)


def cancel_all(*_: Any) -> None:
    _cancelled.set()
    with _lock:
        processes = list(_children.values())
    for process in processes:
        stop_process(process)


def run_subject(job_dir: Path, state: Dict[str, Any], indices: List[int], tasks: List[Dict[str, Any]]) -> None:
    failed = False
    for index in indices:
        task = tasks[index]
        subject = task['subject']
        if failed or is_cancelled(subject):
            update(job_dir, state, index, status='cancelled' if is_cancelled(subject) else 'skipped')
            continue
        update(job_dir, state, index, status='running', started=time.time())
        code = None
        log = CappedLog(job_dir / task_log_name(task))
        try:
            log.text('$ ' + ' '.join(task['command']) + '\n')
            try:
                # Own session/process group: killpg reaches every tool the task starts.
                # Output goes through the runner so the log size can be capped.
                process = subprocess.Popen(
                    task['command'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True,
                )
            except OSError as error:
                log.text(f'Could not start: {error}\n')
                update(job_dir, state, index, status='failed', returncode=-1, ended=time.time())
                failed = True
                continue
            with _lock:
                _children[subject] = process
            update(job_dir, state, index, pgid=process.pid)
            if is_cancelled(subject):  # cancelled while starting
                stop_process(process)
            reader = threading.Thread(target=copy_output, args=(process, log), daemon=True)
            reader.start()
            code = process.wait()
            # Tools left in the background may keep the output open: give them
            # a moment, then stop what remains of the task's process group.
            reader.join(timeout=10)
            if reader.is_alive():
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    pass
                reader.join()
            with _lock:
                _children.pop(subject, None)
            if is_cancelled(subject):
                log.text('\n[runner] Task cancelled.\n')
        finally:
            log.close()
        status = 'cancelled' if is_cancelled(subject) else ('success' if code == 0 else 'failed')
        update(job_dir, state, index, status=status, returncode=code, ended=time.time())
        failed = status != 'success'


def watch(job_dir: Path, state: Dict[str, Any], subjects: List[str]) -> None:
    """Poll the cancel files and refresh the heartbeat while the job runs."""
    last_beat = 0.0
    while not _done.wait(POLL_SECONDS):
        if not _cancelled.is_set() and cancel_file(job_dir).exists():
            cancel_all()
        for subject in subjects:
            if subject not in _cancelled_subjects and cancel_file(job_dir, subject).exists():
                cancel_subject(subject)
        if time.time() - last_beat >= HEARTBEAT_SECONDS:
            last_beat = time.time()
            with _lock:
                state['heartbeat'] = last_beat
                try:
                    write_state(job_dir, state)
                except OSError:
                    pass


def main(job_dir: Path) -> int:
    spec = json.loads((job_dir / 'job.json').read_text())
    tasks = spec['tasks']
    now = time.time()
    state = {
        'pid': os.getpid(),
        'host': socket.gethostname(),
        'status': 'running',
        'started': now,
        'heartbeat': now,
        'ended': None,
        'tasks': [
            {'subject': t['subject'], 'step': t['step'], 'status': 'pending',
             'log': task_log_name(t), 'returncode': None, 'started': None, 'ended': None}
            for t in tasks
        ],
    }
    write_state(job_dir, state)

    signal.signal(signal.SIGTERM, cancel_all)
    signal.signal(signal.SIGINT, cancel_all)

    by_subject: Dict[str, List[int]] = {}
    for index, task in enumerate(tasks):
        by_subject.setdefault(task['subject'], []).append(index)

    watcher = threading.Thread(target=watch, args=(job_dir, state, list(by_subject)), daemon=True)
    watcher.start()
    error = None
    try:
        with ThreadPoolExecutor(max_workers=max(1, int(spec.get('workers', 1)))) as pool:
            futures = [pool.submit(run_subject, job_dir, state, indices, tasks)
                       for indices in by_subject.values()]
            # Short waits keep the main thread responsive to signals.
            pending = set(futures)
            while pending:
                _, pending = wait(pending, timeout=1)
            for future in futures:
                future.result()
    except BaseException:  # noqa: BLE001 - recorded in the state, then re-raised
        error = traceback.format_exc()
        cancel_all()
        raise
    finally:
        _done.set()
        with _lock:
            for task in state['tasks']:
                if task['status'] in ('pending', 'running'):
                    task['status'] = 'cancelled' if _cancelled.is_set() else 'failed'
            statuses = {t['status'] for t in state['tasks']}
            state['status'] = ('failed' if error
                               else 'cancelled' if _cancelled.is_set() or statuses == {'cancelled'}
                               else 'failed' if statuses & {'failed', 'skipped', 'cancelled'}
                               else 'success')
            if error:
                state['error'] = error
            state['ended'] = time.time()
            write_state(job_dir, state)
    return 0


if __name__ == '__main__':
    sys.exit(main(Path(sys.argv[1])))
