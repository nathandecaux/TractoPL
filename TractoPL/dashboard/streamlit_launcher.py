"""
Streamlit launcher for the TractoPL command-line workflow.

One block per workflow step: its options, the inputs it needs and the outputs
it writes (checked in the dataset), the commands it will run, and a button to
run it. Commands run locally or inside an Apptainer image. See
examples/README.md for what each step reads and writes.

Usage:
    streamlit run TractoPL/dashboard/streamlit_launcher.py
"""

import glob
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

import pandas as pd
import streamlit as st

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
        counts = count_matches(dataset, check, subjects)
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
        inputs = [
            Check('Raw DWI', dict(scope='raw', suffix='dwi', extension='nii.gz', desc='!*')),
            Check('Raw bval/bvec', dict(scope='raw', suffix='dwi', extension=['.bval', '.bvec']), 2),
            Check('T1w', dict(scope='raw', suffix='T1w', extension='nii.gz')),
            Check('DICOMs', pattern='{root}/sub-{sub}/sourcedata/sub-{sub}_dicoms/**/*'),
        ]
        if params['with_reversed_b0']:
            inputs.append(Check('Reversed B0', dict(scope='raw', desc='b0reversed', extension='nii.gz')))
        outputs = [
            preproc_dwi(),
            preproc_bvec(),
            brain_mask(),
            Check('DTI', dict(pipeline='preprocessing', model='DTI', metric='!*', extension='nii.gz')),
            fa_map(),
        ]
        return inputs, outputs

    if step == 'msmt_csd':
        pipeline = params['msmt_pipeline']
        tissues = ['WM', 'GM', 'CSF']
        by_step = {
            'response': Check('Responses', dict(pipeline=pipeline, suffix='response', label=tissues), 3),
            'fod': Check('FODs', dict(pipeline=pipeline, suffix='fod', label=tissues), 3),
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
        if not params['with_reversed_b0']:
            command.append('--without-reversed-b0')
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


def apptainer_prefix(execution: Dict[str, Any]) -> List[str]:
    """Return `apptainer exec` with the dataset, atlas, and checkout mounted."""
    binds = [execution['db_root']]
    if execution['atlas']:
        binds.append(os.path.dirname(os.path.abspath(execution['atlas'])))
    # --pwd keeps the host working directory from shadowing the mounted checkout
    # on sys.path (the image wrappers use `python -m`).
    prefix = ['apptainer', 'exec', '--cleanenv', '--writable-tmpfs', '--pwd', '/tmp']
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


def run_command(command: List[str], output_area) -> tuple:
    """Run one command, stream its output into the page, and return (code, log)."""
    lines: List[str] = []
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in process.stdout:
        lines.append(line.rstrip())
        output_area.code('\n'.join(lines[-200:]), language='text')
    return process.wait(), '\n'.join(lines[-200:])


def pipeline_selector(label: str, default: str, key: str) -> str:
    """Select a derivative pipeline, allowing one that does not exist yet."""
    preferred = st.session_state.get(key, default)
    options = sorted(set(DERIVATIVES) | {default, preferred})
    return st.selectbox(label, options, index=options.index(preferred), key=key)


# =============================================================================
# Page
# =============================================================================

st.set_page_config(page_title='TractoPL Launcher', layout='wide')
st.title('TractoPL Launcher')
st.session_state.setdefault('logs', {})

with st.sidebar:
    st.header('Dataset')
    db_root = st.text_input('BIDS root', value=os.environ.get('TRACTOPL_DATASET_ROOT', ''))
    atlas = st.text_input('Atlas manifest (JSON)', value=os.environ.get('TRACTOPL_ATLAS', ''))
    if st.button('Refresh dataset'):
        load_dataset.clear()

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

with st.sidebar:
    st.caption(f"{len(SUBJECTS)} participants, {description['file_count']} files")
    subjects = st.multiselect('Participants', SUBJECTS, default=SUBJECTS[:1], key='subjects')
    st.caption('Existing derivatives: ' + (', '.join(DERIVATIVES) if DERIVATIVES else 'none'))

    st.header('Execution')
    mode = st.radio('Run commands', ['Local', 'Apptainer'], horizontal=True, key='mode')
    execution: Dict[str, Any] = {'mode': mode.lower(), 'db_root': db_root, 'atlas': atlas}
    available_commands: Optional[List[str]] = None
    execution_problem = None

    if mode == 'Apptainer':
        images = sorted(str(path) for path in CHECKOUT.glob('*.sif'))
        execution['image'] = os.path.abspath(st.text_input('Image (.sif)', value=images[0] if images else ''))
        execution['checkout'] = os.path.abspath(st.text_input('TractoPL checkout', value=str(CHECKOUT)))
        execution['pip_install'] = st.checkbox(
            'Install the checkout in the image', value=True, key='pip_install',
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

if not SUBJECTS:
    st.warning('No participant was found in this dataset.')
    st.stop()

params: Dict[str, Any] = {'db_root': db_root, 'atlas': atlas}
output_areas: Dict[str, Any] = {}
status_areas: Dict[str, Any] = {}


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
        included = toggle.checkbox('Include', value=True, key=f'include_{step}')

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
        status_areas[step] = st.empty()
        output_areas[step] = st.empty()
        if step in st.session_state['logs']:
            kind, message, log = st.session_state['logs'][step]
            getattr(status_areas[step], kind)(message)
            output_areas[step].code(log, language='text')

    return step, included, run_now, bool(problems)


# =============================================================================
# Workflow steps
# =============================================================================

def preprocessing_options():
    params['with_reversed_b0'] = st.checkbox(
        'Use a reversed B0 for distortion correction', value=True, key='with_reversed_b0'
    )


def msmt_options():
    left, right = st.columns(2)
    with left:
        params['msmt_pipeline'] = pipeline_selector('Output pipeline', 'msmt_csd', 'msmt_pipeline')
        params['n_streamlines'] = st.number_input(
            'Streamlines', min_value=1000, max_value=10_000_000, value=1_000_000, step=100_000,
            key='n_streamlines',
        )
    with right:
        params['msmt_steps'] = st.multiselect('Steps', MSMT_STEPS, default=MSMT_STEPS, key='msmt_steps')
        params['force'] = st.checkbox('Recompute existing outputs', value=False, key='force')


def mcm_estimation_options():
    left, middle, right = st.columns(3)
    with left:
        st.markdown('**Model**')
        params['mcm_pipeline'] = pipeline_selector('Output pipeline', 'mcm_tensors', 'mcm_pipeline')
        params['tensor_model'] = st.selectbox(
            'Anisotropic compartment type', list(COMPARTMENT_TYPES), index=2,
            format_func=lambda value: f'{value} — {COMPARTMENT_TYPES[value]}', key='tensor_model',
        )
        params['model_selection'] = st.checkbox(
            'Select the number of compartments with AIC', value=False, key='model_selection'
        )
        params['n_comparts'] = st.number_input(
            'Anisotropic compartments', min_value=1, max_value=5, value=3,
            disabled=params['model_selection'], key='n_comparts',
            help='Without AIC, the fixel density map from MSMT-CSD sets the count per voxel.',
        )
    with middle:
        st.markdown('**Isotropic compartments**')
        params['free_water'] = st.checkbox('Free water', value=True, key='free_water')
        params['restricted_water'] = st.checkbox('Restricted water', value=True, key='restricted_water')
        params['stanisz'] = st.checkbox('Stanisz', value=False, key='stanisz')
        params['stationary_water'] = st.checkbox('Stationary water', value=False, key='stationary_water')
    with right:
        st.markdown('**Optimization**')
        params['ml_mode'] = st.selectbox(
            'Maximum-likelihood mode', list(ML_MODES), index=2,
            format_func=lambda value: f'{value} — {ML_MODES[value]}', key='ml_mode',
        )
        params['optimizer'] = st.selectbox('Optimizer', OPTIMIZERS, key='optimizer')
        params['n_threads'] = st.number_input(
            'Threads (0 = all cores)', min_value=0, max_value=256, value=0, key='n_threads'
        )


def bundle_seg_options():
    params['keep_intermediate'] = st.checkbox(
        'Keep intermediate files', value=False, key='keep_intermediate'
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
            'Points per bundle', min_value=10, max_value=500, value=100, step=10, key='n_pts'
        )
    with right:
        params['tractometry_bundle_pipeline'] = pipeline_selector(
            'Bundle pipeline', params.get('projection_bundle_pipeline', 'bundle_seg'),
            'tractometry_bundle_pipeline',
        )


blocks = [
    step_block(
        'preprocessing',
        '1. DWI preprocessing',
        'Corrects the DWI, fits DTI, and writes FA and a brain mask.',
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
    for step in steps_to_run:
        st.session_state['logs'].pop(step, None)
    failed = False
    for step in steps_to_run:
        logs: List[str] = []
        for subject in subjects:
            status_areas[step].info(f"Running sub-{subject}...")
            command = wrap_command(build_command(step, subject, params), execution)
            return_code, log = run_command(command, output_areas[step])
            logs.append(f"### sub-{subject}\n{log}")

            # Several commands exit 0 after catching an error: check the outputs too.
            load_dataset.clear()
            dataset = load_dataset(db_root)
            _, outputs = step_io(step, params)
            table = check_table(dataset, outputs, [subject])
            missing = [label for label in table.columns if not table.loc[subject, label]]
            if return_code != 0 or missing:
                reason = f"exit code {return_code}" if return_code != 0 else 'missing ' + ', '.join(missing)
                st.session_state['logs'][step] = ('error', f"Failed for sub-{subject} ({reason}). Stopped.",
                                                  '\n'.join(logs))
                failed = True
                break
        if failed:
            break
        st.session_state['logs'][step] = ('success', f"Completed for {len(subjects)} participant(s).",
                                          '\n'.join(logs))
    st.rerun()
