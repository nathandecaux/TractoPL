import pytest

from TractoPL.dashboard.streamlit_launcher import DEFAULTS, build_command


def params(**overrides):
    return {**DEFAULTS, 'db_root': 'bids', 'dataset': 'bids', 'atlas': 'atlas.json',
            'msmt_pipeline': 'msmt_csd', 'projection_mcm_pipeline': 'mcm_tensors_restricted',
            'projection_bundle_pipeline': 'bundle_seg', **overrides}


@pytest.mark.parametrize('step,setting', [('bundle_seg', 'bundle_seg_overwrite'),
                                          ('mcm_projection', 'projection_overwrite')])
def test_overwrite_is_off_by_default_and_opt_in(step, setting):
    assert '--overwrite' not in build_command(step, '001', params())
    assert '--overwrite' in build_command(step, '001', params(**{setting: True}))


@pytest.mark.parametrize('step', ['bundle_seg', 'mcm_projection'])
def test_msmt_recompute_does_not_reach_other_steps(step):
    command = build_command(step, '001', params(force=True))
    assert '--overwrite' not in command and '--force' not in command
