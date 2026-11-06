import shutil
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from TractoPL.pipeline import bundle_seg
from TractoPL.utils import registration


class FakeSubject:
    sub_id = '001'

    def __init__(self, fa_path, mask_path, xfm_done=False):
        self.files = {'FA': SimpleNamespace(path=str(fa_path)), 'mask': SimpleNamespace(path=str(mask_path))}
        self.xfm_done = xfm_done

    def get_unique(self, **kwargs):
        return self.files['FA' if kwargs.get('metric') == 'FA' else 'mask']

    def get(self, **kwargs):
        return [object()] if self.xfm_done else []


@pytest.fixture
def images(tmp_path):
    fa = np.ones((4, 4, 4))
    mask = np.zeros((4, 4, 4))
    mask[1:3, 1:3, 1:3] = 1
    nib.save(nib.Nifti1Image(fa, np.eye(4)), tmp_path / 'fa.nii.gz')
    nib.save(nib.Nifti1Image(mask, np.eye(4)), tmp_path / 'mask.nii.gz')
    return tmp_path / 'fa.nii.gz', tmp_path / 'mask.nii.gz', mask


def test_write_masked_fa_zeroes_the_background(images, tmp_path):
    fa, mask_path, mask = images
    output = bundle_seg.write_masked_fa(FakeSubject(fa, mask_path), str(tmp_path))
    assert np.array_equal(nib.load(output).get_fdata(), mask)


def test_write_masked_fa_refuses_different_grids(images, tmp_path):
    fa, _, _ = images
    nib.save(nib.Nifti1Image(np.ones((3, 3, 3)), np.eye(4)), tmp_path / 'small.nii.gz')
    with pytest.raises(ValueError, match='different grids'):
        bundle_seg.write_masked_fa(FakeSubject(fa, tmp_path / 'small.nii.gz'), str(tmp_path))


@pytest.mark.parametrize('xfm_done,force,expected', [(True, False, 0), (True, True, 1), (False, False, 1)])
def test_registration_skip_respects_force(images, monkeypatch, xfm_done, force, expected):
    fa, mask_path, _ = images
    fixed = []

    def fake_registration(moving, fixed_path, **kwargs):
        fixed.append(np.count_nonzero(nib.load(fixed_path).get_fdata()))
        return {}

    monkeypatch.setattr(bundle_seg, 'ants_registration', fake_registration)
    monkeypatch.setattr(bundle_seg, 'copy_from_dict', lambda *args, **kwargs: None)
    atlas = SimpleNamespace(name='atlas', reference='atlas.nii.gz')
    bundle_seg.register_template_to_anat(FakeSubject(fa, mask_path, xfm_done), 'bundle_seg', atlas, force=force)
    assert len(fixed) == expected
    assert all(count == 8 for count in fixed)  # the masked FA reaches the registration


def test_synquick_runs_the_quick_script(monkeypatch, tmp_path):
    commands = []

    def fake_cli(command, inputs, output_pattern, base_entities, command_args=None):
        commands.append(command)
        return {str(tmp_path / 'x.nii.gz'): {}}

    monkeypatch.setattr(registration, 'run_cli_command', fake_cli)
    monkeypatch.setattr(registration, 'writeTransformSerie', lambda *args, **kwargs: str(tmp_path / 'series.xml'))
    registration.ants_registration('a.nii.gz', 'b.nii.gz', transform_type='synquick', moving_space='m', fixed_space='f')
    assert commands == ['antsRegistrationSyNQuick.sh']


@pytest.mark.parametrize('flag', ['--overwrite', '--force'])
def test_segmentation_cli_accepts_overwrite(monkeypatch, flag):
    monkeypatch.setattr('sys.argv', ['bundle_seg', 'segmentation', '-a', 'atlas.json', '-s', '001', '-d', 'bids', flag])
    assert bundle_seg.parse_args().overwrite is True


def test_centroid_of_T_bundle_is_not_taken_from_its_ST_twin(monkeypatch, tmp_path):
    atlas_dir, out_dir = tmp_path / 'atlas', tmp_path / 'out'
    atlas_dir.mkdir()
    out_dir.mkdir()
    for name in ('T_PREF_left', 'ST_PREF_left'):
        (atlas_dir / f'atlas_{name}.vtk').touch()
    atlas = SimpleNamespace(name='MyAtlas', reference='ref.nii.gz', bundles_pattern=str(atlas_dir / '*_{BUNDLE}.vtk'),
                            bundle_mapping={'TPREFleft': 'T_PREF_left'})
    (out_dir / 'atlas_ST_PREF_left_centroids.vtk').touch()  # already written by the thread of the twin
    subject = SimpleNamespace(sub_id='001', get_unique=lambda **kwargs: SimpleNamespace(
        path='x', get_full_entities=lambda: {}))
    commands = []

    def fake_call(command, shell):
        commands.append(command)
        if command.startswith('tractopl-generate-centroid'):
            (out_dir / 'atlas_T_PREF_left_centroids.vtk').touch()

    monkeypatch.setattr(bundle_seg, 'call', fake_call)
    monkeypatch.setattr(bundle_seg, 'copy_from_dict', lambda *args, **kwargs: None)
    bundle_seg.generate_cluster_and_apply_transformation(subject, atlas, out_dir=str(out_dir))
    assert f'tractopl-apply-trans-to-vtk {out_dir}/atlas_T_PREF_left_centroids.vtk' in commands[1]


def test_centroid_temporary_folder_is_removed(monkeypatch, tmp_path):
    made = tmp_path / 'made'
    made.mkdir()
    monkeypatch.setattr(bundle_seg.tempfile, 'mkdtemp', lambda: str(made))
    monkeypatch.setattr(bundle_seg, 'copy_from_dict', lambda *args, **kwargs: None)
    atlas = SimpleNamespace(name='MyAtlas', reference='ref.nii.gz', bundles_pattern='', bundle_mapping={})
    subject = SimpleNamespace(sub_id='001', get_unique=lambda **kwargs: SimpleNamespace(path='x'))
    bundle_seg.generate_cluster_and_apply_transformation(subject, atlas)
    assert not made.exists()
