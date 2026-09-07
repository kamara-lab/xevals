"""Public import compatibility after grouping implementation modules."""

import importlib
import pickle
import subprocess
import sys

import pytest

GROUPS = {
    "core": "types errors registry seeding dimensions config",
    "evaluation": "runner bench suites metrics perturbations judges",
    "integrations": "adapters datasets",
    "environments": "envs robots sim",
    "reporting": "results report charts plots media tradeoffs",
}


@pytest.mark.parametrize(
    ("group", "name"),
    [(group, name) for group, names in GROUPS.items() for name in names.split()],
)
def test_legacy_import_is_the_same_module(group, name):
    legacy = importlib.import_module(f"xevals.{name}")
    implementation = importlib.import_module(f"xevals.{group}.{name}")
    assert legacy is implementation
    assert getattr(importlib.import_module("xevals"), name) is implementation


def test_objects_pickled_with_old_module_paths_still_load():
    from xevals.core.types import Trajectory

    assert pickle.loads(b"cxevals.types\nTrajectory\n.") is Trajectory
    assert type(pickle.loads(pickle.dumps(Trajectory(seed=7)))) is Trajectory


def test_imports_work_without_extras_and_simulators_stay_lazy():
    script = '''
import importlib
import importlib.abc
import pkgutil
import sys

class NoExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'torch', 'jax', 'gymnasium', 'mujoco', 'newton', 'robot_descriptions',
            'lerobot', 'pyarrow', 'h5py', 'av', 'huggingface_hub', 'imageio',
            'PIL', 'matplotlib', 'anthropic', 'openai', 'xwm',
        }:
            raise ImportError(f'Optional dependency imported: {fullname}')

sys.meta_path.insert(0, NoExtras())
import xevals
assert 'xevals.environments.sim' not in sys.modules
for module in pkgutil.walk_packages(xevals.__path__, xevals.__name__ + '.'):
    importlib.import_module(module.name)
assert xevals.sim is importlib.import_module('xevals.environments.sim')
result = xevals.evaluate(
    lambda obs: obs['goal'] - obs['state'][:2],
    'synthetic/reach', suite='smoke', episodes=1, seeds=(0,),
    out=None, verbose=False,
)
assert result.scores['accuracy'] is not None
'''
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
