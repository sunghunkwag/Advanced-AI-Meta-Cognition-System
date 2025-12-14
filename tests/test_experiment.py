import csv
import os

from config import ExperimentConfig
from experiments.run_experiment import run


def test_run_experiment_creates_log(tmp_path):
    cfg = ExperimentConfig()
    cfg.steps = 5
    cfg.log_dir = tmp_path.as_posix()
    cfg.planner.enabled = False
    cfg.meta.enabled = False
    path = run(cfg)
    assert os.path.exists(path)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == cfg.steps
