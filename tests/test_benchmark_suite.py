import os
from pathlib import Path

from experiments import benchmark_suite


def test_benchmark_suite_creates_results(tmp_path):
    results = benchmark_suite.run_suite("AB", seeds=2, steps=3, results_dir=tmp_path)
    assert len(results) == 4
    assert any(p.suffix == ".csv" for p in tmp_path.iterdir())


def test_summary_runs_for_configs(capsys):
    results = [
        benchmark_suite.BenchmarkResult("A", 0, 0.1, 0.9, 0, 0.0),
        benchmark_suite.BenchmarkResult("A", 1, 0.2, 0.8, 0, 0.1),
        benchmark_suite.BenchmarkResult("B", 0, 0.3, 0.7, 1, 0.2),
    ]
    benchmark_suite.summarize(results)
    captured = capsys.readouterr().out
    assert "Config A" in captured
    assert "Config B" in captured
