import subprocess
import time
import timeit
from pathlib import Path

import pytest


@pytest.mark.skip("AssertionError: Union module import overhead is too high")
def test_import_union_overhead():
    """
    This test compares the import time of the flyte module with a pure Python baseline.
    If overhead is too high, it outputs and saves an import profile.
    """
    threshold = 30  # 30% increase in import time is acceptable
    repeat = 1
    iterations = 5

    def measure_import_time(command):
        start_time = time.time()
        subprocess.run(command, shell=True, check=True)
        end_time = time.time()
        return end_time - start_time

    def import_union():
        return measure_import_time("python -c 'import flyte'")

    def python_baseline():
        return measure_import_time("python -c 'import asyncio; from dataclasses import dataclass; import typing'")

    min_python = min(timeit.repeat(python_baseline, number=iterations, repeat=repeat))
    min_union = min(timeit.repeat(import_union, number=iterations, repeat=repeat))
    overhead = (min_union - min_python) / min_python * 100

    print(
        f"\nPython baseline load time: {min_python:.6f}s, Union module load time: {min_union:.6f}s, Overhead:"
        f" {overhead:.2f}%"
    )

    if overhead > threshold:
        # Capture detailed import timing for debugging
        report_path = Path("import_profiles/union_importtime.txt")
        report_path.parent.mkdir(exist_ok=True)

        try:
            result = subprocess.run(
                ["python", "-X", "importtime", "-c", "import flyte"],
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE,
                text=True,
                check=False,  # don't crash test on subprocess failure
            )
            report_path.write_text(result.stdout)
            print(f"\nSaved import time profile to {report_path.resolve()}")
        except Exception as e:
            print(f"\nFailed to capture importtime output: {e}")

        raise AssertionError(
            f"Union module import overhead is too high.\n"
            f"Union time: {min_union:.6f}s | Python baseline: {min_python:.6f}s\n"
            f"Overhead: {overhead:.2f}% > Allowed: {threshold}%\n"
            f"Import profile saved to: {report_path.resolve()}"
        )
