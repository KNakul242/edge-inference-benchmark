"""Result writer — serialises BenchmarkResult to JSON and CSV.

One JSON file is written per runtime × precision combination. A summary CSV
aggregates all results for use in results_analysis.ipynb. Both formats must
preserve every field in BenchmarkResult with no data loss.
"""

import csv
import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

from src.results.result_schema import BenchmarkResult

logger = logging.getLogger(__name__)

_CSV_FILENAME = "summary.csv"


def _serialise_cell(value: Any) -> str | float | int:
    """Prepare a dataclass field value for a CSV cell.

    Rules:
    - None → empty string (distinguishable from 0.0 and "None" string)
    - dict or list → JSON string (parseable by pd.read_csv + json.loads)
    - Anything else → pass through (str/float/int handled natively by csv module)
    """
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return value


class ResultWriter:
    """Writes benchmark results to JSON and CSV without data loss.

    Args:
        output_dir: Directory where result files are written. Created if absent.
    """

    def __init__(self, output_dir: str) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, result: BenchmarkResult) -> str:
        """Serialise a single BenchmarkResult to a JSON file.

        File name encodes runtime and precision for unambiguous identification:
        ``{runtime}_{precision}.json``.

        Args:
            result: Completed benchmark record to serialise.

        Returns:
            Absolute path to the written JSON file.
        """
        # result.runtime already encodes precision (e.g. "onnx_cpu_fp32")
        filename = f"{result.runtime}.json"
        path = self._output_dir / filename

        if path.exists():
            logger.warning("Overwriting existing result file: %s", path)

        data = dataclasses.asdict(result)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Result written: %s", path)
        return str(path)

    def write_csv(self, results: list[BenchmarkResult]) -> str:
        """Serialise a list of BenchmarkResults to the summary CSV.

        If the CSV already exists, it is overwritten. All runs must be
        collected before calling this method.

        Args:
            results: List of completed benchmark records.

        Returns:
            Absolute path to the written CSV file.
        """
        path = self._output_dir / _CSV_FILENAME

        if not results:
            logger.warning("write_csv called with empty results list")
            return str(path)

        fieldnames = list(dataclasses.asdict(results[0]).keys())

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                row = dataclasses.asdict(result)
                row = {k: _serialise_cell(v) for k, v in row.items()}
                writer.writerow(row)

        logger.info("Summary CSV written: %s (%d rows)", path, len(results))
        return str(path)
