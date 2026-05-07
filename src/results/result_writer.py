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

from src.results.result_schema import BenchmarkResult

logger = logging.getLogger(__name__)

_CSV_FILENAME = "summary.csv"


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
                writer.writerow(dataclasses.asdict(result))

        logger.info("Summary CSV written: %s (%d rows)", path, len(results))
        return str(path)
