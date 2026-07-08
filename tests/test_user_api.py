from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from loaders import user_api


class UserApiTest(unittest.TestCase):
    def test_load_symbols_formats_market_bars(self) -> None:
        raw_bars = pd.DataFrame(
            [
                {
                    "_id": "1",
                    "time": "2024-01-02",
                    "symbol": "VNM",
                    "open": 10,
                    "high": 12,
                    "low": 9,
                    "close": 11,
                    "volume": 100,
                },
                {
                    "_id": "2",
                    "time": "2024-01-01",
                    "symbol": "VNM",
                    "open": 9,
                    "high": 11,
                    "low": 8,
                    "close": 10,
                    "volume": 90,
                },
                {
                    "_id": "3",
                    "time": "2024-01-01",
                    "symbol": "FPT",
                    "open": 20,
                    "high": 22,
                    "low": 19,
                    "close": 21,
                    "volume": 200,
                },
            ]
        )

        with patch.object(user_api, "load_market_bars", return_value=raw_bars) as mock_loader:
            result = user_api.load_symbols(
                ["VNM", "FPT"],
                start="2024-01-01",
                end="2024-02-01",
                limit=3,
            )

        mock_loader.assert_called_once_with(
            symbols=["VNM", "FPT"],
            start="2024-01-01",
            end="2024-02-01",
            limit=3,
            frame="pandas",
        )
        self.assertEqual(result.index.name, "time")
        self.assertEqual(
            result.columns.tolist(),
            ["symbol", "open", "high", "low", "close", "volume"],
        )
        self.assertNotIn("_id", result.columns)
        self.assertEqual(
            result.reset_index()[["symbol", "time"]].to_dict("records"),
            [
                {"symbol": "FPT", "time": pd.Timestamp("2024-01-01")},
                {"symbol": "VNM", "time": pd.Timestamp("2024-01-01")},
                {"symbol": "VNM", "time": pd.Timestamp("2024-01-02")},
            ],
        )

    def test_load_symbol_calls_single_symbol_loader(self) -> None:
        raw_bars = pd.DataFrame(
            [
                {
                    "_id": "1",
                    "time": "2024-01-01",
                    "symbol": "FPT",
                    "open": 20,
                    "high": 22,
                    "low": 19,
                    "close": 21,
                    "volume": 200,
                }
            ]
        )

        with patch.object(user_api, "load_market_bars", return_value=raw_bars) as mock_loader:
            result = user_api.load_symbol("FPT", limit=1)

        mock_loader.assert_called_once_with(
            symbols="FPT",
            start=None,
            end=None,
            limit=1,
            frame="pandas",
        )
        self.assertEqual(result["symbol"].tolist(), ["FPT"])

    def test_load_high_liquid_symbols_reads_deduplicated_file_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            universe_path = Path(temp_dir) / "high_liquid.csv"
            pd.DataFrame(
                {
                    "symbol": ["FPT", "VNM", "FPT", " ", "HPG", None],
                    "time": ["2024-01-01"] * 6,
                    "volume": [1, 2, 3, 4, 5, 6],
                    "avg_volume_10d": [1, 2, 3, 4, 5, 6],
                }
            ).to_csv(universe_path, index=False)

            result = user_api.load_high_liquid_symbols(universe_path)

        self.assertEqual(result, ["FPT", "VNM", "HPG"])

    def test_update_high_liquid_symbols_writes_csv_and_returns_symbols(self) -> None:
        rows = []
        for day in range(1, 12):
            rows.append(
                {
                    "symbol": "AAA",
                    "time": f"2024-01-{day:02d}",
                    "volume": 2_000_000,
                }
            )
            rows.append(
                {
                    "symbol": "BBB",
                    "time": f"2024-01-{day:02d}",
                    "volume": 500_000,
                }
            )
        raw_bars = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as temp_dir:
            universe_path = Path(temp_dir) / "high_liquid.csv"
            with patch.object(user_api, "load_market_bars", return_value=raw_bars) as mock_loader:
                result = user_api.update_high_liquid_symbols(universe_path)

            saved = pd.read_csv(universe_path)

        mock_loader.assert_called_once_with(frame="pandas")
        self.assertEqual(result, ["AAA"])
        self.assertEqual(
            saved.columns.tolist(),
            ["symbol", "time", "volume", "avg_volume_10d"],
        )
        self.assertEqual(saved["symbol"].tolist(), ["AAA"])
        self.assertEqual(saved.loc[0, "time"], "2024-01-11")
        self.assertEqual(saved.loc[0, "avg_volume_10d"], 2_000_000.0)


if __name__ == "__main__":
    unittest.main()
