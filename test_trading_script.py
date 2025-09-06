"""Comprehensive test suite for trading_script.py with full coverage and edge cases.

This test suite covers:
- Market data fetching with various failure scenarios
- Portfolio operations (buy/sell with different conditions)
- Date handling and weekend logic
- File I/O operations (CSV and JSON handling)
- Performance metrics calculation
- Configuration management
- Trade logging
- Interactive functionality
- Error handling and fallbacks
"""

import json
import os
import sys
import tempfile
import unittest
import importlib
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, mock_open
from io import StringIO
import warnings

import numpy as np
import pandas as pd
import pytest

# Import the module under test
import trading_script as ts


class TestDateUtilities(unittest.TestCase):
    """Test date-related utility functions."""

    def setUp(self):
        """Reset global state before each test."""
        ts.ASOF_DATE = None

    def test_set_asof_with_string(self):
        """Test setting as-of date with string input."""
        ts.set_asof("2024-01-15")
        expected = pd.Timestamp("2024-01-15").normalize()
        self.assertEqual(ts.ASOF_DATE, expected)

    def test_set_asof_with_datetime(self):
        """Test setting as-of date with datetime input."""
        dt = datetime(2024, 1, 15, 14, 30)
        ts.set_asof(dt)
        expected = pd.Timestamp("2024-01-15").normalize()
        self.assertEqual(ts.ASOF_DATE, expected)

    def test_set_asof_with_none(self):
        """Test setting as-of date with None."""
        ts.set_asof("2024-01-15")  # Set first
        ts.set_asof(None)
        self.assertIsNone(ts.ASOF_DATE)

    def test_effective_now_with_asof(self):
        """Test _effective_now when ASOF_DATE is set."""
        ts.set_asof("2024-01-15")
        result = ts._effective_now()
        expected = datetime(2024, 1, 15)
        self.assertEqual(result.date(), expected.date())

    @patch("trading_script.datetime")
    def test_effective_now_without_asof(self, mock_datetime):
        """Test _effective_now when ASOF_DATE is None."""
        mock_now = datetime(2024, 1, 15, 10, 30)
        mock_datetime.now.return_value = mock_now

        ts.ASOF_DATE = None
        result = ts._effective_now()
        self.assertEqual(result, mock_now)

    def test_last_trading_date_weekday(self):
        """Test last_trading_date on a weekday."""
        # Tuesday
        tuesday = datetime(2024, 1, 16)
        result = ts.last_trading_date(tuesday)
        expected = pd.Timestamp("2024-01-16").normalize()
        self.assertEqual(result, expected)

    def test_last_trading_date_saturday(self):
        """Test last_trading_date on Saturday -> Friday."""
        saturday = datetime(2024, 1, 13)  # Saturday
        result = ts.last_trading_date(saturday)
        expected = pd.Timestamp("2024-01-12").normalize()  # Friday
        self.assertEqual(result, expected)

    def test_last_trading_date_sunday(self):
        """Test last_trading_date on Sunday -> Friday."""
        sunday = datetime(2024, 1, 14)  # Sunday
        result = ts.last_trading_date(sunday)
        expected = pd.Timestamp("2024-01-12").normalize()  # Friday
        self.assertEqual(result, expected)

    def test_check_weekend(self):
        """Test check_weekend function."""
        with patch("trading_script.last_trading_date") as mock_ltd:
            mock_ltd.return_value = pd.Timestamp("2024-01-15")
            result = ts.check_weekend()
            self.assertEqual(result, "2024-01-15")

    def test_trading_day_window(self):
        """Test trading_day_window function."""
        target = datetime(2024, 1, 15)
        start, end = ts.trading_day_window(target)
        expected_start = pd.Timestamp("2024-01-15").normalize()
        expected_end = pd.Timestamp("2024-01-16").normalize()
        self.assertEqual(start, expected_start)
        self.assertEqual(end, expected_end)


class TestConfigurationHelpers(unittest.TestCase):
    """Test configuration loading functions."""

    def test_read_json_file_success(self):
        """Test successful JSON file reading."""
        test_data = {"benchmarks": ["SPY", "IWM"]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = Path(f.name)

        try:
            result = ts._read_json_file(temp_path)
            self.assertEqual(result, test_data)
        finally:
            temp_path.unlink()

    def test_read_json_file_not_found(self):
        """Test JSON file reading when file doesn't exist."""
        result = ts._read_json_file(Path("/nonexistent/file.json"))
        self.assertIsNone(result)

    def test_read_json_file_malformed(self):
        """Test JSON file reading with malformed JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json }")
            temp_path = Path(f.name)

        try:
            with patch("trading_script.logger") as mock_logger:
                result = ts._read_json_file(temp_path)
                self.assertIsNone(result)
                mock_logger.warning.assert_called_once()
        finally:
            temp_path.unlink()

    def test_load_benchmarks_no_file(self):
        """Test loading benchmarks when no tickers.json exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ts.load_benchmarks(Path(tmpdir))
            self.assertEqual(result, ts.DEFAULT_BENCHMARKS.copy())

    def test_load_benchmarks_valid_file(self):
        """Test loading benchmarks from valid tickers.json."""
        test_benchmarks = ["CUSTOM1", "CUSTOM2", "SPY"]
        test_data = {"benchmarks": test_benchmarks}

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "tickers.json"
            with open(json_path, "w") as f:
                json.dump(test_data, f)

            result = ts.load_benchmarks(Path(tmpdir))
            self.assertEqual(result, test_benchmarks)

    def test_load_benchmarks_missing_key(self):
        """Test loading benchmarks when 'benchmarks' key is missing."""
        test_data = {"other_key": ["value"]}

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "tickers.json"
            with open(json_path, "w") as f:
                json.dump(test_data, f)

            with patch("trading_script.logger") as mock_logger:
                result = ts.load_benchmarks(Path(tmpdir))
                self.assertEqual(result, ts.DEFAULT_BENCHMARKS.copy())
                mock_logger.warning.assert_called_once()

    def test_load_benchmarks_invalid_benchmarks(self):
        """Test loading benchmarks when 'benchmarks' is not a list."""
        test_data = {"benchmarks": "not_a_list"}

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "tickers.json"
            with open(json_path, "w") as f:
                json.dump(test_data, f)

            with patch("trading_script.logger") as mock_logger:
                result = ts.load_benchmarks(Path(tmpdir))
                self.assertEqual(result, ts.DEFAULT_BENCHMARKS.copy())
                mock_logger.warning.assert_called_once()

    def test_load_benchmarks_deduplication(self):
        """Test that duplicate benchmarks are removed while preserving order."""
        test_benchmarks = ["SPY", "IWM", "spy", "SPY", "XBI"]
        expected = ["SPY", "IWM", "XBI"]  # Normalized and deduplicated
        test_data = {"benchmarks": test_benchmarks}

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "tickers.json"
            with open(json_path, "w") as f:
                json.dump(test_data, f)

            result = ts.load_benchmarks(Path(tmpdir))
            self.assertEqual(result, expected)

    def test_load_benchmarks_parent_directory(self):
        """Test loading benchmarks from parent directory."""
        test_benchmarks = ["PARENT1", "PARENT2"]
        test_data = {"benchmarks": test_benchmarks}

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create subdirectory
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()

            # Put tickers.json in parent
            json_path = Path(tmpdir) / "tickers.json"
            with open(json_path, "w") as f:
                json.dump(test_data, f)

            result = ts.load_benchmarks(subdir)
            self.assertEqual(result, test_benchmarks)


class TestDataAccess(unittest.TestCase):
    """Test data access layer functions."""

    def test_to_datetime_index(self):
        """Test _to_datetime_index function."""
        df = pd.DataFrame(
            {"A": [1, 2, 3]}, index=["2024-01-01", "2024-01-02", "2024-01-03"]
        )
        result = ts._to_datetime_index(df)
        self.assertIsInstance(result.index, pd.DatetimeIndex)

    def test_normalize_ohlcv_complete(self):
        """Test _normalize_ohlcv with complete data."""
        df = pd.DataFrame(
            {
                "Open": [100, 101],
                "High": [102, 103],
                "Low": [99, 100],
                "Close": [101, 102],
                "Adj Close": [101, 102],
                "Volume": [1000, 1100],
            }
        )
        result = ts._normalize_ohlcv(df)
        expected_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        self.assertListEqual(list(result.columns), expected_cols)

    def test_normalize_ohlcv_missing_columns(self):
        """Test _normalize_ohlcv with missing columns."""
        df = pd.DataFrame({"Close": [101, 102], "Volume": [1000, 1100]})
        result = ts._normalize_ohlcv(df)
        expected_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        self.assertListEqual(list(result.columns), expected_cols)
        # Adj Close should equal Close when missing
        pd.testing.assert_series_equal(
            result["Adj Close"], result["Close"], check_names=False
        )

    @patch("trading_script.yf.download")
    @patch("requests.Session")
    def test_yahoo_download_success(self, mock_session, mock_download):
        """Test successful Yahoo download."""
        mock_df = pd.DataFrame(
            {
                "Open": [100],
                "High": [102],
                "Low": [99],
                "Close": [101],
                "Volume": [1000],
            }
        )
        mock_download.return_value = mock_df

        result = ts._yahoo_download("AAPL", start="2024-01-01", end="2024-01-02")
        self.assertFalse(result.empty)
        mock_download.assert_called_once()

    @patch("trading_script.yf.download")
    def test_yahoo_download_exception(self, mock_download):
        """Test Yahoo download with exception."""
        mock_download.side_effect = Exception("Network error")

        result = ts._yahoo_download("AAPL")
        self.assertTrue(result.empty)

    @patch("requests.get")
    def test_stooq_csv_download_success(self, mock_get):
        """Test successful Stooq CSV download."""
        csv_data = "Date,Open,High,Low,Close,Volume\n2024-01-01,100,102,99,101,1000\n"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = csv_data
        mock_get.return_value = mock_response

        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-01-02")
        result = ts._stooq_csv_download("AAPL", start, end)

        self.assertFalse(result.empty)
        self.assertIn("Adj Close", result.columns)

    @patch("requests.get")
    def test_stooq_csv_download_failure(self, mock_get):
        """Test Stooq CSV download failure."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-01-02")
        result = ts._stooq_csv_download("AAPL", start, end)

        self.assertTrue(result.empty)

    def test_stooq_csv_download_blocklist(self):
        """Test Stooq CSV download with blocklisted ticker."""
        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-01-02")
        result = ts._stooq_csv_download("^RUT", start, end)
        self.assertTrue(result.empty)

    @patch("trading_script._HAS_PDR", True)
    @patch("trading_script.pd.read_csv")
    def test_stooq_download_success(self, mock_read_csv):
        """Test successful Stooq download via pandas-datareader."""
        mock_df = pd.DataFrame(
            {
                "Open": [100],
                "High": [102],
                "Low": [99],
                "Close": [101],
                "Volume": [1000],
            },
            index=pd.DatetimeIndex(["2024-01-01"]),
        )

        # Mock the import and DataReader
        mock_pdr = Mock()
        mock_pdr.DataReader.return_value = mock_df

        # Set up the module hierarchy correctly
        pandas_datareader_mock = Mock()
        pandas_datareader_mock.data = mock_pdr

        with patch.dict(
            "sys.modules",
            {
                "pandas_datareader": pandas_datareader_mock,
                "pandas_datareader.data": mock_pdr,
            },
        ):
            start = datetime(2024, 1, 1)
            end = datetime(2024, 1, 2)
            result = ts._stooq_download("AAPL", start, end)
            self.assertFalse(result.empty)

    def test_stooq_download_no_pdr(self):
        """Test Stooq download when pandas-datareader is not available."""
        with patch("trading_script._HAS_PDR", False):
            start = datetime(2024, 1, 1)
            end = datetime(2024, 1, 2)
            result = ts._stooq_download("AAPL", start, end)
            self.assertTrue(result.empty)

    def test_weekend_safe_range_explicit_dates(self):
        """Test _weekend_safe_range with explicit start/end dates."""
        start = "2024-01-01"
        end = "2024-01-05"
        result_start, result_end = ts._weekend_safe_range(None, start, end)

        expected_start = pd.Timestamp("2024-01-01").normalize()
        expected_end = pd.Timestamp("2024-01-05").normalize()
        self.assertEqual(result_start, expected_start)
        self.assertEqual(result_end, expected_end)

    @patch("trading_script.last_trading_date")
    def test_weekend_safe_range_period(self, mock_ltd):
        """Test _weekend_safe_range with period parameter."""
        mock_ltd.return_value = pd.Timestamp("2024-01-15")

        result_start, result_end = ts._weekend_safe_range("5d", None, None)

        expected_start = pd.Timestamp("2024-01-10").normalize()
        expected_end = pd.Timestamp("2024-01-16").normalize()
        self.assertEqual(result_start, expected_start)
        self.assertEqual(result_end, expected_end)

    @patch("trading_script._yahoo_download")
    def test_download_price_data_yahoo_success(self, mock_yahoo):
        """Test download_price_data with successful Yahoo fetch."""
        mock_df = pd.DataFrame(
            {
                "Open": [100],
                "High": [102],
                "Low": [99],
                "Close": [101],
                "Adj Close": [101],
                "Volume": [1000],
            }
        )
        mock_yahoo.return_value = mock_df

        result = ts.download_price_data("AAPL", period="1d")
        self.assertEqual(result.source, "yahoo")
        self.assertFalse(result.df.empty)

    @patch("trading_script._yahoo_download")
    @patch("trading_script._stooq_download")
    def test_download_price_data_fallback_to_stooq(self, mock_stooq, mock_yahoo):
        """Test download_price_data fallback to Stooq when Yahoo fails."""
        mock_yahoo.return_value = pd.DataFrame()  # Empty
        mock_df = pd.DataFrame(
            {
                "Open": [100],
                "High": [102],
                "Low": [99],
                "Close": [101],
                "Volume": [1000],
            }
        )
        mock_stooq.return_value = mock_df

        result = ts.download_price_data("AAPL", period="1d")
        self.assertEqual(result.source, "stooq-pdr")
        self.assertFalse(result.df.empty)

    @patch("trading_script._yahoo_download")
    @patch("trading_script._stooq_download")
    @patch("trading_script._stooq_csv_download")
    def test_download_price_data_fallback_to_csv(
        self, mock_csv, mock_stooq, mock_yahoo
    ):
        """Test download_price_data fallback to Stooq CSV."""
        mock_yahoo.return_value = pd.DataFrame()
        mock_stooq.return_value = pd.DataFrame()
        mock_df = pd.DataFrame(
            {
                "Open": [100],
                "High": [102],
                "Low": [99],
                "Close": [101],
                "Adj Close": [101],
                "Volume": [1000],
            }
        )
        mock_csv.return_value = mock_df

        result = ts.download_price_data("AAPL", period="1d")
        self.assertEqual(result.source, "stooq-csv")
        self.assertFalse(result.df.empty)

    @patch("trading_script._yahoo_download")
    @patch("trading_script._stooq_download")
    @patch("trading_script._stooq_csv_download")
    def test_download_price_data_proxy_fallback(self, mock_csv, mock_stooq, mock_yahoo):
        """Test download_price_data proxy fallback for indices."""
        # First call (original ticker) returns empty
        # Second call (proxy) returns data
        mock_yahoo.side_effect = [
            pd.DataFrame(),  # Original ticker fails
            pd.DataFrame(
                {  # Proxy succeeds
                    "Open": [100],
                    "High": [102],
                    "Low": [99],
                    "Close": [101],
                    "Adj Close": [101],
                    "Volume": [1000],
                }
            ),
        ]
        mock_stooq.return_value = pd.DataFrame()
        mock_csv.return_value = pd.DataFrame()

        result = ts.download_price_data("^GSPC", period="1d")
        self.assertEqual(result.source, "yahoo:SPY-proxy")
        self.assertFalse(result.df.empty)

    @patch("trading_script._yahoo_download")
    @patch("trading_script._stooq_download")
    @patch("trading_script._stooq_csv_download")
    def test_download_price_data_all_fail(self, mock_csv, mock_stooq, mock_yahoo):
        """Test download_price_data when all sources fail."""
        mock_yahoo.return_value = pd.DataFrame()
        mock_stooq.return_value = pd.DataFrame()
        mock_csv.return_value = pd.DataFrame()

        result = ts.download_price_data("INVALID", period="1d")
        self.assertEqual(result.source, "empty")
        self.assertTrue(result.df.empty)


class TestFileOperations(unittest.TestCase):
    """Test file path configuration and operations."""

    def test_set_data_dir(self):
        """Test setting data directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_data_dir = ts.DATA_DIR
            original_portfolio_csv = ts.PORTFOLIO_CSV
            original_trade_log_csv = ts.TRADE_LOG_CSV

            try:
                ts.set_data_dir(Path(tmpdir))
                self.assertEqual(ts.DATA_DIR, Path(tmpdir))
                self.assertEqual(
                    ts.PORTFOLIO_CSV, Path(tmpdir) / "chatgpt_portfolio_update.csv"
                )
                self.assertEqual(
                    ts.TRADE_LOG_CSV, Path(tmpdir) / "chatgpt_trade_log.csv"
                )
            finally:
                # Restore original values
                ts.DATA_DIR = original_data_dir
                ts.PORTFOLIO_CSV = original_portfolio_csv
                ts.TRADE_LOG_CSV = original_trade_log_csv


class TestPortfolioOperations(unittest.TestCase):
    """Test portfolio-related operations."""

    def test_ensure_df_with_dataframe(self):
        """Test _ensure_df with DataFrame input."""
        df = pd.DataFrame({"ticker": ["AAPL"], "shares": [100]})
        result = ts._ensure_df(df)
        pd.testing.assert_frame_equal(result, df)
        # Should be a copy
        self.assertIsNot(result, df)

    def test_ensure_df_with_dict(self):
        """Test _ensure_df with dict input."""
        data = {"ticker": ["AAPL"], "shares": [100]}
        result = ts._ensure_df(data)
        expected = pd.DataFrame(data)
        pd.testing.assert_frame_equal(result, expected)

    def test_ensure_df_with_list_of_dicts(self):
        """Test _ensure_df with list of dicts input."""
        data = [{"ticker": "AAPL", "shares": 100}, {"ticker": "MSFT", "shares": 50}]
        result = ts._ensure_df(data)
        expected = pd.DataFrame(data)
        pd.testing.assert_frame_equal(result, expected)

    def test_ensure_df_with_invalid_input(self):
        """Test _ensure_df with invalid input."""
        with self.assertRaises(TypeError):
            ts._ensure_df("invalid")

    @patch("trading_script.download_price_data")
    @patch("trading_script.trading_day_window")
    @patch("trading_script.last_trading_date")
    def test_process_portfolio_non_interactive(self, mock_ltd, mock_tdw, mock_dpd):
        """Test process_portfolio in non-interactive mode."""
        # Setup mocks
        mock_ltd.return_value = pd.Timestamp("2024-01-15")
        mock_tdw.return_value = (pd.Timestamp("2024-01-15"), pd.Timestamp("2024-01-16"))

        mock_data = pd.DataFrame(
            {
                "Open": [100],
                "High": [102],
                "Low": [99],
                "Close": [101],
                "Adj Close": [101],
                "Volume": [1000],
            },
            index=pd.DatetimeIndex(["2024-01-15"]),
        )
        mock_dpd.return_value = ts.FetchResult(mock_data, "yahoo")

        portfolio = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "shares": [100],
                "stop_loss": [90],
                "buy_price": [95],
                "cost_basis": [9500],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ts.set_data_dir(Path(tmpdir))

            result_portfolio, result_cash = ts.process_portfolio(
                portfolio, cash=5000, interactive=False
            )

            # Check that portfolio CSV was created
            self.assertTrue(ts.PORTFOLIO_CSV.exists())

    @patch("trading_script.download_price_data")
    @patch("trading_script.trading_day_window")
    @patch("trading_script.last_trading_date")
    def test_process_portfolio_stop_loss_triggered(self, mock_ltd, mock_tdw, mock_dpd):
        """Test process_portfolio with stop loss triggered."""
        mock_ltd.return_value = pd.Timestamp("2024-01-15")
        mock_tdw.return_value = (pd.Timestamp("2024-01-15"), pd.Timestamp("2024-01-16"))

        # Price drops below stop loss
        mock_data = pd.DataFrame(
            {
                "Open": [85],
                "High": [88],
                "Low": [84],
                "Close": [86],
                "Adj Close": [86],
                "Volume": [1000],
            },
            index=pd.DatetimeIndex(["2024-01-15"]),
        )
        mock_dpd.return_value = ts.FetchResult(mock_data, "yahoo")

        portfolio = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "shares": [100],
                "stop_loss": [90],  # Stop loss at 90, low is 84
                "buy_price": [95],
                "cost_basis": [9500],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ts.set_data_dir(Path(tmpdir))

            result_portfolio, result_cash = ts.process_portfolio(
                portfolio, cash=1000, interactive=False
            )

            # Portfolio should be empty (stock sold)
            self.assertTrue(result_portfolio.empty)
            # Cash should increase from sale
            self.assertGreater(result_cash, 1000)

    @patch("trading_script.download_price_data")
    @patch("trading_script.trading_day_window")
    @patch("trading_script.last_trading_date")
    def test_process_portfolio_no_data(self, mock_ltd, mock_tdw, mock_dpd):
        """Test process_portfolio when no market data is available."""
        mock_ltd.return_value = pd.Timestamp("2024-01-15")
        mock_tdw.return_value = (pd.Timestamp("2024-01-15"), pd.Timestamp("2024-01-16"))
        mock_dpd.return_value = ts.FetchResult(pd.DataFrame(), "empty")

        portfolio = pd.DataFrame(
            {
                "ticker": ["INVALID"],
                "shares": [100],
                "stop_loss": [90],
                "buy_price": [95],
                "cost_basis": [9500],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ts.set_data_dir(Path(tmpdir))

            result_portfolio, result_cash = ts.process_portfolio(
                portfolio, cash=1000, interactive=False
            )

            # Check that portfolio CSV was created with NO DATA action
            self.assertTrue(ts.PORTFOLIO_CSV.exists())
            df = pd.read_csv(ts.PORTFOLIO_CSV)
            self.assertIn("NO DATA", df["Action"].values)


class TestTradeLogging(unittest.TestCase):
    """Test trade logging functions."""

    def setUp(self):
        """Set up temporary directory for each test."""
        self.tmpdir = tempfile.mkdtemp()
        ts.set_data_dir(Path(self.tmpdir))

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil

        shutil.rmtree(self.tmpdir)

    @patch("trading_script.check_weekend")
    def test_log_sell(self, mock_check_weekend):
        """Test log_sell function."""
        mock_check_weekend.return_value = "2024-01-15"

        portfolio = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "shares": [100, 50],
                "stop_loss": [90, 200],
                "buy_price": [95, 210],
                "cost_basis": [9500, 10500],
            }
        )

        result_portfolio = ts.log_sell("AAPL", 100, 92, 95, -300, portfolio)

        # AAPL should be removed from portfolio
        self.assertNotIn("AAPL", result_portfolio["ticker"].values)
        self.assertIn("MSFT", result_portfolio["ticker"].values)

        # Trade log should be created
        self.assertTrue(ts.TRADE_LOG_CSV.exists())
        log_df = pd.read_csv(ts.TRADE_LOG_CSV)
        self.assertEqual(len(log_df), 1)
        self.assertEqual(log_df.iloc[0]["Ticker"], "AAPL")
        self.assertEqual(log_df.iloc[0]["PnL"], -300)

    @patch("trading_script.download_price_data")
    @patch("trading_script.trading_day_window")
    @patch("trading_script.check_weekend")
    @patch("builtins.input")
    def test_log_manual_buy_success(
        self, mock_input, mock_check_weekend, mock_tdw, mock_dpd
    ):
        """Test successful manual buy."""
        mock_input.return_value = ""  # Don't cancel
        mock_check_weekend.return_value = "2024-01-15"
        mock_tdw.return_value = (pd.Timestamp("2024-01-15"), pd.Timestamp("2024-01-16"))

        mock_data = pd.DataFrame(
            {
                "Open": [100],
                "High": [102],
                "Low": [99],
                "Close": [101],
                "Adj Close": [101],
                "Volume": [1000],
            },
            index=pd.DatetimeIndex(["2024-01-15"]),
        )
        mock_dpd.return_value = ts.FetchResult(mock_data, "yahoo")

        portfolio = pd.DataFrame(
            columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"]
        )

        cash, new_portfolio = ts.log_manual_buy(
            buy_price=102,
            shares=50,
            ticker="AAPL",
            stoploss=95,
            cash=10000,
            chatgpt_portfolio=portfolio,
        )

        # Should have bought at open price (100)
        self.assertEqual(cash, 5000)  # 10000 - (50 * 100)
        self.assertEqual(len(new_portfolio), 1)
        self.assertEqual(new_portfolio.iloc[0]["ticker"], "AAPL")
        self.assertEqual(new_portfolio.iloc[0]["shares"], 50)

    @patch("trading_script.download_price_data")
    @patch("trading_script.trading_day_window")
    @patch("trading_script.check_weekend")
    @patch("builtins.input")
    def test_log_manual_buy_insufficient_cash(
        self, mock_input, mock_check_weekend, mock_tdw, mock_dpd
    ):
        """Test manual buy with insufficient cash."""
        mock_input.return_value = ""
        mock_check_weekend.return_value = "2024-01-15"
        mock_tdw.return_value = (pd.Timestamp("2024-01-15"), pd.Timestamp("2024-01-16"))

        mock_data = pd.DataFrame(
            {
                "Open": [100],
                "High": [102],
                "Low": [99],
                "Close": [101],
                "Adj Close": [101],
                "Volume": [1000],
            },
            index=pd.DatetimeIndex(["2024-01-15"]),
        )
        mock_dpd.return_value = ts.FetchResult(mock_data, "yahoo")

        portfolio = pd.DataFrame(
            columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"]
        )

        cash, new_portfolio = ts.log_manual_buy(
            buy_price=102,
            shares=100,
            ticker="AAPL",
            stoploss=95,
            cash=1000,
            chatgpt_portfolio=portfolio,  # Only $1000 cash
        )

        # Should not have bought anything
        self.assertEqual(cash, 1000)  # Unchanged
        self.assertTrue(new_portfolio.empty)

    @patch("trading_script.download_price_data")
    @patch("trading_script.trading_day_window")
    @patch("trading_script.check_weekend")
    @patch("builtins.input")
    def test_log_manual_buy_limit_not_reached(
        self, mock_input, mock_check_weekend, mock_tdw, mock_dpd
    ):
        """Test manual buy when limit price is not reached."""
        mock_input.return_value = ""
        mock_check_weekend.return_value = "2024-01-15"
        mock_tdw.return_value = (pd.Timestamp("2024-01-15"), pd.Timestamp("2024-01-16"))

        # Price never goes down to limit
        mock_data = pd.DataFrame(
            {
                "Open": [105],
                "High": [108],
                "Low": [104],
                "Close": [107],
                "Adj Close": [107],
                "Volume": [1000],
            },
            index=pd.DatetimeIndex(["2024-01-15"]),
        )
        mock_dpd.return_value = ts.FetchResult(mock_data, "yahoo")

        portfolio = pd.DataFrame(
            columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"]
        )

        cash, new_portfolio = ts.log_manual_buy(
            buy_price=100,
            shares=50,
            ticker="AAPL",  # Limit at 100, but low is 104
            stoploss=95,
            cash=10000,
            chatgpt_portfolio=portfolio,
        )

        # Should not have bought anything
        self.assertEqual(cash, 10000)  # Unchanged
        self.assertTrue(new_portfolio.empty)

    @patch("trading_script.download_price_data")
    @patch("trading_script.trading_day_window")
    @patch("trading_script.check_weekend")
    @patch("builtins.input")
    def test_log_manual_sell_success(
        self, mock_input, mock_check_weekend, mock_tdw, mock_dpd
    ):
        """Test successful manual sell."""
        mock_input.return_value = "Taking profits"  # Reason
        mock_check_weekend.return_value = "2024-01-15"
        mock_tdw.return_value = (pd.Timestamp("2024-01-15"), pd.Timestamp("2024-01-16"))

        mock_data = pd.DataFrame(
            {
                "Open": [110],
                "High": [112],
                "Low": [109],
                "Close": [111],
                "Adj Close": [111],
                "Volume": [1000],
            },
            index=pd.DatetimeIndex(["2024-01-15"]),
        )
        mock_dpd.return_value = ts.FetchResult(mock_data, "yahoo")

        portfolio = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "shares": [100],
                "stop_loss": [90],
                "buy_price": [95],
                "cost_basis": [9500],
            }
        )

        cash, new_portfolio = ts.log_manual_sell(
            sell_price=109,
            shares_sold=50,
            ticker="AAPL",
            cash=1000,
            chatgpt_portfolio=portfolio,
        )

        # Should have sold at open price (110)
        self.assertEqual(cash, 6500)  # 1000 + (50 * 110)
        self.assertEqual(new_portfolio.iloc[0]["shares"], 50)  # 100 - 50

    @patch("builtins.input")
    def test_log_manual_sell_ticker_not_in_portfolio(self, mock_input):
        """Test manual sell when ticker is not in portfolio."""
        mock_input.return_value = "reason"

        portfolio = pd.DataFrame(
            {
                "ticker": ["MSFT"],
                "shares": [100],
                "stop_loss": [90],
                "buy_price": [95],
                "cost_basis": [9500],
            }
        )

        cash, new_portfolio = ts.log_manual_sell(
            sell_price=100,
            shares_sold=50,
            ticker="AAPL",  # AAPL not in portfolio
            cash=1000,
            chatgpt_portfolio=portfolio,
        )

        # Should not change anything
        self.assertEqual(cash, 1000)
        pd.testing.assert_frame_equal(new_portfolio, portfolio)

    @patch("builtins.input")
    def test_log_manual_sell_insufficient_shares(self, mock_input):
        """Test manual sell when trying to sell more shares than owned."""
        mock_input.return_value = "reason"

        portfolio = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "shares": [50],  # Only 50 shares
                "stop_loss": [90],
                "buy_price": [95],
                "cost_basis": [4750],
            }
        )

        cash, new_portfolio = ts.log_manual_sell(
            sell_price=100,
            shares_sold=100,
            ticker="AAPL",  # Trying to sell 100
            cash=1000,
            chatgpt_portfolio=portfolio,
        )

        # Should not change anything
        self.assertEqual(cash, 1000)
        pd.testing.assert_frame_equal(new_portfolio, portfolio)


class TestReportingMetrics(unittest.TestCase):
    """Test reporting and metrics functions."""

    def setUp(self):
        """Set up temporary directory for each test."""
        self.tmpdir = tempfile.mkdtemp()
        ts.set_data_dir(Path(self.tmpdir))

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil

        shutil.rmtree(self.tmpdir)

    @patch("trading_script.download_price_data")
    @patch("trading_script.load_benchmarks")
    @patch("trading_script.check_weekend")
    @patch("trading_script.last_trading_date")
    @patch("builtins.input")
    def test_daily_results_with_data(
        self, mock_input, mock_ltd, mock_check_weekend, mock_load_benchmarks, mock_dpd
    ):
        """Test daily_results with market data."""
        mock_input.return_value = "10000"  # Starting equity
        mock_check_weekend.return_value = "2024-01-15"
        mock_ltd.return_value = pd.Timestamp("2024-01-15")
        mock_load_benchmarks.return_value = ["SPY", "IWO"]

        # Mock price data with 2 days for percentage calculation
        mock_data = pd.DataFrame(
            {
                "Open": [100, 102],
                "High": [101, 103],
                "Low": [99, 101],
                "Close": [100.5, 102.5],
                "Adj Close": [100.5, 102.5],
                "Volume": [1000, 1100],
            },
            index=pd.DatetimeIndex(["2024-01-14", "2024-01-15"], name="Date"),
        )
        mock_dpd.return_value = ts.FetchResult(mock_data, "yahoo")

        # Create portfolio history
        portfolio_data = [
            {
                "Date": "2024-01-13",
                "Ticker": "AAPL",
                "Shares": 100,
                "Buy Price": 95,
                "Cost Basis": 9500,
                "Stop Loss": 90,
                "Current Price": 99,
                "Total Value": 9900,
                "PnL": 400,
                "Action": "HOLD",
                "Cash Balance": "",
                "Total Equity": "",
            },
            {
                "Date": "2024-01-13",
                "Ticker": "TOTAL",
                "Shares": "",
                "Buy Price": "",
                "Cost Basis": "",
                "Stop Loss": "",
                "Current Price": "",
                "Total Value": 9900,
                "PnL": 400,
                "Action": "",
                "Cash Balance": 1000,
                "Total Equity": 10900,
            },
            {
                "Date": "2024-01-14",
                "Ticker": "AAPL",
                "Shares": 100,
                "Buy Price": 95,
                "Cost Basis": 9500,
                "Stop Loss": 90,
                "Current Price": 100,
                "Total Value": 10000,
                "PnL": 500,
                "Action": "HOLD",
                "Cash Balance": "",
                "Total Equity": "",
            },
            {
                "Date": "2024-01-14",
                "Ticker": "TOTAL",
                "Shares": "",
                "Buy Price": "",
                "Cost Basis": "",
                "Stop Loss": "",
                "Current Price": "",
                "Total Value": 10000,
                "PnL": 500,
                "Action": "",
                "Cash Balance": 1000,
                "Total Equity": 11000,
            },
            {
                "Date": "2024-01-15",
                "Ticker": "TOTAL",
                "Shares": "",
                "Buy Price": "",
                "Cost Basis": "",
                "Stop Loss": "",
                "Current Price": "",
                "Total Value": 10250,
                "PnL": 750,
                "Action": "",
                "Cash Balance": 1000,
                "Total Equity": 11250,
            },
        ]

        portfolio_df = pd.DataFrame(portfolio_data)
        portfolio_df.to_csv(ts.PORTFOLIO_CSV, index=False)

        chatgpt_portfolio = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "shares": [100],
                "stop_loss": [90],
                "buy_price": [95],
                "cost_basis": [9500],
            }
        )

        # Capture output
        with patch("builtins.print") as mock_print:
            ts.daily_results(chatgpt_portfolio, 1000)

            # Check that various sections were printed
            call_args = [call[0][0] for call in mock_print.call_args_list if call[0]]
            output_text = "\n".join(str(arg) for arg in call_args)

            self.assertIn("Daily Results", output_text)
            self.assertIn("Price & Volume", output_text)
            self.assertIn("Risk & Return", output_text)

    @patch("trading_script.check_weekend")
    def test_daily_results_no_portfolio_history(self, mock_check_weekend):
        """Test daily_results when no portfolio history exists."""
        mock_check_weekend.return_value = "2024-01-15"

        # Empty portfolio CSV
        pd.DataFrame().to_csv(ts.PORTFOLIO_CSV, index=False)

        chatgpt_portfolio = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "shares": [100],
                "stop_loss": [90],
                "buy_price": [95],
                "cost_basis": [9500],
            }
        )

        with patch("builtins.print") as mock_print:
            with patch("trading_script.download_price_data") as mock_dpd:
                mock_data = pd.DataFrame(
                    {"Close": [100], "Volume": [1000]},
                    index=pd.DatetimeIndex(["2024-01-15"]),
                )
                mock_dpd.return_value = ts.FetchResult(mock_data, "yahoo")

                ts.daily_results(chatgpt_portfolio, 1000)

                # Should still print basic info
                call_args = [
                    call[0][0] for call in mock_print.call_args_list if call[0]
                ]
                output_text = "\n".join(str(arg) for arg in call_args)
                self.assertIn("Daily Results", output_text)


class TestLoadLatestPortfolioState(unittest.TestCase):
    """Test load_latest_portfolio_state function."""

    def setUp(self):
        """Set up temporary directory for each test."""
        self.tmpdir = tempfile.mkdtemp()
        ts.set_data_dir(Path(self.tmpdir))

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil

        shutil.rmtree(self.tmpdir)

    @patch("builtins.input")
    def test_load_latest_portfolio_state_empty_csv(self, mock_input):
        """Test loading from empty CSV."""
        mock_input.return_value = "5000"

        # Create empty CSV
        pd.DataFrame().to_csv(ts.PORTFOLIO_CSV, index=False)

        portfolio, cash = ts.load_latest_portfolio_state(str(ts.PORTFOLIO_CSV))

        self.assertTrue(isinstance(portfolio, pd.DataFrame))
        self.assertTrue(portfolio.empty)
        self.assertEqual(cash, 5000)

    def test_load_latest_portfolio_state_with_data(self):
        """Test loading from CSV with data."""
        # Create sample portfolio data
        portfolio_data = [
            {
                "Date": "2024-01-14",
                "Ticker": "AAPL",
                "Shares": 100,
                "Buy Price": 95,
                "Cost Basis": 9500,
                "Stop Loss": 90,
                "Current Price": 100,
                "Total Value": 10000,
                "PnL": 500,
                "Action": "HOLD",
                "Cash Balance": "",
                "Total Equity": "",
            },
            {
                "Date": "2024-01-14",
                "Ticker": "MSFT",
                "Shares": 50,
                "Buy Price": 200,
                "Cost Basis": 10000,
                "Stop Loss": 180,
                "Current Price": 210,
                "Total Value": 10500,
                "PnL": 500,
                "Action": "HOLD",
                "Cash Balance": "",
                "Total Equity": "",
            },
            {
                "Date": "2024-01-14",
                "Ticker": "TOTAL",
                "Shares": "",
                "Buy Price": "",
                "Cost Basis": "",
                "Stop Loss": "",
                "Current Price": "",
                "Total Value": 20500,
                "PnL": 1000,
                "Action": "",
                "Cash Balance": 2000,
                "Total Equity": 22500,
            },
            {
                "Date": "2024-01-15",
                "Ticker": "AAPL",
                "Shares": 100,
                "Buy Price": 95,
                "Cost Basis": 9500,
                "Stop Loss": 90,
                "Current Price": 102,
                "Total Value": 10200,
                "PnL": 700,
                "Action": "HOLD",
                "Cash Balance": "",
                "Total Equity": "",
            },
            {
                "Date": "2024-01-15",
                "Ticker": "MSFT",
                "Shares": 50,
                "Buy Price": 200,
                "Cost Basis": 10000,
                "Stop Loss": 180,
                "Current Price": 205,
                "Total Value": 10250,
                "PnL": 250,
                "Action": "HOLD",
                "Cash Balance": "",
                "Total Equity": "",
            },
            {
                "Date": "2024-01-15",
                "Ticker": "TSLA",
                "Shares": 25,
                "Buy Price": 800,
                "Cost Basis": 20000,
                "Stop Loss": 750,
                "Current Price": 820,
                "Total Value": 20500,
                "PnL": 500,
                "Action": "SELL - Stop Loss Triggered",
                "Cash Balance": "",
                "Total Equity": "",
            },
            {
                "Date": "2024-01-15",
                "Ticker": "TOTAL",
                "Shares": "",
                "Buy Price": "",
                "Cost Basis": "",
                "Stop Loss": "",
                "Current Price": "",
                "Total Value": 20500,
                "PnL": 1000,
                "Action": "",
                "Cash Balance": 3000,
                "Total Equity": 23500,
            },
        ]

        portfolio_df = pd.DataFrame(portfolio_data)
        portfolio_df.to_csv(ts.PORTFOLIO_CSV, index=False)

        portfolio, cash = ts.load_latest_portfolio_state(str(ts.PORTFOLIO_CSV))

        # Should load latest non-TOTAL, non-SELL entries
        self.assertEqual(len(portfolio), 2)  # AAPL and MSFT (TSLA was sold)
        self.assertEqual(cash, 3000)  # Latest TOTAL cash balance

        # Check portfolio structure
        tickers = [stock["ticker"] for stock in portfolio]
        self.assertIn("AAPL", tickers)
        self.assertIn("MSFT", tickers)
        self.assertNotIn("TSLA", tickers)  # Should be excluded due to SELL action

    @patch("builtins.input")
    def test_load_latest_portfolio_state_invalid_cash_input(self, mock_input):
        """Test loading with invalid cash input."""
        mock_input.return_value = "invalid"

        # Create empty CSV
        pd.DataFrame().to_csv(ts.PORTFOLIO_CSV, index=False)

        with self.assertRaises(ValueError):
            ts.load_latest_portfolio_state(str(ts.PORTFOLIO_CSV))


class TestMainFunction(unittest.TestCase):
    """Test main function and command-line interface."""

    def setUp(self):
        """Set up temporary directory for each test."""
        self.tmpdir = tempfile.mkdtemp()
        ts.set_data_dir(Path(self.tmpdir))

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil

        shutil.rmtree(self.tmpdir)

    @patch("trading_script.process_portfolio")
    @patch("trading_script.daily_results")
    @patch("trading_script.load_latest_portfolio_state")
    def test_main_function(self, mock_load_state, mock_daily_results, mock_process):
        """Test main function execution."""
        # Mock return values
        mock_portfolio = pd.DataFrame({"ticker": ["AAPL"], "shares": [100]})
        mock_load_state.return_value = (mock_portfolio, 5000)
        mock_process.return_value = (mock_portfolio, 4500)

        # Create a dummy CSV file
        test_file = str(ts.PORTFOLIO_CSV)
        pd.DataFrame().to_csv(test_file, index=False)

        ts.main(test_file, Path(self.tmpdir))

        mock_load_state.assert_called_once_with(test_file)
        mock_process.assert_called_once()
        mock_daily_results.assert_called_once()


class TestEnvironmentVariableHandling(unittest.TestCase):
    """Test environment variable handling."""

    def test_env_var_asof_date(self):
        """Test ASOF_DATE environment variable handling."""
        original_asof = ts.ASOF_DATE

        try:
            with patch.dict(os.environ, {"ASOF_DATE": "2024-01-15"}):
                # Re-import to trigger env var processing
                import importlib

                importlib.reload(ts)

                self.assertEqual(ts.ASOF_DATE, pd.Timestamp("2024-01-15").normalize())
        finally:
            ts.ASOF_DATE = original_asof


class TestEdgeCasesAndErrorHandling(unittest.TestCase):
    """Test edge cases and error handling scenarios."""

    def test_download_price_data_with_nan_values(self):
        """Test handling of NaN values in price data."""
        with patch("trading_script._yahoo_download") as mock_yahoo:
            mock_df = pd.DataFrame(
                {
                    "Open": [np.nan],
                    "High": [102],
                    "Low": [99],
                    "Close": [101],
                    "Adj Close": [101],
                    "Volume": [1000],
                }
            )
            mock_yahoo.return_value = mock_df

            result = ts.download_price_data("AAPL", period="1d")
            self.assertFalse(result.df.empty)
            # Should handle NaN gracefully

    def test_performance_metrics_with_insufficient_data(self):
        """Test performance metrics calculation with insufficient data."""
        # Create minimal portfolio history
        portfolio_data = [
            {
                "Date": "2024-01-15",
                "Ticker": "TOTAL",
                "Shares": "",
                "Buy Price": "",
                "Cost Basis": "",
                "Stop Loss": "",
                "Current Price": "",
                "Total Value": 10000,
                "PnL": 0,
                "Action": "",
                "Cash Balance": 1000,
                "Total Equity": 11000,
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            ts.set_data_dir(Path(tmpdir))
            portfolio_df = pd.DataFrame(portfolio_data)
            portfolio_df.to_csv(ts.PORTFOLIO_CSV, index=False)

            chatgpt_portfolio = pd.DataFrame()

            with patch("trading_script.download_price_data") as mock_dpd:
                mock_data = pd.DataFrame(
                    {"Close": [100], "Volume": [1000]},
                    index=pd.DatetimeIndex(["2024-01-15"]),
                )
                mock_dpd.return_value = ts.FetchResult(mock_data, "yahoo")

                with patch("builtins.print"):
                    with patch("builtins.input", return_value="10000"):
                        # Should not crash with insufficient data
                        ts.daily_results(chatgpt_portfolio, 1000)

    def test_invalid_ticker_symbols(self):
        """Test handling of invalid ticker symbols."""
        with patch("trading_script._yahoo_download") as mock_yahoo:
            with patch("trading_script._stooq_download") as mock_stooq:
                with patch("trading_script._stooq_csv_download") as mock_csv:
                    # All sources return empty
                    mock_yahoo.return_value = pd.DataFrame()
                    mock_stooq.return_value = pd.DataFrame()
                    mock_csv.return_value = pd.DataFrame()

                    result = ts.download_price_data("INVALID123", period="1d")
                    self.assertEqual(result.source, "empty")
                    self.assertTrue(result.df.empty)

    def test_network_timeout_handling(self):
        """Test handling of network timeouts."""
        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("Timeout")

            start = pd.Timestamp("2024-01-01")
            end = pd.Timestamp("2024-01-02")
            result = ts._stooq_csv_download("AAPL", start, end)

            self.assertTrue(result.empty)

    def test_malformed_csv_data(self):
        """Test handling of malformed CSV data from Stooq."""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "invalid,csv,data\n"
            mock_get.return_value = mock_response

            start = pd.Timestamp("2024-01-01")
            end = pd.Timestamp("2024-01-02")
            result = ts._stooq_csv_download("AAPL", start, end)

            # Should handle malformed data gracefully
            self.assertTrue(result.empty)

    def test_extreme_date_ranges(self):
        """Test handling of extreme date ranges."""
        # Very old date
        start = pd.Timestamp("1900-01-01")
        end = pd.Timestamp("1900-01-02")

        result_start, result_end = ts._weekend_safe_range(None, start, end)
        self.assertEqual(result_start, start.normalize())
        self.assertEqual(result_end, end.normalize())

    def test_portfolio_with_zero_shares(self):
        """Test portfolio operations with zero shares."""
        portfolio = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "shares": [0],  # Zero shares
                "stop_loss": [90],
                "buy_price": [95],
                "cost_basis": [0],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ts.set_data_dir(Path(tmpdir))

            with patch("trading_script.download_price_data") as mock_dpd:
                mock_data = pd.DataFrame(
                    {
                        "Open": [100],
                        "High": [102],
                        "Low": [99],
                        "Close": [101],
                        "Adj Close": [101],
                        "Volume": [1000],
                    },
                    index=pd.DatetimeIndex(["2024-01-15"]),
                )
                mock_dpd.return_value = ts.FetchResult(mock_data, "yahoo")

                # Should handle zero shares gracefully
                result_portfolio, result_cash = ts.process_portfolio(
                    portfolio, cash=1000, interactive=False
                )


if __name__ == "__main__":
    # Configure logging to reduce noise during testing
    import logging

    logging.getLogger("trading_script").setLevel(logging.CRITICAL)
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    # Suppress warnings during testing
    warnings.filterwarnings("ignore")

    # Run tests
    unittest.main(verbosity=2)
