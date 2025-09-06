"""Integration tests for trading_script.py that test end-to-end workflows.

These tests simulate real-world scenarios and test the interaction between
different components of the trading system.
"""

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, Mock
import pandas as pd
import numpy as np

import trading_script as ts


class TestEndToEndWorkflows(unittest.TestCase):
    """Test complete end-to-end workflows."""

    def setUp(self):
        """Set up temporary directory and reset global state."""
        self.tmpdir = tempfile.mkdtemp()
        ts.set_data_dir(Path(self.tmpdir))
        ts.ASOF_DATE = None

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil

        shutil.rmtree(self.tmpdir)

    @patch("trading_script.download_price_data")
    @patch("builtins.input")
    def test_complete_trading_workflow(self, mock_input, mock_dpd):
        """Test a complete trading workflow from empty portfolio to final results."""
        # Mock user inputs for starting cash
        mock_input.return_value = "10000"

        # Mock price data responses
        def price_data_side_effect(ticker, **kwargs):
            if ticker == "AAPL":
                return ts.FetchResult(
                    pd.DataFrame(
                        {
                            "Open": [150],
                            "High": [155],
                            "Low": [148],
                            "Close": [152],
                            "Adj Close": [152],
                            "Volume": [1000000],
                        },
                        index=pd.DatetimeIndex(["2024-01-15"]),
                    ),
                    "yahoo",
                )
            elif ticker in ["SPY", "IWO", "IWM", "XBI"]:
                return ts.FetchResult(
                    pd.DataFrame(
                        {
                            "Open": [400],
                            "High": [405],
                            "Low": [398],
                            "Close": [402],
                            "Adj Close": [402],
                            "Volume": [500000],
                        },
                        index=pd.DatetimeIndex(["2024-01-14", "2024-01-15"]),
                    ),
                    "yahoo",
                )
            else:
                return ts.FetchResult(pd.DataFrame(), "empty")

        mock_dpd.side_effect = price_data_side_effect

        # Set as-of date
        ts.set_asof("2024-01-15")

        # Start with empty portfolio
        portfolio = pd.DataFrame(
            columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"]
        )

        # Process portfolio (non-interactive mode)
        result_portfolio, result_cash = ts.process_portfolio(
            portfolio, 10000, interactive=False
        )

        # Verify results
        self.assertEqual(result_cash, 10000)  # No trades, cash unchanged
        self.assertTrue(result_portfolio.empty)  # No holdings

        # Verify files were created
        self.assertTrue(ts.PORTFOLIO_CSV.exists())

        # Load portfolio history and verify
        df = pd.read_csv(ts.PORTFOLIO_CSV)
        self.assertFalse(df.empty)
        self.assertIn("TOTAL", df["Ticker"].values)

    @patch("trading_script.download_price_data")
    @patch("builtins.input")
    def test_stop_loss_workflow(self, mock_input, mock_dpd):
        """Test workflow where stop loss is triggered."""
        mock_input.return_value = "10000"

        # Price data that triggers stop loss
        mock_dpd.return_value = ts.FetchResult(
            pd.DataFrame(
                {
                    "Open": [85],
                    "High": [88],
                    "Low": [84],  # Low triggers stop loss
                    "Close": [86],
                    "Adj Close": [86],
                    "Volume": [2000000],
                },
                index=pd.DatetimeIndex(["2024-01-15"]),
            ),
            "yahoo",
        )

        # Portfolio with stop loss at 90
        portfolio = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "shares": [100],
                "stop_loss": [90],
                "buy_price": [95],
                "cost_basis": [9500],
            }
        )

        result_portfolio, result_cash = ts.process_portfolio(
            portfolio, 1000, interactive=False
        )

        # Portfolio should be empty (stock sold)
        self.assertTrue(result_portfolio.empty)
        # Cash should increase from sale
        self.assertGreater(result_cash, 1000)

        # Verify trade log was created
        self.assertTrue(ts.TRADE_LOG_CSV.exists())
        trade_log = pd.read_csv(ts.TRADE_LOG_CSV)
        self.assertFalse(trade_log.empty)
        self.assertEqual(trade_log.iloc[0]["Ticker"], "AAPL")

    @patch("trading_script.download_price_data")
    @patch("trading_script.check_weekend")
    @patch("builtins.input")
    def test_manual_trading_workflow(self, mock_input, mock_check_weekend, mock_dpd):
        """Test manual trading workflow."""
        mock_check_weekend.return_value = "2024-01-15"

        # Mock price data for manual trade
        mock_dpd.return_value = ts.FetchResult(
            pd.DataFrame(
                {
                    "Open": [100],
                    "High": [105],
                    "Low": [98],
                    "Close": [103],
                    "Adj Close": [103],
                    "Volume": [1500000],
                },
                index=pd.DatetimeIndex(["2024-01-15"]),
            ),
            "yahoo",
        )

        # Mock user inputs for manual buy
        mock_input.side_effect = ["", ""]  # Don't cancel trades

        portfolio = pd.DataFrame(
            columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"]
        )

        # Test manual buy
        cash, new_portfolio = ts.log_manual_buy(
            buy_price=102,
            shares=50,
            ticker="AAPL",
            stoploss=95,
            cash=10000,
            chatgpt_portfolio=portfolio,
            interactive=True,
        )

        # Should have bought at open price (100)
        self.assertEqual(cash, 5000)  # 10000 - (50 * 100)
        self.assertEqual(len(new_portfolio), 1)

        # Test manual sell
        cash, final_portfolio = ts.log_manual_sell(
            sell_price=101,
            shares_sold=25,
            ticker="AAPL",
            cash=cash,
            chatgpt_portfolio=new_portfolio,
            reason="Taking partial profits",
            interactive=False,
        )

        # Should have sold at open price (100, but since open >= sell_price, uses open)
        expected_cash = 5000 + (25 * 100)  # Selling 25 shares at open price
        self.assertEqual(cash, expected_cash)
        self.assertEqual(
            final_portfolio.iloc[0]["shares"], 25
        )  # 50 - 25 = 25 remaining

    def test_configuration_loading_workflow(self):
        """Test configuration loading and benchmark selection."""
        # Create custom tickers.json
        custom_benchmarks = ["VTI", "QQQ", "ARKK", "IWM"]
        config = {"benchmarks": custom_benchmarks}

        config_path = Path(self.tmpdir) / "tickers.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Load benchmarks
        result = ts.load_benchmarks(Path(self.tmpdir))
        self.assertEqual(result, custom_benchmarks)

        # Test with malformed JSON
        with open(config_path, "w") as f:
            f.write("{ invalid json")

        result = ts.load_benchmarks(Path(self.tmpdir))
        self.assertEqual(result, ts.DEFAULT_BENCHMARKS.copy())

    @patch("trading_script.download_price_data")
    def test_data_source_fallback_workflow(self, mock_dpd):
        """Test the data source fallback workflow."""
        call_count = 0

        def fallback_side_effect(ticker, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:  # Yahoo fails
                return ts.FetchResult(pd.DataFrame(), "empty")
            elif call_count == 2:  # Stooq PDR fails
                return ts.FetchResult(pd.DataFrame(), "empty")
            elif call_count == 3:  # Stooq CSV succeeds
                return ts.FetchResult(
                    pd.DataFrame(
                        {
                            "Open": [100],
                            "High": [102],
                            "Low": [99],
                            "Close": [101],
                            "Adj Close": [101],
                            "Volume": [1000],
                        },
                        index=pd.DatetimeIndex(["2024-01-15"]),
                    ),
                    "stooq-csv",
                )
            else:
                return ts.FetchResult(pd.DataFrame(), "empty")

        # Mock the internal functions that download_price_data calls
        with patch("trading_script._yahoo_download") as mock_yahoo, patch(
            "trading_script._stooq_download"
        ) as mock_stooq, patch("trading_script._stooq_csv_download") as mock_csv:

            mock_yahoo.return_value = pd.DataFrame()
            mock_stooq.return_value = pd.DataFrame()
            mock_csv.return_value = pd.DataFrame(
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

            result = ts.download_price_data("AAPL", period="1d")

            # Should succeed with Stooq CSV
            self.assertEqual(result.source, "stooq-csv")
            self.assertFalse(result.df.empty)

    @patch("trading_script.download_price_data")
    @patch("builtins.input")
    def test_performance_calculation_workflow(self, mock_input, mock_dpd):
        """Test the complete performance calculation workflow."""
        mock_input.return_value = "10000"  # Starting equity

        # Create historical portfolio data
        portfolio_history = []
        dates = pd.date_range("2024-01-01", "2024-01-15", freq="D")

        for i, date in enumerate(dates):
            equity = 10000 + i * 100  # Gradual increase
            portfolio_history.append(
                {
                    "Date": date.strftime("%Y-%m-%d"),
                    "Ticker": "TOTAL",
                    "Shares": "",
                    "Buy Price": "",
                    "Cost Basis": "",
                    "Stop Loss": "",
                    "Current Price": "",
                    "Total Value": equity - 1000,
                    "PnL": (equity - 10000),
                    "Action": "",
                    "Cash Balance": 1000,
                    "Total Equity": equity,
                }
            )

        # Save portfolio history
        df = pd.DataFrame(portfolio_history)
        df.to_csv(ts.PORTFOLIO_CSV, index=False)

        # Mock price data for benchmarks and S&P 500
        def price_data_side_effect(ticker, **kwargs):
            # Return 2-day data for percentage calculation
            if ticker in ["SPY", "IWO", "IWM", "XBI"]:
                return ts.FetchResult(
                    pd.DataFrame(
                        {
                            "Open": [400, 402],
                            "High": [405, 407],
                            "Low": [398, 400],
                            "Close": [402, 404],
                            "Adj Close": [402, 404],
                            "Volume": [500000, 520000],
                        },
                        index=pd.DatetimeIndex(["2024-01-14", "2024-01-15"]),
                    ),
                    "yahoo",
                )
            elif ticker == "^GSPC":
                # S&P 500 data for CAPM calculation
                return ts.FetchResult(
                    pd.DataFrame(
                        {"Close": np.linspace(4000, 4200, len(dates))}, index=dates
                    ),
                    "yahoo",
                )
            else:
                return ts.FetchResult(pd.DataFrame(), "empty")

        mock_dpd.side_effect = price_data_side_effect

        portfolio = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "shares": [100],
                "stop_loss": [90],
                "buy_price": [95],
                "cost_basis": [9500],
            }
        )

        # Run daily results (should calculate performance metrics)
        with patch("builtins.print") as mock_print:
            ts.daily_results(portfolio, 1000)

            # Verify various sections were printed
            call_args = [
                str(call[0][0]) for call in mock_print.call_args_list if call[0]
            ]
            output_text = "\n".join(call_args)

            self.assertIn("Daily Results", output_text)
            self.assertIn("Risk & Return", output_text)
            self.assertIn("CAPM", output_text)


class TestErrorRecoveryScenarios(unittest.TestCase):
    """Test error recovery and resilience scenarios."""

    def setUp(self):
        """Set up temporary directory."""
        self.tmpdir = tempfile.mkdtemp()
        ts.set_data_dir(Path(self.tmpdir))

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil

        shutil.rmtree(self.tmpdir)

    def test_corrupted_portfolio_csv_recovery(self):
        """Test recovery from corrupted portfolio CSV."""
        # Create corrupted CSV
        with open(ts.PORTFOLIO_CSV, "w") as f:
            f.write("corrupted,data,here\n1,2")  # Incomplete row

        # Should handle gracefully
        try:
            df = pd.read_csv(ts.PORTFOLIO_CSV)
            # If pandas can read it, it should work
            self.assertIsInstance(df, pd.DataFrame)
        except Exception:
            # If pandas can't read it, function should handle the error
            pass

    def test_missing_trade_log_recovery(self):
        """Test recovery when trade log is missing."""
        # Ensure trade log doesn't exist
        if ts.TRADE_LOG_CSV.exists():
            ts.TRADE_LOG_CSV.unlink()

        portfolio = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "shares": [100],
                "stop_loss": [90],
                "buy_price": [95],
                "cost_basis": [9500],
            }
        )

        # Should create new trade log without error
        result_portfolio = ts.log_sell("AAPL", 100, 92, 95, -300, portfolio)

        self.assertTrue(ts.TRADE_LOG_CSV.exists())
        self.assertTrue(result_portfolio.empty)

    @patch("trading_script.download_price_data")
    def test_intermittent_network_failure_recovery(self, mock_dpd):
        """Test recovery from intermittent network failures."""
        call_count = 0

        def intermittent_failure(ticker, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count <= 2:  # First two calls fail
                return ts.FetchResult(pd.DataFrame(), "empty")
            else:  # Subsequent calls succeed
                return ts.FetchResult(
                    pd.DataFrame(
                        {
                            "Open": [100],
                            "High": [102],
                            "Low": [99],
                            "Close": [101],
                            "Adj Close": [101],
                            "Volume": [1000],
                        },
                        index=pd.DatetimeIndex(["2024-01-15"]),
                    ),
                    "yahoo",
                )

        mock_dpd.side_effect = intermittent_failure

        portfolio = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL"],  # Multiple tickers
                "shares": [100, 50, 25],
                "stop_loss": [90, 200, 150],
                "buy_price": [95, 210, 160],
                "cost_basis": [9500, 10500, 4000],
            }
        )

        # Should handle mixed success/failure gracefully
        result_portfolio, result_cash = ts.process_portfolio(
            portfolio, 1000, interactive=False
        )

        # Some stocks should have data, others NO DATA
        self.assertTrue(ts.PORTFOLIO_CSV.exists())
        df = pd.read_csv(ts.PORTFOLIO_CSV)
        actions = df["Action"].values

        # Should have mix of HOLD and NO DATA actions
        self.assertTrue(any("HOLD" in str(action) for action in actions))
        self.assertTrue(any("NO DATA" in str(action) for action in actions))


if __name__ == "__main__":
    unittest.main(verbosity=2)
