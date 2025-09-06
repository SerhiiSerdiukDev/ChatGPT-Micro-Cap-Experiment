@echo off
REM Script to run comprehensive tests for trading_script.py
REM Usage: run_tests.bat

echo Setting up test environment...

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install test dependencies
echo Installing test dependencies...
pip install -r test_requirements.txt

REM Run tests with coverage
echo Running comprehensive tests...

REM Run different test categories
echo Running unit tests...
python -m pytest test_trading_script.py::TestDateUtilities -v
python -m pytest test_trading_script.py::TestConfigurationHelpers -v
python -m pytest test_trading_script.py::TestDataAccess -v
python -m pytest test_trading_script.py::TestFileOperations -v
python -m pytest test_trading_script.py::TestPortfolioOperations -v
python -m pytest test_trading_script.py::TestTradeLogging -v

echo Running integration tests...
python -m pytest test_trading_script.py::TestReportingMetrics -v
python -m pytest test_trading_script.py::TestLoadLatestPortfolioState -v
python -m pytest test_trading_script.py::TestMainFunction -v

echo Running edge case tests...
python -m pytest test_trading_script.py::TestEnvironmentVariableHandling -v
python -m pytest test_trading_script.py::TestEdgeCasesAndErrorHandling -v

REM Run all tests with coverage
echo Running full test suite with coverage...
python -m pytest test_trading_script.py --cov=trading_script --cov-report=html --cov-report=term-missing --cov-fail-under=85

echo Test execution completed!
echo Coverage report available in: htmlcov\index.html

if %errorlevel% equ 0 (
    echo All tests passed successfully!
) else (
    echo Some tests failed. Please check the output above.
    exit /b 1
)
