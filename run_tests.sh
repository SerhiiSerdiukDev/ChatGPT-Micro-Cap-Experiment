#!/bin/bash

# Script to run comprehensive tests for trading_script.py
# Usage: ./run_tests.sh [options]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up test environment...${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python -m venv venv
fi

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install test dependencies
echo -e "${YELLOW}Installing test dependencies...${NC}"
pip install -r test_requirements.txt

# Run tests with coverage
echo -e "${YELLOW}Running comprehensive tests...${NC}"

# Run different test categories
echo -e "${GREEN}Running unit tests...${NC}"
python -m pytest test_trading_script.py::TestDateUtilities -v
python -m pytest test_trading_script.py::TestConfigurationHelpers -v
python -m pytest test_trading_script.py::TestDataAccess -v
python -m pytest test_trading_script.py::TestFileOperations -v
python -m pytest test_trading_script.py::TestPortfolioOperations -v
python -m pytest test_trading_script.py::TestTradeLogging -v

echo -e "${GREEN}Running integration tests...${NC}"
python -m pytest test_trading_script.py::TestReportingMetrics -v
python -m pytest test_trading_script.py::TestLoadLatestPortfolioState -v
python -m pytest test_trading_script.py::TestMainFunction -v

echo -e "${GREEN}Running edge case tests...${NC}"
python -m pytest test_trading_script.py::TestEnvironmentVariableHandling -v
python -m pytest test_trading_script.py::TestEdgeCasesAndErrorHandling -v

# Run all tests with coverage
echo -e "${GREEN}Running full test suite with coverage...${NC}"
python -m pytest test_trading_script.py --cov=trading_script --cov-report=html --cov-report=term-missing --cov-fail-under=85

# Generate coverage badge (optional)
if command -v coverage-badge &> /dev/null; then
    echo -e "${YELLOW}Generating coverage badge...${NC}"
    coverage-badge -o coverage.svg
fi

echo -e "${GREEN}Test execution completed!${NC}"
echo -e "${YELLOW}Coverage report available in: htmlcov/index.html${NC}"

# Check if tests passed
if [ $? -eq 0 ]; then
    echo -e "${GREEN}All tests passed successfully!${NC}"
else
    echo -e "${RED}Some tests failed. Please check the output above.${NC}"
    exit 1
fi
