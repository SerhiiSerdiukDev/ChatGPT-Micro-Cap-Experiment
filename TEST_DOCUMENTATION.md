# Trading Script Test Suite Documentation

## Overview

This comprehensive test suite provides full coverage for `trading_script.py`, including unit tests, integration tests, edge cases, and error handling scenarios. The test suite is designed to ensure reliability, correctness, and robustness of the trading portfolio management system.

## Test Structure

### Core Test Files

1. **`test_trading_script.py`** - Main test suite with unit tests
2. **`test_integration.py`** - Integration and end-to-end workflow tests
3. **`test_requirements.txt`** - Test dependencies
4. **`pytest.ini`** - Test configuration
5. **`run_tests.sh`** / **`run_tests.bat`** - Test execution scripts

### Test Categories

#### 1. Unit Tests (`test_trading_script.py`)

- **TestDateUtilities**: Date handling, weekend logic, trading windows
- **TestConfigurationHelpers**: JSON configuration loading, benchmark management
- **TestDataAccess**: Market data fetching, fallback mechanisms, data normalization
- **TestFileOperations**: File path configuration, directory management
- **TestPortfolioOperations**: Portfolio processing, trade execution
- **TestTradeLogging**: Buy/sell logging, trade history management
- **TestReportingMetrics**: Performance calculations, risk metrics
- **TestLoadLatestPortfolioState**: Portfolio state loading and validation
- **TestMainFunction**: Main function execution and CLI interface
- **TestEnvironmentVariableHandling**: Environment variable processing
- **TestEdgeCasesAndErrorHandling**: Edge cases, error scenarios, resilience

#### 2. Integration Tests (`test_integration.py`)

- **TestEndToEndWorkflows**: Complete trading workflows
- **TestErrorRecoveryScenarios**: Error recovery and system resilience

## Test Coverage Areas

### 1. Market Data Fetching

- Yahoo Finance API integration
- Stooq fallback mechanisms
- Data source prioritization
- Network failure handling
- Malformed data handling
- Empty response handling
- Proxy ticker fallbacks (e.g., ^GSPC -> SPY)

### 2. Portfolio Management

- Portfolio processing (interactive/non-interactive)
- Stop-loss execution
- Manual buy/sell operations
- Position sizing and cash management
- Portfolio state persistence
- Multi-ticker handling

### 3. Trade Execution

- Market-on-open (MOO) orders
- Limit orders
- Stop-loss triggers
- Trade logging and history
- Price validation
- Insufficient cash handling

### 4. Date and Time Handling

- Weekend detection and adjustment
- Trading day calculations
- Date window computations
- As-of date functionality
- Time zone considerations

### 5. File Operations

- CSV reading/writing
- JSON configuration loading
- Directory management
- File corruption recovery
- Permission handling

### 6. Performance Metrics

- Return calculations
- Risk metrics (Sharpe, Sortino ratios)
- Drawdown analysis
- CAPM calculations (Alpha, Beta)
- Benchmark comparisons

### 7. Configuration Management

- Benchmark ticker loading
- Configuration file validation
- Default value handling
- Environment variable processing

### 8. Error Handling

- Network timeouts
- Invalid ticker symbols
- Corrupted data files
- Missing dependencies
- API rate limiting
- Insufficient permissions

## Running the Tests

### Prerequisites

1. Python 3.8 or higher
2. Required dependencies (see `test_requirements.txt`)

### Installation

```bash
# Install test dependencies
pip install -r test_requirements.txt
```

### Execution Options

#### Option 1: Using Test Scripts (Recommended)

**Windows:**

```cmd
run_tests.bat
```

**Unix/Linux/macOS:**

```bash
chmod +x run_tests.sh
./run_tests.sh
```

#### Option 2: Direct pytest Commands

**Run all tests:**

```bash
pytest test_trading_script.py test_integration.py -v
```

**Run with coverage:**

```bash
pytest test_trading_script.py test_integration.py --cov=trading_script --cov-report=html --cov-report=term-missing
```

**Run specific test categories:**

```bash
# Unit tests only
pytest test_trading_script.py -v

# Integration tests only
pytest test_integration.py -v

# Specific test class
pytest test_trading_script.py::TestDateUtilities -v

# Specific test method
pytest test_trading_script.py::TestDateUtilities::test_last_trading_date_weekend -v
```

#### Option 3: Standard unittest

```bash
python -m unittest test_trading_script.py -v
python -m unittest test_integration.py -v
```

## Test Configuration

### Coverage Requirements

- Minimum coverage threshold: 85%
- Target coverage: 90%+
- HTML coverage reports generated in `htmlcov/`

### Test Markers

- `slow`: Long-running tests
- `integration`: Integration tests
- `unit`: Unit tests

### Skipping Slow Tests

```bash
pytest -m "not slow" test_trading_script.py
```

## Mocking Strategy

The test suite uses extensive mocking to isolate components and simulate various scenarios:

### External Dependencies Mocked

- `yfinance.download()` - Market data API
- `requests.get()` - HTTP requests
- `pandas_datareader` - Alternative data source
- `builtins.input()` - User input
- `datetime.now()` - Time-dependent functions
- File system operations

### Mocking Patterns

1. **Data Source Mocking**: Simulate different market data scenarios
2. **Network Failure Simulation**: Test resilience to network issues
3. **User Input Simulation**: Test interactive functionality
4. **Time Manipulation**: Test date-dependent logic
5. **File System Isolation**: Use temporary directories

## Edge Cases Covered

### 1. Market Data Edge Cases

- Empty market data responses
- NaN values in price data
- Inconsistent date indices
- Missing OHLC columns
- Zero volume trading days
- Market holidays
- After-hours data

### 2. Portfolio Edge Cases

- Empty portfolios
- Zero-share positions
- Negative prices (invalid data)
- Extremely large position sizes
- Fractional shares
- Currency precision issues

### 3. Date Edge Cases

- Weekend trading attempts
- Market holidays
- Year boundaries
- Leap year handling
- Time zone edge cases
- Historical date limits

### 4. File System Edge Cases

- Read-only file systems
- Disk space limitations
- File corruption scenarios
- Concurrent file access
- Unicode/encoding issues

### 5. Network Edge Cases

- Complete network failures
- Intermittent connectivity
- API rate limiting
- Malformed responses
- Timeout scenarios

## Error Scenarios Tested

### 1. Data Validation Errors

- Invalid ticker symbols
- Malformed CSV data
- Inconsistent data types
- Missing required fields

### 2. Business Logic Errors

- Insufficient cash for purchases
- Selling more shares than owned
- Invalid stop-loss prices
- Negative share quantities

### 3. System Errors

- File permission errors
- Disk space errors
- Memory limitations
- Dependency availability

### 4. User Input Errors

- Invalid date formats
- Non-numeric inputs
- Empty inputs
- Special characters

## Performance Testing

The test suite includes performance considerations:

### 1. Memory Usage

- Large dataset handling
- Memory leak detection
- Efficient data structures

### 2. Execution Time

- API timeout handling
- Bulk operations
- File I/O optimization

### 3. Scalability

- Multiple ticker handling
- Large portfolio processing
- Historical data processing

## Continuous Integration

The test suite is designed for CI/CD integration:

### 1. Automated Testing

- All tests run automatically
- Coverage reports generated
- Test result notifications

### 2. Environment Compatibility

- Cross-platform testing (Windows/Unix)
- Multiple Python versions
- Dependency version testing

### 3. Quality Gates

- Minimum coverage enforcement
- Test failure blocking
- Performance regression detection

## Best Practices Implemented

### 1. Test Independence

- Each test is isolated
- No shared state between tests
- Proper setup/teardown

### 2. Readable Test Names

- Descriptive test method names
- Clear test documentation
- Scenario-based naming

### 3. Maintainable Tests

- DRY principle application
- Reusable test utilities
- Modular test structure

### 4. Comprehensive Assertions

- Multiple assertion types
- Error message validation
- State verification

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **File Permission Errors**: Run with appropriate permissions
3. **Network Timeouts**: Check internet connectivity
4. **Memory Issues**: Use smaller test datasets

### Debug Mode

```bash
pytest --pdb test_trading_script.py::test_name
```

### Verbose Output

```bash
pytest -vvv --tb=long test_trading_script.py
```

## Contributing

When adding new features to `trading_script.py`:

1. Add corresponding unit tests
2. Include edge case testing
3. Update integration tests if needed
4. Maintain coverage threshold
5. Document test scenarios

## Test Metrics

Expected test metrics:

- **Total Tests**: 80+ test methods
- **Coverage**: 90%+ line coverage
- **Execution Time**: < 2 minutes
- **Success Rate**: 100% in stable environments

## Future Enhancements

Planned test improvements:

1. Property-based testing with Hypothesis
2. Performance benchmarking
3. Load testing capabilities
4. Security testing
5. Compliance testing
