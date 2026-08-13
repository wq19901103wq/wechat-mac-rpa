def pytest_addoption(parser):
    parser.addoption("--run-api", action="store_true", default=False, help="Run with real API calls")
    parser.addoption("--n-runs", type=int, default=3, help="Number of runs per case")
