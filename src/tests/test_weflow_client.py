from src.perception.weflow_client import WeFlowClient


def test_weflow_token_prefers_environment(monkeypatch):
    monkeypatch.setenv("WEFLOW_ACCESS_TOKEN", "random-local-token")

    client = WeFlowClient()

    assert client.access_token == "random-local-token"


def test_explicit_weflow_token_overrides_environment(monkeypatch):
    monkeypatch.setenv("WEFLOW_ACCESS_TOKEN", "environment-token")

    client = WeFlowClient(access_token="explicit-token")

    assert client.access_token == "explicit-token"
