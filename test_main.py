import asyncio
import pytest
from fastapi.testclient import TestClient
from main import app, calculate_variance, WINDOW_SIZE, _agent_data, _lock

client = TestClient(app)


def setup_function():
    with _lock:
        _agent_data.clear()


def test_calculate_variance_basic():
    assert calculate_variance([1, 1, 1, 1]) == 0.0
    result = calculate_variance([1, 2, 3, 4, 5])
    assert abs(result - 2.0) < 0.001


def test_calculate_variance_insufficient_data():
    assert calculate_variance([]) is None
    assert calculate_variance([5]) is None


def test_ping_endpoint_success():
    response = client.post(
        "/api/v1/telephony/ping",
        json={"agent_id": "agent_001", "volume": 50.0, "latency": 30.0}
    )
    assert response.status_code == 204
    assert response.content == b""


def test_ping_endpoint_validation():
    response = client.post(
        "/api/v1/telephony/ping",
        json={"agent_id": "agent_001", "volume": 150.0, "latency": 30.0}
    )
    assert response.status_code == 422

    response = client.post(
        "/api/v1/telephony/ping",
        json={"agent_id": "agent_001", "volume": 50.0, "latency": -5.0}
    )
    assert response.status_code == 422

    response = client.post(
        "/api/v1/telephony/ping",
        json={"agent_id": "", "volume": 50.0, "latency": 30.0}
    )
    assert response.status_code == 422


def test_sliding_window_50_points():
    for i in range(60):
        client.post(
            "/api/v1/telephony/ping",
            json={"agent_id": "agent_002", "volume": float(i), "latency": 10.0}
        )
    response = client.get("/api/v1/telephony/agent/agent_002")
    data = response.json()
    assert data["data_points"] == 50
    assert data["avg_volume"] >= 10


def test_detect_yelling_high_variance():
    volumes = [10, 90, 10, 90, 10, 90, 10, 90, 10, 90,
               10, 90, 10, 90, 10, 90, 10, 90, 10, 90,
               10, 90, 10, 90, 10, 90, 10, 90, 10, 90]
    for v in volumes:
        client.post(
            "/api/v1/telephony/ping",
            json={"agent_id": "yeller", "volume": float(v), "latency": 20.0}
        )
    response = client.get("/api/v1/telephony/agent/yeller")
    data = response.json()
    assert data["is_yelling"] is True
    assert data["volume_variance"] > 400


def test_detect_silence():
    for _ in range(15):
        client.post(
            "/api/v1/telephony/ping",
            json={"agent_id": "silent", "volume": 2.0, "latency": 25.0}
        )
    response = client.get("/api/v1/telephony/agent/silent")
    data = response.json()
    assert data["is_silent"] is True
    assert data["avg_volume"] < 5.0


def test_detect_loud_volume():
    for _ in range(10):
        client.post(
            "/api/v1/telephony/ping",
            json={"agent_id": "loud", "volume": 95.0, "latency": 15.0}
        )
    response = client.get("/api/v1/telephony/agent/loud")
    data = response.json()
    assert data["is_yelling"] is True


def test_normal_voice():
    for _ in range(20):
        client.post(
            "/api/v1/telephony/ping",
            json={"agent_id": "normal", "volume": 50.0, "latency": 20.0}
        )
    response = client.get("/api/v1/telephony/agent/normal")
    data = response.json()
    assert data["is_yelling"] is False
    assert data["is_silent"] is False
    assert data["volume_variance"] == 0.0


def test_get_all_agents():
    client.post("/api/v1/telephony/ping", json={"agent_id": "a1", "volume": 50, "latency": 10})
    client.post("/api/v1/telephony/ping", json={"agent_id": "a2", "volume": 60, "latency": 15})

    response = client.get("/api/v1/telephony/agents")
    data = response.json()
    assert len(data) == 2
    agent_ids = {item["agent_id"] for item in data}
    assert "a1" in agent_ids
    assert "a2" in agent_ids


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_unknown_agent():
    response = client.get("/api/v1/telephony/agent/nonexistent")
    data = response.json()
    assert data["agent_id"] == "nonexistent"
    assert data["data_points"] == 0
    assert data["avg_volume"] is None
    assert data["is_yelling"] is False
    assert data["is_silent"] is False


def test_high_concurrency():
    async def send_pings(agent_id, count):
        for i in range(count):
            client.post(
                "/api/v1/telephony/ping",
                json={"agent_id": agent_id, "volume": 50.0, "latency": 30.0}
            )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = [send_pings(f"concurrency_{i}", 20) for i in range(10)]
    loop.run_until_complete(asyncio.gather(*tasks))
    loop.close()

    response = client.get("/api/v1/telephony/agents")
    data = response.json()
    assert len(data) == 10
    for agent in data:
        assert agent["data_points"] == 20
