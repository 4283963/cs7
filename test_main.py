import asyncio
import json
import random
import threading
import time
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

import main
from main import (
    app, calculate_variance, WINDOW_SIZE, _agent_data, _rw_lock,
    _last_alert_time, _detect_silence_anomaly, _should_send_alert,
    SILENCE_RATIO_THRESHOLD, LOW_LATENCY_THRESHOLD_MS,
    MIN_POINTS_FOR_SILENCE_ALERT, DataPoint, REDIS_ALERT_QUEUE_KEY,
    ALERT_COOLDOWN_SECONDS
)
from datetime import datetime, timezone

client = TestClient(app)


def setup_function():
    with _rw_lock:
        _agent_data.clear()
        _last_alert_time.clear()
    try:
        from main import _redis_client as rc
        import main as m
        with m._redis_init_lock:
            m._redis_client = None
    except Exception:
        pass


def make_points(count, low_vol_count, low_latency=True, low_val=2.0, high_val=50.0, latency=30.0):
    pts = []
    now = datetime.utcnow()
    for i in range(count):
        v = low_val if i < low_vol_count else high_val
        l = latency if low_latency else 200.0
        pts.append(DataPoint(timestamp=now, volume=v, latency=l))
    return pts


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
    data = response.json()
    assert data["status"] == "ok"
    assert "redis" in data


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


def test_race_condition_dict_iteration_with_writes():
    errors = []
    stop_event = threading.Event()

    def writer_thread(writer_id):
        try:
            while not stop_event.is_set():
                for i in range(5):
                    agent_id = f"writer_{writer_id}_agent_{random.randint(1, 50)}"
                    client.post(
                        "/api/v1/telephony/ping",
                        json={"agent_id": agent_id, "volume": random.uniform(0, 100), "latency": random.uniform(10, 200)}
                    )
        except Exception as e:
            errors.append(("writer", str(e)))

    def monitor_thread():
        try:
            while not stop_event.is_set():
                r = client.get("/api/v1/telephony/agents")
                assert r.status_code == 200
                r.json()
                r2 = client.get(f"/api/v1/telephony/agent/monitor_probe_{random.randint(1, 100)}")
                assert r2.status_code == 200
        except Exception as e:
            errors.append(("monitor", str(e)))

    writers = [threading.Thread(target=writer_thread, args=(i,)) for i in range(5)]
    monitors = [threading.Thread(target=monitor_thread) for _ in range(3)]

    for t in writers + monitors:
        t.start()

    time.sleep(2.0)

    stop_event.set()
    for t in writers + monitors:
        t.join(timeout=5.0)

    assert len(errors) == 0, f"Concurrency errors occurred: {errors[:5]}"

    with _rw_lock:
        assert len(_agent_data) > 0


def test_dict_changed_size_during_iteration_prevented():
    errors = []

    def aggressive_writer():
        for i in range(500):
            agent_id = f"aggressive_{i}"
            client.post(
                "/api/v1/telephony/ping",
                json={"agent_id": agent_id, "volume": 50.0, "latency": 20.0}
            )

    def aggressive_reader():
        try:
            for _ in range(500):
                client.get("/api/v1/telephony/agents")
        except Exception as e:
            errors.append(str(e))

    t1 = threading.Thread(target=aggressive_writer)
    t2 = threading.Thread(target=aggressive_reader)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(errors) == 0, f"Got errors: {errors[:3]}"


def test_detect_silence_anomaly_insufficient_points():
    points = make_points(5, 5)
    assert _detect_silence_anomaly(points) is None


def test_detect_silence_anomaly_true_80_percent_low_volume_low_latency():
    points = make_points(50, 40)
    result = _detect_silence_anomaly(points)
    assert result is not None
    assert result["silence_ratio"] == 0.80
    assert result["avg_latency"] <= LOW_LATENCY_THRESHOLD_MS
    assert result["data_points"] == 50


def test_detect_silence_anomaly_false_not_enough_low_volume():
    points = make_points(50, 39)
    result = _detect_silence_anomaly(points)
    assert result is None


def test_detect_silence_anomaly_false_high_latency():
    points = make_points(50, 45, low_latency=False)
    result = _detect_silence_anomaly(points)
    assert result is None


def test_detect_silence_anomaly_boundary_exactly_80_percent():
    points = make_points(10, 8)
    result = _detect_silence_anomaly(points)
    assert result is not None
    assert result["silence_ratio"] == 0.80


def test_alert_cooldown_mechanism():
    with _rw_lock:
        _last_alert_time.pop("cool_agent", None)
        assert _should_send_alert("cool_agent") is True
        assert _should_send_alert("cool_agent") is False
        _last_alert_time["cool_agent"] = time.monotonic() - ALERT_COOLDOWN_SECONDS - 1
        assert _should_send_alert("cool_agent") is True


def test_silence_alert_pushed_to_redis():
    with _rw_lock:
        _last_alert_time.pop("muted_007", None)
        _agent_data.pop("muted_007", None)

    mock_redis = MagicMock()
    mock_redis.ping.return_value = True

    captured = {}

    def fake_rpush(key, payload):
        captured["key"] = key
        captured["payload"] = payload
        return 1

    mock_redis.rpush.side_effect = fake_rpush

    with patch.object(main, "_redis_client", mock_redis):
        with patch("main.get_redis_client", return_value=mock_redis):
            for i in range(50):
                v = 1.0 if i < 42 else 60.0
                r = client.post(
                    "/api/v1/telephony/ping",
                    json={"agent_id": "muted_007", "volume": v, "latency": 25.0}
                )
                assert r.status_code == 204

    assert "key" in captured, "Expected Redis rpush to be called for silence alert"
    assert captured["key"] == REDIS_ALERT_QUEUE_KEY
    payload = json.loads(captured["payload"])
    assert payload["alert_type"] == "agent_silence"
    assert payload["agent_id"] == "muted_007"
    assert payload["silence_ratio"] >= 0.80
    assert payload["avg_latency"] <= LOW_LATENCY_THRESHOLD_MS
    assert "alert_id" in payload
    assert "timestamp" in payload


def test_silence_alert_not_fired_when_high_latency():
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True

    with patch.object(main, "_redis_client", mock_redis):
        with patch("main.get_redis_client", return_value=mock_redis):
            for i in range(50):
                v = 1.0 if i < 48 else 80.0
                client.post(
                    "/api/v1/telephony/ping",
                    json={"agent_id": "bad_net", "volume": v, "latency": 300.0}
                )

    mock_redis.rpush.assert_not_called()


def test_silence_alert_not_fired_when_enough_volume():
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True

    with patch.object(main, "_redis_client", mock_redis):
        with patch("main.get_redis_client", return_value=mock_redis):
            for i in range(50):
                client.post(
                    "/api/v1/telephony/ping",
                    json={"agent_id": "talking_01", "volume": 55.0, "latency": 30.0}
                )

    mock_redis.rpush.assert_not_called()


def test_agent_status_includes_silence_ratio():
    for i in range(20):
        v = 1.0 if i < 16 else 50.0
        client.post(
            "/api/v1/telephony/ping",
            json={"agent_id": "ratio_test", "volume": v, "latency": 20.0}
        )
    response = client.get("/api/v1/telephony/agent/ratio_test")
    data = response.json()
    assert "silence_ratio" in data
    assert abs(data["silence_ratio"] - 0.80) < 0.001
    assert data["is_silent"] is True


def test_health_check_redis_status():
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True
    with patch.object(main, "_redis_client", mock_redis):
        with patch("main.get_redis_client", return_value=mock_redis):
            response = client.get("/health")
    data = response.json()
    assert data["status"] == "ok"
    assert data["redis"] == "connected"


def test_redis_fail_graceful_alert_still_logged():
    import main as m
    m._redis_client = None

    bad_redis = MagicMock()
    bad_redis.ping.side_effect = Exception("connection refused")

    with patch("main.redis_lib") as mock_lib:
        mock_lib.ConnectionPool.return_value = MagicMock()
        mock_lib.Redis.return_value = bad_redis
        with patch("main.get_redis_client", return_value=None):
            for i in range(15):
                v = 1.0 if i < 13 else 70.0
                r = client.post(
                    "/api/v1/telephony/ping",
                    json={"agent_id": "redis_down", "volume": v, "latency": 22.0}
                )
                assert r.status_code == 204, f"Should not crash when Redis down: {r.status_code}"
