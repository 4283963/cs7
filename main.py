from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Optional
import json
import logging
import math
import os
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, PositiveFloat

try:
    import redis as redis_lib
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


app = FastAPI(
    title="呼叫中心通话质量动态审计系统",
    description="实时监控坐席通话音量波动与网络质量",
    version="1.0.0"
)

logger = logging.getLogger("telephony_audit")

WINDOW_SIZE = 50
HIGH_VARIANCE_THRESHOLD = 400.0
LOW_VOLUME_THRESHOLD = 5.0

SILENCE_RATIO_THRESHOLD = 0.80
LOW_LATENCY_THRESHOLD_MS = 80.0
MIN_POINTS_FOR_SILENCE_ALERT = 10

REDIS_ALERT_QUEUE_KEY = os.getenv("REDIS_ALERT_QUEUE_KEY", "telephony:alerts:silence")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
ALERT_COOLDOWN_SECONDS = 60

ALERT_TYPE_SILENCE = "agent_silence"


class PingRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, description="坐席唯一标识")
    volume: float = Field(..., ge=0.0, le=100.0, description="当前音量分贝值 0-100")
    latency: PositiveFloat = Field(..., description="网络延时，单位毫秒")


class DataPoint(BaseModel):
    timestamp: datetime
    volume: float
    latency: float


class AgentStatus(BaseModel):
    agent_id: str
    data_points: int
    avg_volume: Optional[float]
    volume_variance: Optional[float]
    volume_std: Optional[float]
    avg_latency: Optional[float]
    is_yelling: bool
    is_silent: bool
    silence_ratio: Optional[float]
    last_update: Optional[datetime]


class SilenceAlert(BaseModel):
    alert_id: str
    alert_type: str
    agent_id: str
    timestamp: datetime
    silence_ratio: float
    avg_latency: float
    data_points: int
    avg_volume: float


_rw_lock = threading.RLock()
_agent_data: Dict[str, deque] = {}
_last_alert_time: Dict[str, float] = {}

_redis_client = None
_redis_init_lock = threading.Lock()


def get_redis_client():
    global _redis_client
    if not _REDIS_AVAILABLE:
        return None
    with _redis_init_lock:
        if _redis_client is None:
            try:
                pool = redis_lib.ConnectionPool(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    db=REDIS_DB,
                    max_connections=20,
                    socket_timeout=2.0,
                    socket_connect_timeout=2.0
                )
                _redis_client = redis_lib.Redis(connection_pool=pool)
                _redis_client.ping()
                logger.info("Redis client connected: %s:%s/%s", REDIS_HOST, REDIS_PORT, REDIS_DB)
            except Exception as e:
                logger.warning("Redis connection failed, alerts will be logged only: %s", e)
                _redis_client = None
    return _redis_client


def calculate_variance(values: List[float]) -> Optional[float]:
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    return sum((x - mean) ** 2 for x in values) / n


def _get_or_create_deque(agent_id: str) -> deque:
    dq = _agent_data.get(agent_id)
    if dq is None:
        dq = deque(maxlen=WINDOW_SIZE)
        _agent_data[agent_id] = dq
    return dq


def _should_send_alert(agent_id: str) -> bool:
    now = time.monotonic()
    last = _last_alert_time.get(agent_id)
    if last is not None and now - last < ALERT_COOLDOWN_SECONDS:
        return False
    _last_alert_time[agent_id] = now
    return True


def _push_silence_alert(agent_id: str, silence_ratio: float, avg_latency: float,
                        data_points: int, avg_volume: float) -> Optional[str]:
    alert = SilenceAlert(
        alert_id=str(uuid.uuid4()),
        alert_type=ALERT_TYPE_SILENCE,
        agent_id=agent_id,
        timestamp=datetime.now(timezone.utc),
        silence_ratio=round(silence_ratio, 4),
        avg_latency=round(avg_latency, 2),
        data_points=data_points,
        avg_volume=round(avg_volume, 2)
    )
    payload = alert.model_dump_json()
    logger.warning("SILENCE ALERT: agent=%s ratio=%.2f avg_latency=%.1fms points=%d avg_vol=%.1f",
                   agent_id, silence_ratio, avg_latency, data_points, avg_volume)
    client = get_redis_client()
    if client is not None:
        try:
            client.rpush(REDIS_ALERT_QUEUE_KEY, payload)
        except Exception as e:
            logger.error("Failed to push alert to Redis queue: %s", e)
    return alert.alert_id


def _detect_silence_anomaly(points: List[DataPoint]) -> Optional[Dict]:
    n = len(points)
    if n < MIN_POINTS_FOR_SILENCE_ALERT:
        return None

    low_volume_count = sum(1 for p in points if p.volume < LOW_VOLUME_THRESHOLD)
    silence_ratio = low_volume_count / n
    avg_latency = sum(p.latency for p in points) / n

    if silence_ratio >= SILENCE_RATIO_THRESHOLD and avg_latency <= LOW_LATENCY_THRESHOLD_MS:
        avg_volume = sum(p.volume for p in points) / n
        return {
            "silence_ratio": silence_ratio,
            "avg_latency": avg_latency,
            "data_points": n,
            "avg_volume": avg_volume
        }
    return None


def analyze_agent(agent_id: str) -> AgentStatus:
    points = _agent_data.get(agent_id)
    if not points:
        return AgentStatus(
            agent_id=agent_id,
            data_points=0,
            avg_volume=None,
            volume_variance=None,
            volume_std=None,
            avg_latency=None,
            is_yelling=False,
            is_silent=False,
            silence_ratio=None,
            last_update=None
        )

    snapshot = list(points)

    volumes = [p.volume for p in snapshot]
    latencies = [p.latency for p in snapshot]

    avg_volume = sum(volumes) / len(volumes)
    variance = calculate_variance(volumes)
    std = math.sqrt(variance) if variance is not None else None
    avg_latency = sum(latencies) / len(latencies)

    low_volume_count = sum(1 for v in volumes if v < LOW_VOLUME_THRESHOLD)
    silence_ratio = low_volume_count / len(volumes)

    is_yelling = (variance is not None and variance > HIGH_VARIANCE_THRESHOLD) or avg_volume > 85
    is_silent = silence_ratio >= SILENCE_RATIO_THRESHOLD and len(volumes) >= MIN_POINTS_FOR_SILENCE_ALERT \
        and avg_latency <= LOW_LATENCY_THRESHOLD_MS

    return AgentStatus(
        agent_id=agent_id,
        data_points=len(snapshot),
        avg_volume=round(avg_volume, 2),
        volume_variance=round(variance, 2) if variance is not None else None,
        volume_std=round(std, 2) if std is not None else None,
        avg_latency=round(avg_latency, 2),
        is_yelling=is_yelling,
        is_silent=is_silent,
        silence_ratio=round(silence_ratio, 4),
        last_update=snapshot[-1].timestamp
    )


@app.post("/api/v1/telephony/ping", status_code=status.HTTP_204_NO_CONTENT, summary="高频接收通话实时数据")
async def telephony_ping(request: PingRequest):
    """
    前端通话 SDK 每隔 200ms 调用一次，上报当前音量和网络延时。
    - **agent_id**: 坐席工号
    - **volume**: 音量 0-100
    - **latency**: 网络延时 ms
    """
    point = DataPoint(
        timestamp=datetime.utcnow(),
        volume=request.volume,
        latency=request.latency
    )
    alert_id = None
    with _rw_lock:
        dq = _get_or_create_deque(request.agent_id)
        dq.append(point)
        snapshot = list(dq)
        anomaly = _detect_silence_anomaly(snapshot)
        if anomaly is not None and _should_send_alert(request.agent_id):
            alert_id = _push_silence_alert(request.agent_id, **anomaly)
    return None


@app.get("/api/v1/telephony/agent/{agent_id}", response_model=AgentStatus, summary="查询坐席实时状态")
async def get_agent_status(agent_id: str):
    """
    查询指定坐席最近 10 秒的通话质量分析结果。
    """
    with _rw_lock:
        status_result = analyze_agent(agent_id)
    return status_result


@app.get("/api/v1/telephony/agents", response_model=List[AgentStatus], summary="查询所有在线坐席状态")
async def get_all_agents():
    """
    查询所有上报过数据的坐席状态。
    """
    with _rw_lock:
        agent_ids = list(_agent_data.keys())
        results = [analyze_agent(aid) for aid in agent_ids]
    return results


@app.get("/health")
async def health_check():
    result = {"status": "ok", "redis": "unavailable"}
    client = get_redis_client()
    if client is not None:
        try:
            if client.ping():
                result["redis"] = "connected"
        except Exception:
            pass
    return result

