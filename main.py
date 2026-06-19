from collections import deque
from datetime import datetime
from typing import Dict, List, Optional
import math
import threading

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, PositiveFloat


app = FastAPI(
    title="呼叫中心通话质量动态审计系统",
    description="实时监控坐席通话音量波动与网络质量",
    version="1.0.0"
)

WINDOW_SIZE = 50
HIGH_VARIANCE_THRESHOLD = 400.0
LOW_VOLUME_THRESHOLD = 5.0


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
    last_update: Optional[datetime]


_rw_lock = threading.RLock()
_agent_data: Dict[str, deque] = {}


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
            last_update=None
        )

    snapshot = list(points)

    volumes = [p.volume for p in snapshot]
    latencies = [p.latency for p in snapshot]

    avg_volume = sum(volumes) / len(volumes)
    variance = calculate_variance(volumes)
    std = math.sqrt(variance) if variance is not None else None
    avg_latency = sum(latencies) / len(latencies)

    is_yelling = (variance is not None and variance > HIGH_VARIANCE_THRESHOLD) or avg_volume > 85
    is_silent = avg_volume < LOW_VOLUME_THRESHOLD and len(volumes) >= 10

    return AgentStatus(
        agent_id=agent_id,
        data_points=len(snapshot),
        avg_volume=round(avg_volume, 2),
        volume_variance=round(variance, 2) if variance is not None else None,
        volume_std=round(std, 2) if std is not None else None,
        avg_latency=round(avg_latency, 2),
        is_yelling=is_yelling,
        is_silent=is_silent,
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
    with _rw_lock:
        dq = _get_or_create_deque(request.agent_id)
        dq.append(point)
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
    return {"status": "ok"}
