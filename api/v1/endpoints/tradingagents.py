# -*- coding: utf-8 -*-
"""TradingAgents 多智能体深度研究 API。

一次研究约 20-40 次 LLM 调用、耗时数分钟，因此只提供后台任务接口。
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.deps import get_config_dep
from api.v1.errors import api_error
from src.config import Config
from src.services.task_queue import TaskStatus as QueueTaskStatus
from src.services.task_queue import get_task_queue
from src.services.tradingagents_service import TradingAgentsError, TradingAgentsService

router = APIRouter()

_TASK_REPORT_TYPE = "tradingagents"


class TradingAgentsTaskRequest(BaseModel):
    stock_code: str = Field(..., min_length=1, max_length=32)
    trade_date: Optional[str] = Field(None, description="YYYY-MM-DD，默认今天")
    analysts: Optional[List[str]] = Field(None, description="market/social/news/fundamentals 子集，默认取配置")


class TradingAgentsTaskAccepted(BaseModel):
    task_id: str
    trace_id: str
    status: str = "pending"
    message: str
    stock_code: str


class TradingAgentsTaskStatus(BaseModel):
    task_id: str
    trace_id: Optional[str] = None
    status: str
    progress: int = 0
    message: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


def _to_http_error(exc: TradingAgentsError) -> HTTPException:
    return api_error(exc.status_code, exc.code, exc.message)


def _status_value(status: Any) -> str:
    return status.value if isinstance(status, QueueTaskStatus) else str(status)


@router.get("/status")
def tradingagents_status(config: Config = Depends(get_config_dep)) -> Dict[str, Any]:
    return TradingAgentsService(config).status()


@router.post("/tasks", status_code=202, response_model=TradingAgentsTaskAccepted)
def tradingagents_start_task(
    request: TradingAgentsTaskRequest,
    config: Config = Depends(get_config_dep),
) -> TradingAgentsTaskAccepted:
    service = TradingAgentsService(config)
    try:
        service.ensure_ready()
    except TradingAgentsError as exc:
        raise _to_http_error(exc) from exc

    task_id = uuid.uuid4().hex
    task_queue = get_task_queue()

    def run_research() -> Dict[str, Any]:
        task_queue.update_task_progress(task_id, 10, "TradingAgents 多智能体研究进行中，通常需要数分钟")
        return service.analyze(
            request.stock_code,
            trade_date=request.trade_date,
            analysts=request.analysts,
        )

    task = task_queue.submit_background_task(
        run_research,
        stock_code=request.stock_code,
        stock_name=request.stock_code,
        report_type=_TASK_REPORT_TYPE,
        message="TradingAgents 研究任务已提交",
        task_id=task_id,
        trace_id=task_id,
    )
    return TradingAgentsTaskAccepted(
        task_id=task.task_id,
        trace_id=task.trace_id or task.task_id,
        status=_status_value(task.status),
        message=task.message or "TradingAgents 研究任务已提交",
        stock_code=request.stock_code,
    )


@router.get("/tasks/{task_id}", response_model=TradingAgentsTaskStatus)
def tradingagents_task_status(task_id: str) -> TradingAgentsTaskStatus:
    task = get_task_queue().get_task(task_id)
    if task is None or task.report_type != _TASK_REPORT_TYPE:
        raise api_error(404, "tradingagents_task_not_found", f"TradingAgents 任务 {task_id} 不存在或已过期")

    result = task.result if task.status == QueueTaskStatus.COMPLETED and isinstance(task.result, dict) else None
    return TradingAgentsTaskStatus(
        task_id=task.task_id,
        trace_id=task.trace_id or task.task_id,
        status=_status_value(task.status),
        progress=task.progress,
        message=task.message,
        error=task.error,
        result=result,
    )
