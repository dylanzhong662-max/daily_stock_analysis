# -*- coding: utf-8 -*-
"""TradingAgents 多智能体深度研究 API。

一次研究约 20-40 次 LLM 调用、耗时数分钟，因此只提供后台任务接口；
为控制内存，同一时间只允许一个研究任务运行。
"""

from __future__ import annotations

import threading
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.deps import get_config_dep
from api.v1.errors import api_error
from src.config import Config
from src.services.task_queue import TaskStatus as QueueTaskStatus
from src.services.task_queue import get_task_queue
from src.services.tradingagents_service import (
    TradingAgentsError,
    TradingAgentsService,
    list_reports,
    read_report,
    report_html_path,
)

router = APIRouter()

_TASK_REPORT_TYPE = "tradingagents"

# 单任务护栏：TradingAgents 一次运行占用数百 MB 内存，不允许并发
_run_lock = threading.Lock()
_running_task_id: Optional[str] = None


def _active_task_id() -> Optional[str]:
    """当前仍在排队/运行的研究任务 id；任务已结束或已被队列清理时返回 None。"""
    task_id = _running_task_id
    if not task_id:
        return None
    task = get_task_queue().get_task(task_id)
    if task is None or task.status not in (QueueTaskStatus.PENDING, QueueTaskStatus.PROCESSING):
        return None
    return task_id


class TradingAgentsTaskRequest(BaseModel):
    stock_code: str = Field(..., min_length=1, max_length=32)
    trade_date: Optional[str] = Field(None, description="YYYY-MM-DD，默认今天")
    analysts: Optional[List[str]] = Field(None, description="market/social/news/fundamentals 子集，默认取配置")
    polymarket_topics: Optional[List[str]] = Field(
        None, description="Polymarket 参考主题（最多 5 个），如 ['Anthropic IPO']，抓取后作为所有角色的参考"
    )


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

    global _running_task_id
    task_id = uuid.uuid4().hex
    task_queue = get_task_queue()

    def run_research() -> Dict[str, Any]:
        global _running_task_id
        try:
            task_queue.update_task_progress(task_id, 10, "TradingAgents 多智能体研究进行中，通常需要 3-5 分钟")
            return service.analyze(
                request.stock_code,
                trade_date=request.trade_date,
                analysts=request.analysts,
                polymarket_topics=request.polymarket_topics,
            )
        finally:
            with _run_lock:
                if _running_task_id == task_id:
                    _running_task_id = None

    with _run_lock:
        busy_id = _active_task_id()
        if busy_id:
            raise api_error(
                409,
                "tradingagents_busy",
                "已有 TradingAgents 研究任务在运行，请等待完成后再提交",
                detail={"task_id": busy_id},
            )
        _running_task_id = task_id
        try:
            task = task_queue.submit_background_task(
                run_research,
                stock_code=request.stock_code,
                stock_name=request.stock_code,
                report_type=_TASK_REPORT_TYPE,
                message="TradingAgents 研究任务已提交",
                task_id=task_id,
                trace_id=task_id,
            )
        except Exception:
            _running_task_id = None
            raise
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


@router.get("/running")
def tradingagents_running_task() -> Dict[str, Optional[str]]:
    with _run_lock:
        return {"task_id": _active_task_id()}


@router.get("/reports")
def tradingagents_reports(
    limit: int = Query(50, ge=1, le=200),
    config: Config = Depends(get_config_dep),
) -> Dict[str, Any]:
    return {"reports": list_reports(config, limit=limit)}


@router.get("/reports/{report_id}")
def tradingagents_report_detail(report_id: str, config: Config = Depends(get_config_dep)) -> Dict[str, Any]:
    try:
        return read_report(config, report_id)
    except TradingAgentsError as exc:
        raise _to_http_error(exc) from exc


@router.get("/reports/{report_id}/html")
def tradingagents_report_html(report_id: str) -> FileResponse:
    try:
        path = report_html_path(report_id)
    except TradingAgentsError as exc:
        raise _to_http_error(exc) from exc
    return FileResponse(path, media_type="text/html; charset=utf-8")
