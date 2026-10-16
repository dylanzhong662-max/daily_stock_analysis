# -*- coding: utf-8 -*-
"""美股选股 API（FMP company-screener）。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.deps import get_config_dep
from src.config import Config
from src.services.us_screener_service import PRESETS, screen_us_stocks

router = APIRouter()


class UsScreenRequest(BaseModel):
    preset: str = Field("largecap", max_length=32)
    sector: Optional[str] = Field(None, max_length=64)
    market_cap_min: Optional[float] = None
    price_min: Optional[float] = None
    volume_min: Optional[float] = None
    exchange: Optional[str] = Field(None, max_length=16)
    top: int = Field(10, ge=1, le=30)
    momentum: bool = False


class UsScreenCandidate(BaseModel):
    symbol: str
    name: str = ""
    sector: str = ""
    industry: str = ""
    marketCap: Optional[float] = None
    price: Optional[float] = None
    beta: Optional[float] = None
    exchange: str = ""
    aboveMa50Pct: Optional[float] = None


class UsScreenResponse(BaseModel):
    preset: str
    momentum: bool
    count: int
    candidates: List[UsScreenCandidate]


@router.get("/presets")
def us_screener_presets() -> Dict[str, Any]:
    """返回可用预设及其条件，供前端下拉展示。"""
    return {"presets": list(PRESETS.keys()), "details": PRESETS}


@router.post("/us", response_model=UsScreenResponse)
def screen_us(request: UsScreenRequest, config: Config = Depends(get_config_dep)) -> UsScreenResponse:
    try:
        result = screen_us_stocks(
            preset=request.preset,
            sector=request.sector,
            market_cap_min=request.market_cap_min,
            price_min=request.price_min,
            volume_min=request.volume_min,
            exchange=request.exchange,
            top=request.top,
            momentum=request.momentum,
            config=config,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail={"error": "us_screener_failed", "message": str(exc)})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail={"error": "us_screener_error", "message": str(exc)})
    return UsScreenResponse(**result)
