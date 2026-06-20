from typing import List, Optional

from pydantic import BaseModel


class StockTrend(BaseModel):
    symbol: str
    name: str
    current_price: float
    previous_close: float
    change: float
    change_percent: float
    high: Optional[float]
    low: Optional[float]
    open: Optional[float]
    volume: Optional[float]
    turnover: Optional[float]
    trend: str
    summary: str
    timestamp: str


class StockItem(BaseModel):
    symbol: str
    name: str
    current_price: Optional[float]
    previous_close: Optional[float]
    change: Optional[float]
    change_percent: Optional[float]
    high: Optional[float]
    low: Optional[float]
    open: Optional[float]
    volume: Optional[float]
    volume_delta: Optional[float] = None     # 即时成交量（一个刷新周期内的成交量增量）
    turnover: Optional[float]
    turnover_delta: Optional[float] = None   # 即时成交额
    volume_baseline: Optional[float] = None  # 成交量增量基线（中位数）
    volume_ratio: Optional[float] = None     # 放量倍数 = 本次增量 / 基线
    alert: Optional[str] = None              # None | big_buy | big_sell | big_trade
    alert_level: int = 0                     # 0-3 异动强度
    trend: str
    timestamp: str


class StockListResponse(BaseModel):
    page: int
    page_size: int
    total: int
    stocks: List[StockItem]
    sort_by: Optional[str] = None
    order: Optional[str] = None
    updated_at: Optional[str] = None
    refresh_interval: Optional[int] = None


class StockHistoryPoint(BaseModel):
    date: str
    open: Optional[float]
    close: Optional[float]
    high: Optional[float]
    low: Optional[float]
    volume: Optional[float]


class StockHistoryResponse(BaseModel):
    symbol: str
    name: str
    history: List[StockHistoryPoint]


class StockSummary(BaseModel):
    symbols: List[str]
    stocks: List[StockTrend]


# ---- 关注列表 / 大户异动监控 ----

class WatchStock(BaseModel):
    symbol: str
    name: str
    current_price: Optional[float]
    previous_close: Optional[float]
    change: Optional[float]
    change_percent: Optional[float]
    high: Optional[float]
    low: Optional[float]
    open: Optional[float]
    volume: Optional[float]
    turnover: Optional[float]
    trend: str
    summary: Optional[str] = None
    timestamp: str
    # 异动监控字段
    volume_delta: Optional[float] = None       # 即时成交量（一个刷新周期内的成交量）
    turnover_delta: Optional[float] = None      # 即时成交额
    price_delta: Optional[float] = None         # 相比上次快照的价格变化
    volume_baseline: Optional[float] = None     # 成交量增量基线（中位数）
    volume_ratio: Optional[float] = None        # 本次增量 / 基线
    alert: Optional[str] = None                 # None | big_buy | big_sell | big_trade
    alert_level: int = 0                        # 0-3 异动强度


class WatchAlert(BaseModel):
    symbol: str
    name: str
    alert: str
    alert_level: int
    volume_delta: Optional[float] = None
    turnover_delta: Optional[float] = None
    ratio: Optional[float] = None
    change_percent: Optional[float] = None
    current_price: Optional[float] = None


class WatchlistResponse(BaseModel):
    stocks: List[WatchStock]
    alerts: List[WatchAlert]
    updated_at: str
    refresh_interval: int


class WatchlistCodesResponse(BaseModel):
    codes: List[str]


class AddWatchRequest(BaseModel):
    symbol: str
