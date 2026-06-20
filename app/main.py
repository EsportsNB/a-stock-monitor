from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .schemas import (
    AddWatchRequest,
    StockHistoryResponse,
    StockListResponse,
    StockSummary,
    WatchlistCodesResponse,
    WatchlistResponse,
)
from .stock_monitor import (
    add_to_watchlist,
    cache_ready,
    disable_proxy_for_china_data,
    get_bid_ask,
    get_index_snapshots,
    get_stock_history,
    get_stock_list,
    get_stock_trends,
    get_watchlist_data,
    remove_from_watchlist,
    spot_source,
    start_background_cache_refresh,
)
from .realtime import start_realtime_watchlist

app = FastAPI(title="中国大陆股市实时监控服务")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.on_event("startup")
def startup_event():
    # 国内行情数据源无需走代理，直连可大幅加速（实测 ~60s → ~18s）
    disable_proxy_for_china_data()
    # 后台预热缓存，不阻塞启动，服务可立即响应
    start_background_cache_refresh()
    # 可选的实时行情推送（需配置 FINNHUB_API_KEY 环境变量以启用）
    try:
        start_realtime_watchlist()
    except Exception:
        # 不让 realtime 错误阻塞服务启动
        pass
    # 启动关注列表的快速轮询（无需外部 API key，提升 watchlist 的响应速度）
    try:
        from .stock_monitor import start_watchlist_polling

        start_watchlist_polling()
    except Exception:
        pass


def _require_cache_ready():
    if not cache_ready():
        raise HTTPException(status_code=503, detail="行情数据正在加载中，请几秒后重试。")


@app.get("/", response_class=FileResponse)
def index():
    return FileResponse("app/static/index.html")


@app.get("/health")
def health():
    """轻量健康检查：服务是否已就绪（行情缓存是否加载完成）及数据来源。"""
    ready = cache_ready()
    return {
        "status": "ok" if ready else "warming_up",
        "cache_ready": ready,
        "data_source": spot_source(),
    }


@app.get("/api/stocks", response_model=StockSummary)
async def get_stocks(symbols: str = Query(..., description="逗号分隔股票代码，支持 000001.SZ、600000.SH 或 000001")):
    codes = [part.strip() for part in symbols.split(",") if part.strip()]
    if not codes:
        raise HTTPException(status_code=400, detail="symbols 参数不能为空")
    _require_cache_ready()
    try:
        stocks = await run_in_threadpool(get_stock_trends, codes)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"symbols": codes, "stocks": stocks}


@app.get("/api/stocks/list", response_model=StockListResponse)
async def stock_list(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=10, le=100),
    market: Optional[str] = Query("all", regex="^(all|sh|sz|bj)$"),
    query: Optional[str] = Query(None),
    sort_by: str = Query("code", regex="^(code|name|current_price|change|change_percent|high|low|volume|volume_delta|volume_ratio|turnover)$"),
    order: str = Query("asc", regex="^(asc|desc)$"),
    min_pct: Optional[float] = Query(None, description="涨跌幅下限(%)"),
    max_pct: Optional[float] = Query(None, description="涨跌幅上限(%)"),
    trend: Optional[str] = Query(None, description="按趋势筛选，如 强势上涨/快速下跌"),
):
    _require_cache_ready()
    return await run_in_threadpool(
        get_stock_list, page, page_size, market or "all", query, sort_by, order, min_pct, max_pct, trend
    )


@app.get("/api/stock/{symbol}/history", response_model=StockHistoryResponse)
async def stock_history(symbol: str = Path(..., description="股票代码，支持 000001.SZ 或 600000.SH"), count: int = Query(30, ge=5, le=120)):
    _require_cache_ready()
    try:
        return await run_in_threadpool(get_stock_history, symbol, count)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/stock/{symbol}/bidask")
async def stock_bidask(symbol: str = Path(..., description="股票代码，支持 600519 / 600519.SH")):
    try:
        return await run_in_threadpool(get_bid_ask, symbol)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/indices")
async def get_indices():
    try:
        return await run_in_threadpool(get_index_snapshots)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


# ---- 关注列表 / 大户异动监控 ----

@app.get("/api/watchlist", response_model=WatchlistResponse)
async def watchlist():
    _require_cache_ready()
    return await run_in_threadpool(get_watchlist_data)


@app.post("/api/watchlist", response_model=WatchlistCodesResponse)
async def watchlist_add(req: AddWatchRequest):
    _require_cache_ready()
    try:
        codes = await run_in_threadpool(add_to_watchlist, req.symbol)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"codes": codes}


@app.delete("/api/watchlist/{symbol}", response_model=WatchlistCodesResponse)
async def watchlist_remove(symbol: str = Path(..., description="股票代码，支持 600519 / 600519.SH")):
    codes = await run_in_threadpool(remove_from_watchlist, symbol)
    return {"codes": codes}
