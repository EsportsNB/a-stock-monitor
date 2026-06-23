import threading
from datetime import datetime
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
    get_stock_intraday,
    get_stock_list,
    get_stock_trends,
    get_watchlist_data,
    remove_from_watchlist,
    spot_source,
    start_background_cache_refresh,
    start_watchlist_polling,
)
from .ai_predictor import (
    list_predictions,
    load_prediction,
    review_pending,
    review_prediction,
    run_daily_prediction,
    update_summary,
    zhipu_available,
)
app = FastAPI(title="中国大陆股市实时监控服务")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.on_event("startup")
def startup_event():
    # 国内行情数据源无需走代理，直连可大幅加速（实测 ~60s → ~18s）
    disable_proxy_for_china_data()
    # 后台预热缓存，不阻塞启动，服务可立即响应
    start_background_cache_refresh()
    # 关注列表高频刷新（独立通道，比全市场 30s 更实时）
    start_watchlist_polling()


def _require_cache_ready():
    if not cache_ready():
        raise HTTPException(status_code=503, detail="行情数据正在加载中，请几秒后重试。")


@app.get("/", response_class=FileResponse)
def index():
    return FileResponse("app/static/index.html")


@app.get("/history", response_class=FileResponse)
def history_page():
    return FileResponse("app/static/history.html")


@app.get("/health")
def health():
    """轻量健康检查：服务是否已就绪（行情缓存是否加载完成）及数据来源。"""
    ready = cache_ready()
    return {
        "status": "ok" if ready else "warming_up",
        "cache_ready": ready,
        "data_source": spot_source(),
        "ai_ready": zhipu_available(),
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
    market: Optional[str] = Query("all", regex="^(all|sh|sz|bj|tw)$"),
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


@app.get("/api/stock/{symbol}/intraday")
async def stock_intraday(
    symbol: str = Path(..., description="股票代码，支持 600519 / 600519.SH"),
    scale: int = Query(1, ge=1, le=60, description="K 线周期（分钟），1/5/15/30/60"),
    day_offset: int = Query(0, ge=0, le=15, description="0=当天，1=前一交易日，依次往前"),
):
    """分时行情（新浪分钟级 K 线）。day_offset 切换历史交易日（受新浪分钟数据保留天数限制）。"""
    try:
        return await run_in_threadpool(get_stock_intraday, symbol, scale, day_offset)
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


# ---- AI 涨跌预测 ----

_predict_status = {"running": False, "last_date": None, "error": None}


@app.post("/api/predict/run")
async def predict_run(web_search: bool = Query(True, description="是否启用联网搜索新闻")):
    """触发当日预测（后台运行，约几分钟）。立即返回，结果用 GET /api/predict/today 查看。"""
    _require_cache_ready()
    if not zhipu_available():
        raise HTTPException(status_code=503, detail="未配置 ZHIPU_API_KEY 环境变量，无法调用 AI 模型。请到 https://open.bigmodel.cn 免费获取后设置。")
    if _predict_status["running"]:
        return {"status": "already_running", "message": "预测正在进行中，请稍候。"}

    def _job():
        _predict_status.update(running=True, error=None)
        try:
            rec = run_daily_prediction(use_web_search=web_search)
            _predict_status["last_date"] = rec["date"]
        except Exception as exc:  # noqa: BLE001
            _predict_status["error"] = str(exc)
            print(f"AI 预测任务失败：{exc}")
        finally:
            _predict_status["running"] = False

    threading.Thread(target=_job, daemon=True).start()
    return {"status": "started", "message": "预测已在后台开始，约几分钟后用 GET /api/predict/today 查看结果。"}


@app.post("/api/predict/review/{date}")
async def predict_review(date: str = Path(..., description="要复盘的预测日期 YYYY-MM-DD（在其次日收盘后调用）")):
    _require_cache_ready()
    try:
        return await run_in_threadpool(review_prediction, date)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/api/predict/review-pending")
async def predict_review_pending():
    """复盘最近一个未复盘的预测（供每日调度调用，自动对上一交易日的预测打分）。"""
    _require_cache_ready()
    rec = await run_in_threadpool(review_pending)
    if rec is None:
        return {"status": "nothing_to_review"}
    return rec


@app.get("/api/predict/status")
def predict_status():
    return _predict_status



@app.get("/api/predict/history")
def predict_history():
    """所有历史预测概览 + 累计准确率，供历史页展示。"""
    return list_predictions()


@app.get("/api/predict/today")
def predict_today():
    rec = load_prediction(datetime.now().strftime("%Y-%m-%d"))
    if rec is None:
        return {"status": "no_prediction", "running": _predict_status["running"], "error": _predict_status["error"]}
    return rec


@app.get("/api/predict/{date}")
def predict_get(date: str = Path(..., description="预测日期 YYYY-MM-DD")):
    rec = load_prediction(date)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"未找到 {date} 的预测记录")
    return rec
