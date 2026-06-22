import os
import json
import math
import re
import akshare as ak
import pandas as pd
import requests
import time
from collections import deque
from datetime import datetime, timedelta
from threading import RLock, Thread
from typing import List, Optional

CACHE_TTL_SECONDS = 30
# 关注列表要做“即时成交量/大户异动”监控，刷新越快越灵敏。
# 绕过代理后全市场单次拉取约 15s，30s 间隔兼顾灵敏度与负载（间隔过小会与拉取时间重叠）。
CACHE_REFRESH_INTERVAL = 30

# ---- 异动检测参数 ----
VOLUME_HISTORY_LEN = 20      # 每只股票保留最近 N 次成交量增量，用于计算基线
MIN_BASELINE_SAMPLES = 3     # 基线至少需要的样本数（不足则不告警，避免冷启动误报）
ALERT_MULTIPLIER = 3.0       # 当前增量 ≥ 基线中位数 × 该倍数 视为放量异动

_cache_lock = RLock()
_cache = {
    "spot_df": None,
    "spot_at": None,
    "spot_source": None,
    "index_df": None,
    "index_at": None,
}

# 全市场每只股票上一次的累计值/价格 + 成交量增量历史，用于计算即时成交量与异动。
# 仅后台刷新线程写入，故无需加锁。
_market_prev_volume = {}
_market_prev_turnover = {}
_market_prev_price = {}
_market_delta_history = {}      # full_code -> deque[成交量增量]

# 关注列表高频刷新通道（独立于全市场 30s 刷新，用独立状态避免污染全市场基线）。
WATCH_POLL_INTERVAL = 8         # 关注列表单独刷新间隔（秒）
_watch_prev_volume = {}
_watch_prev_price = {}
_watch_delta_history = {}       # full_code -> deque[成交量增量]
_watch_cache = {"items": None, "at": None}   # 关注列表高频快照

# 关注列表持久化
WATCHLIST_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "watchlist.json"
)
_watchlist_lock = RLock()


def disable_proxy_for_china_data():
    """行情数据源（新浪/东财）都是国内服务器，走境外代理会变慢甚至连不上。
    本服务只访问国内数据源，因此在进程内移除代理环境变量，让 akshare 直连。
    """
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        os.environ.pop(key, None)
    # 显式声明无代理，避免底层库重新探测系统代理
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


def cache_ready() -> bool:
    """实时行情快照是否已加载（首次启动预热完成后为 True）。"""
    with _cache_lock:
        return _cache["spot_df"] is not None


def spot_source() -> Optional[str]:
    """当前实时行情数据来源：'eastmoney'（东财）或 'sina'（新浪）。"""
    with _cache_lock:
        return _cache["spot_source"]


def normalize_symbol(symbol: str) -> str:
    """Return symbol in NNNNNN.XX form (e.g. 600519.SH).
    Accepts: 600519, sh600519, 600519.SH, etc.
    """
    s = symbol.strip()
    lower = s.lower()
    # Already in dot-suffix form
    upper = s.upper()
    if upper.endswith(".SH") or upper.endswith(".SZ") or upper.endswith(".BJ"):
        return upper
    # akshare-style prefix (sh/sz/bj)
    if lower.startswith("sh"):
        return f"{s[2:].upper()}.SH"
    if lower.startswith("sz"):
        return f"{s[2:].upper()}.SZ"
    if lower.startswith("bj"):
        return f"{s[2:].upper()}.BJ"
    # Auto-detect from numeric code. The exact exchange is confirmed later by a
    # bare-code lookup against live data, so these prefixes are only a best guess.
    code = s.upper()
    if code.startswith("6"):
        return f"{code}.SH"
    if code.startswith("4") or code.startswith("8") or code.startswith("9"):
        return f"{code}.BJ"
    return f"{code}.SZ"


def normalize_history_symbol(symbol: str) -> str:
    """Convert to akshare daily format: sh600519 / sz000001."""
    sym = normalize_symbol(symbol)
    code = sym.split(".")[0]
    if sym.endswith(".SH"):
        return f"sh{code}"
    if sym.endswith(".BJ"):
        return f"bj{code}"
    return f"sz{code}"


def _parse_number(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        return None if math.isnan(f) else f   # NaN 不是合法 JSON，统一转 None
    text = str(value).strip().replace(",", "")
    if text == "-" or text == "":
        return None
    try:
        f = float(text)
        return None if math.isnan(f) else f
    except ValueError:
        return None


def _normalize_change_percent(value) -> Optional[float]:
    """Return change as a decimal fraction (0.03 = 3%). akshare returns percentage floats/strings."""
    if value is None:
        return None
    if isinstance(value, str) and "%" in value:
        try:
            return float(value.replace("%", "")) / 100.0
        except ValueError:
            return None
    num = _parse_number(value)
    if num is None:
        return None
    return num / 100.0


def _needs_refresh(timestamp):
    if timestamp is None:
        return True
    return (datetime.now() - timestamp).total_seconds() > CACHE_TTL_SECONDS


def _normalize_spot_df(df):
    if "代码" in df.columns:
        df["代码"] = df["代码"].astype(str).str.strip()
    return df


def _normalize_em_codes(df):
    """东财接口返回纯数字代码（600519），这里统一补上市场前缀（sh600519），
    与新浪接口格式保持一致，使下游逻辑无需区分数据源。
    """
    if "代码" in df.columns:
        df = df.copy()
        df["代码"] = df["代码"].astype(str).str.strip().map(_symbol_to_full_code)
    return df


def _fetch_spot_data():
    """返回 (DataFrame, 数据源名称)。
    优先尝试腾讯接口（若可用通常更稳定），随后回退到东财，再回退到新浪逐页接口。
    """
    # 如果 akshare 版本支持腾讯接口则优先尝试，否则静默跳过
    if hasattr(ak, "stock_zh_a_spot_tx"):
        try:
            df = ak.stock_zh_a_spot_tx()
            return _normalize_spot_df(df), "tencent"
        except Exception as exc:
            print(f"Info: 腾讯行情接口不可用，尝试东财（{exc}）")
    else:
        # akshare 旧版本可能没有该接口，直接回退到东财
        pass
    try:
        df = ak.stock_zh_a_spot_em()
        return _normalize_spot_df(_normalize_em_codes(df)), "eastmoney"
    except Exception as exc:
        print(f"Info: 东财行情接口不可用，回落新浪（{exc}）")
        df = ak.stock_zh_a_spot()
        return _normalize_spot_df(df), "sina"


def _fetch_index_data():
    return ak.stock_zh_index_spot_sina()


def _is_nan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


def _classify_alert(ratio, price_delta):
    """放量倍数达到阈值时，按价格方向判定异动类型与强度。"""
    if ratio is None or ratio < ALERT_MULTIPLIER:
        return None, 0
    if price_delta is not None and price_delta > 0:
        alert = "big_buy"
    elif price_delta is not None and price_delta < 0:
        alert = "big_sell"
    else:
        alert = "big_trade"   # 放量但价格暂未明确方向
    level = 1 if ratio < 5 else (2 if ratio < 10 else 3)
    return alert, level


def _compute_trend_series(df):
    """向量化计算趋势列（与 describe_trend 同口径），用于全市场展示与筛选。"""
    pct = pd.to_numeric(df["涨跌幅"], errors="coerce") / 100.0
    price = pd.to_numeric(df["最新价"], errors="coerce")
    prevc = pd.to_numeric(df["昨收"], errors="coerce")
    trend = pd.Series("横盘震荡", index=df.index, dtype=object)
    trend[price > prevc] = "盘中微涨"
    trend[price < prevc] = "盘中微跌"
    trend[pct >= 0.01] = "温和上涨"
    trend[pct >= 0.03] = "强势上涨"
    trend[pct <= -0.01] = "温和下跌"
    trend[pct <= -0.03] = "快速下跌"
    trend[pct.isna()] = "趋势未知"
    return trend


def _attach_market_metrics(df):
    """为全市场快照逐只附加即时成交量/成交额、基线、放量倍数、异动、异动级别、趋势列。

    即时成交量 = 本次累计成交量 − 上次累计成交量（一个刷新周期内的成交量）；
    基线 = 该股最近若干次增量的中位数；放量倍数 = 本次增量 / 基线；
    放量倍数达阈值时按价格方向判定大买/大卖。5500 只逐只循环，开销约几十毫秒。
    """
    codes = df["代码"].astype(str).tolist()
    vols = pd.to_numeric(df["成交量"], errors="coerce").tolist()
    turnovers = pd.to_numeric(df["成交额"], errors="coerce").tolist()
    prices = pd.to_numeric(df["最新价"], errors="coerce").tolist()

    deltas, tdeltas, baselines, ratios, alerts, levels = [], [], [], [], [], []
    for code, vol, tov, price in zip(codes, vols, turnovers, prices):
        prev_vol = _market_prev_volume.get(code)
        prev_tov = _market_prev_turnover.get(code)
        prev_price = _market_prev_price.get(code)

        vol_delta = None
        tov_delta = None
        if prev_vol is not None and not _is_nan(vol):
            d = vol - prev_vol
            vol_delta = d if d >= 0 else None
            if vol_delta is not None and prev_tov is not None and not _is_nan(tov):
                td = tov - prev_tov
                tov_delta = td if td >= 0 else None
        price_delta = None
        if prev_price is not None and not _is_nan(price):
            price_delta = price - prev_price

        baseline = ratio = None
        alert = None
        level = 0
        if vol_delta is not None:
            hist = _market_delta_history.setdefault(code, deque(maxlen=VOLUME_HISTORY_LEN))
            if len(hist) >= MIN_BASELINE_SAMPLES:
                baseline = _median(hist)
                if baseline and baseline > 0:
                    ratio = vol_delta / baseline
                    if vol_delta > 0:
                        alert, level = _classify_alert(ratio, price_delta)
            hist.append(vol_delta)   # 本次增量在判定后再入历史，避免污染当次基线

        if not _is_nan(vol):
            _market_prev_volume[code] = vol
        if not _is_nan(tov):
            _market_prev_turnover[code] = tov
        if not _is_nan(price):
            _market_prev_price[code] = price

        deltas.append(vol_delta)
        tdeltas.append(tov_delta)
        baselines.append(baseline)
        ratios.append(round(ratio, 2) if ratio is not None else None)
        alerts.append(alert)
        levels.append(level)

    out = df.copy()
    out["即时成交量"] = deltas
    out["即时成交额"] = tdeltas
    out["基线成交量"] = baselines
    out["放量倍数"] = ratios
    out["异动"] = alerts
    out["异动级别"] = levels
    out["趋势"] = _compute_trend_series(out).values
    return out


def _refresh_spot_cache():
    df, source = _fetch_spot_data()
    try:
        df = _attach_market_metrics(df)   # 全市场即时成交量与异动指标
    except Exception as exc:
        print(f"Warning: attach market metrics failed: {exc}")
    with _cache_lock:
        _cache["spot_df"] = df
        _cache["spot_at"] = datetime.now()
        _cache["spot_source"] = source


def _refresh_index_cache():
    df = _fetch_index_data()   # 网络请求在锁外，避免阻塞其他读者
    with _cache_lock:
        _cache["index_df"] = df
        _cache["index_at"] = datetime.now()


def refresh_all_caches():
    try:
        _refresh_spot_cache()
    except Exception as exc:
        print(f"Warning: refresh spot cache failed: {exc}")
    try:
        _refresh_index_cache()
    except Exception as exc:
        print(f"Warning: refresh index cache failed: {exc}")


def start_background_cache_refresh(interval_seconds: int = CACHE_REFRESH_INTERVAL):
    """启动后台刷新线程：立即预热一次，之后每 interval_seconds 刷新。
    线程是 daemon，且不阻塞调用方，因此服务可立即对外提供（首页等）服务。
    """
    def _loop():
        refresh_all_caches()  # 立即预热
        while True:
            time.sleep(interval_seconds)
            refresh_all_caches()

    thread = Thread(target=_loop, daemon=True)
    thread.start()


def get_spot_data():
    with _cache_lock:
        if _cache["spot_df"] is None:
            _refresh_spot_cache()
        return _cache["spot_df"]


def get_index_data():
    with _cache_lock:
        if _cache["index_df"] is None:
            _refresh_index_cache()
        return _cache["index_df"]


def describe_trend(change_pct: float, current_price: Optional[float], prev_close: Optional[float]) -> str:
    if change_pct is None:
        return "趋势未知"
    if change_pct >= 0.03:
        return "强势上涨"
    if change_pct >= 0.01:
        return "温和上涨"
    if change_pct <= -0.03:
        return "快速下跌"
    if change_pct <= -0.01:
        return "温和下跌"
    if current_price is not None and prev_close is not None:
        if current_price > prev_close:
            return "盘中微涨"
        if current_price < prev_close:
            return "盘中微跌"
    return "横盘震荡"


def make_summary(name: str, symbol: str, current_price: float, prev_close: float, change: float, change_pct: float, trend: str) -> str:
    direction = "上涨" if change >= 0 else "下跌"
    return (
        f"{name}（{symbol}）现价 {current_price:.2f} 元，较昨收 {prev_close:.2f} 元 {direction} {abs(change):.2f} 元，"
        f"涨跌幅 {change_pct * 100:.2f}% 。当前判断：{trend}。"
    )


def _full_code_to_symbol(full_code: str) -> str:
    """Convert akshare-style code (sh600519) to dot-suffix form (600519.SH)."""
    lower = full_code.lower()
    if lower.startswith("sh"):
        return f"{full_code[2:]}.SH"
    if lower.startswith("bj"):
        return f"{full_code[2:]}.BJ"
    if lower.startswith("sz"):
        return f"{full_code[2:]}.SZ"
    # Fallback: guess from numeric prefix
    if full_code.startswith("6"):
        return f"{full_code}.SH"
    return f"{full_code}.SZ"


def _symbol_to_full_code(symbol: str) -> str:
    """Convert dot-suffix form (600519.SH) to akshare-style code (sh600519)."""
    sym = normalize_symbol(symbol)
    code = sym.split(".")[0]
    if sym.endswith(".SH"):
        return f"sh{code}"
    if sym.endswith(".BJ"):
        return f"bj{code}"
    return f"sz{code}"


def _build_stock_item(row) -> dict:
    full_code = str(row.get("代码", ""))
    symbol = _full_code_to_symbol(full_code)
    current_price = _parse_number(row.get("最新价"))
    prev_close = _parse_number(row.get("昨收"))
    change = _parse_number(row.get("涨跌额"))
    change_pct = _normalize_change_percent(row.get("涨跌幅"))
    if change_pct is None:
        change_pct = 0.0
    trend_val = row.get("趋势")
    trend = trend_val if isinstance(trend_val, str) and trend_val else describe_trend(change_pct, current_price, prev_close)
    summary = make_summary(
        name=str(row.get("名称")),
        symbol=symbol,
        current_price=current_price or 0.0,
        prev_close=prev_close or 0.0,
        change=change or 0.0,
        change_pct=change_pct,
        trend=trend,
    )
    return {
        "symbol": symbol,
        "name": str(row.get("名称")),
        "current_price": current_price,
        "previous_close": prev_close,
        "change": round(change or 0.0, 2),
        "change_percent": round(change_pct * 100, 2),
        "high": _parse_number(row.get("最高")),
        "low": _parse_number(row.get("最低")),
        "open": _parse_number(row.get("今开")),
        "volume": _parse_number(row.get("成交量")),
        "volume_delta": _parse_number(row.get("即时成交量")),
        "turnover": _parse_number(row.get("成交额")),
        "turnover_delta": _parse_number(row.get("即时成交额")),
        "volume_baseline": _parse_number(row.get("基线成交量")),
        "volume_ratio": _parse_number(row.get("放量倍数")),
        "alert": (row.get("异动") if isinstance(row.get("异动"), str) else None),
        "alert_level": int(_parse_number(row.get("异动级别")) or 0),
        "trend": trend,
        "summary": summary,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _find_spot_row(all_spot, symbol: str):
    """Locate a stock row by user-supplied symbol.

    Handles akshare 1.18.x where 代码 carries a market prefix (sh600519).
    If the user specifies an explicit exchange we match the exact full code;
    otherwise we match by the bare numeric code across all exchanges, so codes
    like 920000 (Beijing) resolve without guessing the exchange from the prefix.
    """
    sym = normalize_symbol(symbol)
    code = sym.split(".")[0]
    codes = all_spot["代码"].astype(str)

    raw = symbol.strip()
    has_explicit_market = (
        "." in raw or raw.lower().startswith(("sh", "sz", "bj"))
    )
    if has_explicit_market:
        full_code = _symbol_to_full_code(symbol)
        row = all_spot.loc[codes == full_code]
        if not row.empty:
            return row.iloc[0]

    # Fall back to matching the bare numeric code (strip any market prefix).
    bare = codes.str.replace(r"^(sh|sz|bj)", "", regex=True)
    row = all_spot.loc[bare == code]
    if not row.empty:
        return row.iloc[0]
    return None


def get_stock_trends(symbols: List[str]) -> List[dict]:
    all_spot = get_spot_data()
    results = []

    for symbol in symbols:
        row = _find_spot_row(all_spot, symbol)
        if row is None:
            raise ValueError(f"未找到股票：{symbol}")
        results.append(_build_stock_item(row))

    return results


# 前端排序字段 -> spot DataFrame 中文列名
SORT_COLUMN_MAP = {
    "code": "代码",
    "name": "名称",
    "current_price": "最新价",
    "change": "涨跌额",
    "change_percent": "涨跌幅",
    "high": "最高",
    "low": "最低",
    "volume": "成交量",
    "volume_delta": "即时成交量",
    "volume_ratio": "放量倍数",
    "turnover": "成交额",
}
# 这些字段按数值排序（其余按字符串）
NUMERIC_SORT_FIELDS = {"current_price", "change", "change_percent", "high", "low", "volume", "volume_delta", "volume_ratio", "turnover"}


def get_stock_list(
    page: int = 1,
    page_size: int = 50,
    market: str = "all",
    query: Optional[str] = None,
    sort_by: str = "code",
    order: str = "asc",
    min_pct: Optional[float] = None,
    max_pct: Optional[float] = None,
    trend: Optional[str] = None,
) -> dict:
    all_spot = get_spot_data()
    if market.lower() == "sh":
        all_spot = all_spot[all_spot["代码"].astype(str).str.startswith("sh")]
    elif market.lower() == "sz":
        all_spot = all_spot[all_spot["代码"].astype(str).str.startswith("sz")]
    elif market.lower() == "bj":
        all_spot = all_spot[all_spot["代码"].astype(str).str.startswith("bj")]

    if query:
        q = str(query).strip()
        code_mask = all_spot["代码"].astype(str).str.contains(q, case=False)
        name_mask = all_spot["名称"].astype(str).str.contains(q, case=False)
        all_spot = all_spot[code_mask | name_mask]

    # 涨跌幅范围筛选（单位 %）
    if min_pct is not None or max_pct is not None:
        pct = pd.to_numeric(all_spot["涨跌幅"], errors="coerce")
        mask = pd.Series(True, index=all_spot.index)
        if min_pct is not None:
            mask &= pct >= min_pct
        if max_pct is not None:
            mask &= pct <= max_pct
        all_spot = all_spot[mask]

    # 趋势筛选（精确匹配趋势列，与列表展示口径一致）
    if trend and trend != "all" and "趋势" in all_spot.columns:
        all_spot = all_spot[all_spot["趋势"].astype(str) == trend]

    total = int(all_spot.shape[0])

    # 排序
    col = SORT_COLUMN_MAP.get(sort_by, "代码")
    ascending = str(order or "asc").lower() != "desc"
    if col not in all_spot.columns:
        col = "代码"
        sort_by = "code"
    if sort_by in NUMERIC_SORT_FIELDS:
        sort_key = pd.to_numeric(all_spot[col], errors="coerce")
        all_spot = (
            all_spot.assign(_sort_key=sort_key)
            .sort_values(by="_sort_key", ascending=ascending, na_position="last")
            .drop(columns=["_sort_key"])
        )
    else:
        all_spot = all_spot.sort_values(by=col, ascending=ascending)

    start = (page - 1) * page_size
    end = start + page_size
    page_rows = all_spot.iloc[start:end]
    updated_at = None
    with _cache_lock:
        if _cache.get("spot_at"):
            updated_at = _cache["spot_at"].strftime("%Y-%m-%d %H:%M:%S")

    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "sort_by": sort_by,
        "order": "asc" if ascending else "desc",
        "stocks": [_build_stock_item(row) for _, row in page_rows.iterrows()],
        "updated_at": updated_at,
        "refresh_interval": CACHE_REFRESH_INTERVAL,
    }


def get_stock_history(symbol: str, count: int = 30) -> dict:
    normalized = normalize_symbol(symbol)
    history_symbol = normalize_history_symbol(symbol)
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=count * 3)).strftime("%Y%m%d")
    try:
        df = ak.stock_zh_a_daily(symbol=history_symbol, start_date=start_date, end_date=end_date)
    except Exception as exc:
        raise ValueError(f"历史行情数据获取失败：{exc}") from exc

    if df.empty:
        raise ValueError(f"未查询到历史数据：{normalized}")

    if "日期" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "日期"})
    if "日期" not in df.columns:
        raise ValueError(f"历史行情数据列名不匹配：{list(df.columns)}")

    df = df.sort_values(by="日期").tail(count)
    history = []
    for _, row in df.iterrows():
        history.append(
            {
                "date": str(row.get("日期"))[:10],
                "open": _parse_number(row.get("开盘")) or _parse_number(row.get("open")),
                "close": _parse_number(row.get("收盘")) or _parse_number(row.get("close")),
                "high": _parse_number(row.get("最高")) or _parse_number(row.get("high")),
                "low": _parse_number(row.get("最低")) or _parse_number(row.get("low")),
                "volume": _parse_number(row.get("成交量")) or _parse_number(row.get("volume")),
            }
        )

    stock_info = get_stock_trends([normalized])[0]
    return {
        "symbol": normalized,
        "name": stock_info["name"],
        "history": history,
    }


def get_index_snapshots() -> List[dict]:
    df = get_index_data()
    codes = {"sh000001": "上证指数", "sz399001": "深证成指", "sz399006": "创业板指"}
    if "指数代码" in df.columns:
        code_field = "指数代码"
    elif "代码" in df.columns:
        code_field = "代码"
    else:
        raise ValueError("无法识别指数数据字段")

    snapshots = []
    for index_code, name in codes.items():
        row = df.loc[df[code_field] == index_code]
        if row.empty:
            continue
        row = row.iloc[0]
        snapshots.append(
            {
                "code": index_code,
                "name": name,
                "price": _parse_number(row.get("最新价")),
                "change": _parse_number(row.get("涨跌额")),
                "change_percent": _parse_number(row.get("涨跌幅")),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    if not snapshots:
        raise ValueError("未查询到主要指数数据")
    return snapshots


# ============================================================
# 关注列表 + 即时成交量 + 大买/大卖异动检测
# ============================================================

def load_watchlist() -> List[str]:
    """读取关注列表，返回 akshare 全代码格式（如 sh600519）。"""
    with _watchlist_lock:
        try:
            with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [str(c) for c in data.get("codes", [])]
        except (FileNotFoundError, json.JSONDecodeError):
            return []


def _save_watchlist(codes: List[str]):
    with _watchlist_lock:
        with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
            json.dump({"codes": codes}, f, ensure_ascii=False, indent=2)


def add_to_watchlist(symbol: str) -> List[str]:
    full = _symbol_to_full_code(symbol)
    spot = get_spot_data()
    if spot is not None and not (spot["代码"].astype(str) == full).any():
        raise ValueError(f"未找到股票：{symbol}")
    with _watchlist_lock:
        codes = load_watchlist()
        if full not in codes:
            codes.append(full)
            _save_watchlist(codes)
        return codes


def remove_from_watchlist(symbol: str) -> List[str]:
    full = _symbol_to_full_code(symbol)
    with _watchlist_lock:
        codes = [c for c in load_watchlist() if c != full]
        _save_watchlist(codes)
    # 全市场指标历史一直维护，移除关注无需清理
    return codes


def _median(values) -> Optional[float]:
    vals = sorted(v for v in values if v is not None)
    n = len(vals)
    if n == 0:
        return None
    mid = n // 2
    if n % 2 == 1:
        return float(vals[mid])
    return (vals[mid - 1] + vals[mid]) / 2.0


def get_watchlist_data() -> dict:
    """返回关注列表实时数据。每次按当前 watchlist 文件对齐：
    高频缓存（8s 刷新、含即时成交量/异动）里有的直接用；刚加入、缓存还没轮询到的，
    立即用全市场快照补上；已删除的立即剔除。这样加/减关注后面板即时生效，不必等下一次轮询。"""
    codes = load_watchlist()
    want = [(_full_code_to_symbol(c), c) for c in codes]

    with _cache_lock:
        cached = _watch_cache.get("items")
        watch_at = _watch_cache.get("at")

    if cached is not None:
        by_symbol = {it["symbol"]: it for it in cached}
        interval = WATCH_POLL_INTERVAL
        updated = (watch_at or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    else:
        by_symbol = {}
        interval = CACHE_REFRESH_INTERVAL
        updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 缓存里缺的代码（刚加入还没轮询到），用全市场快照即时补全
    missing = [full for sym, full in want if sym not in by_symbol]
    if missing:
        for it in _watchlist_from_spot(missing):
            by_symbol[it["symbol"]] = it

    # 严格按当前关注列表顺序输出，已删除的自然被排除
    items = [by_symbol[sym] for sym, _ in want if sym in by_symbol]

    return {
        "stocks": items,
        "alerts": _extract_watch_alerts(items),
        "updated_at": updated,
        "refresh_interval": interval,
    }


def get_bid_ask(symbol: str) -> dict:
    """获取单只股票的买卖五档盘口（新浪实时行情）。
    新浪 hq.sinajs.cn 返回逗号分隔字段，含买一~买五、卖一~卖五的价与量(单位:股)。
    """
    full = _symbol_to_full_code(symbol)        # sh600519
    normalized = normalize_symbol(symbol)      # 600519.SH
    try:
        resp = requests.get(
            f"https://hq.sinajs.cn/list={full}",
            headers={"Referer": "https://finance.sina.com.cn"},
            timeout=8,
        )
        resp.encoding = "gbk"
    except Exception as exc:
        raise ValueError(f"五档行情获取失败：{exc}") from exc

    text = resp.text.strip()
    if '"' not in text:
        raise ValueError(f"未查询到五档数据：{normalized}")
    data = text.split('"')[1].split(",")
    if len(data) < 30 or not data[0]:
        raise ValueError(f"五档数据格式异常或股票停牌：{normalized}")

    def num(i):
        return _parse_number(data[i]) if i < len(data) else None

    # 买盘 买一~买五：量在偶数下标 10/12/14/16/18，价在其后 11/13/15/17/19
    bids = [{"level": i + 1, "price": num(11 + 2 * i), "volume": num(10 + 2 * i)} for i in range(5)]
    # 卖盘 卖一~卖五：量在 20/22/24/26/28，价在 21/23/25/27/29
    asks = [{"level": i + 1, "price": num(21 + 2 * i), "volume": num(20 + 2 * i)} for i in range(5)]

    return {
        "symbol": normalized,
        "name": data[0],
        "current_price": num(3),
        "previous_close": num(2),
        "open": num(1),
        "high": num(4),
        "low": num(5),
        "bids": bids,   # 买一(最高买价)→买五
        "asks": asks,   # 卖一(最低卖价)→卖五
        "timestamp": f"{data[30]} {data[31]}" if len(data) > 31 else "",
    }


# ============================================================
# 关注列表高频刷新（独立于全市场，用新浪多代码接口，几只一次请求很快）
# ============================================================

def _fetch_sina_quotes(codes: List[str]) -> dict:
    """批量拉取关注列表的新浪实时行情，返回 {full_code: 逗号分隔字段列表}。"""
    if not codes:
        return {}
    resp = requests.get(
        f"https://hq.sinajs.cn/list={','.join(codes)}",
        headers={"Referer": "https://finance.sina.com.cn"},
        timeout=6,
    )
    resp.encoding = "gbk"
    out = {}
    for line in resp.text.strip().splitlines():
        if "hq_str_" not in line or '"' not in line:
            continue
        code = line.split("=")[0].split("hq_str_")[-1].strip()
        payload = line.split('"', 1)[1].rsplit('"', 1)[0]
        parts = payload.split(",")
        if len(parts) >= 32 and parts[0]:
            out[code] = parts
    return out


def _build_watch_item(full_code, parts, vol_delta, baseline, ratio, alert, level) -> dict:
    """从新浪实时字段构造关注列表条目（含即时成交量/异动）。"""
    price = _parse_number(parts[3])
    prev_close = _parse_number(parts[2])
    change = (price - prev_close) if (price is not None and prev_close) else None
    change_pct = (change / prev_close * 100) if (change is not None and prev_close) else None
    return {
        "symbol": _full_code_to_symbol(full_code),
        "name": parts[0],
        "current_price": price,
        "previous_close": prev_close,
        "change": round(change, 2) if change is not None else 0.0,
        "change_percent": round(change_pct, 2) if change_pct is not None else 0.0,
        "high": _parse_number(parts[4]),
        "low": _parse_number(parts[5]),
        "open": _parse_number(parts[1]),
        "volume": _parse_number(parts[8]),
        "turnover": _parse_number(parts[9]),
        "volume_delta": vol_delta,
        "turnover_delta": None,
        "volume_baseline": baseline,
        "volume_ratio": ratio,
        "alert": alert,
        "alert_level": level,
        "trend": describe_trend((change_pct or 0) / 100.0, price, prev_close),
        "summary": None,
        "timestamp": f"{parts[30]} {parts[31]}" if len(parts) > 31 else datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _poll_watchlist_once():
    """关注列表高频刷新一次：拉新浪实时行情，用【独立状态】算即时成交量与异动。"""
    codes = load_watchlist()
    if not codes:
        with _cache_lock:
            _watch_cache["items"] = []
            _watch_cache["at"] = datetime.now()
        return
    quotes = _fetch_sina_quotes(codes)
    items = []
    for code in codes:
        parts = quotes.get(code)
        if not parts:
            continue
        cur_vol = _parse_number(parts[8]) or 0.0
        cur_price = _parse_number(parts[3])
        prev_vol = _watch_prev_volume.get(code)
        prev_price = _watch_prev_price.get(code)

        vol_delta = None
        price_delta = None
        if prev_vol is not None:
            vol_delta = max(cur_vol - prev_vol, 0.0)
        if prev_price is not None and cur_price is not None:
            price_delta = cur_price - prev_price

        baseline = ratio = None
        alert = None
        level = 0
        if vol_delta is not None:
            hist = _watch_delta_history.setdefault(code, deque(maxlen=VOLUME_HISTORY_LEN))
            if len(hist) >= MIN_BASELINE_SAMPLES:
                baseline = _median(hist)
                if baseline and baseline > 0:
                    ratio = vol_delta / baseline
                    if vol_delta > 0:
                        alert, level = _classify_alert(ratio, price_delta)
            hist.append(vol_delta)

        _watch_prev_volume[code] = cur_vol
        if cur_price is not None:
            _watch_prev_price[code] = cur_price

        items.append(_build_watch_item(
            code, parts, vol_delta, baseline,
            round(ratio, 2) if ratio is not None else None, alert, level))

    with _cache_lock:
        _watch_cache["items"] = items
        _watch_cache["at"] = datetime.now()


def start_watchlist_polling(interval_seconds: int = WATCH_POLL_INTERVAL):
    """启动关注列表高频刷新后台线程（独立于全市场刷新）。"""
    def _loop():
        while True:
            try:
                _poll_watchlist_once()
            except Exception as exc:
                print(f"Watchlist poll failed: {exc}")
            time.sleep(max(2, interval_seconds))

    Thread(target=_loop, daemon=True).start()


def _watchlist_from_spot(codes: Optional[List[str]] = None) -> List[dict]:
    """从全市场快照取指定代码的条目（codes 为空则取整个关注列表）。
    用于高频缓存还没数据、或刚加入的股票还没轮询到时的即时补全。"""
    if codes is None:
        codes = load_watchlist()
    spot = get_spot_data()
    code_col = spot["代码"].astype(str) if spot is not None else None
    if code_col is None:
        return []
    items = []
    for full_code in codes:
        row = spot.loc[code_col == full_code]
        if not row.empty:
            items.append(_build_stock_item(row.iloc[0]))
    return items


def _extract_watch_alerts(items: List[dict]) -> List[dict]:
    alerts = []
    for s in items:
        if s.get("alert"):
            alerts.append({
                "symbol": s["symbol"],
                "name": s["name"],
                "alert": s["alert"],
                "alert_level": s.get("alert_level", 0),
                "volume_delta": s.get("volume_delta"),
                "turnover_delta": s.get("turnover_delta"),
                "ratio": s.get("volume_ratio"),
                "change_percent": s["change_percent"],
                "current_price": s["current_price"],
            })
    return alerts


def get_stock_intraday(symbol: str, scale: int = 1) -> dict:
    """获取分时行情（新浪 JSONP 分钟级 K 线）。
    scale=1 为 1 分钟，scale=5 为 5 分钟，最大 60。
    datalen=241 覆盖一个完整交易日（09:30-11:30 + 13:00-15:00 共 240 分钟）。
    """
    full = _symbol_to_full_code(symbol)   # sh600519
    url = (
        "https://quotes.sina.cn/cn/api/jsonp_v2.php/var_d"
        "/CN_MarketDataService.getKLineData"
        f"?symbol={full}&scale={scale}&ma=no&datalen=241"
    )
    try:
        resp = requests.get(
            url,
            headers={
                "Referer": "https://finance.sina.com.cn",
                "User-Agent": "Mozilla/5.0",
            },
            timeout=10,
        )
        resp.encoding = "utf-8"
    except Exception as exc:
        raise ValueError(f"分时行情获取失败：{exc}") from exc

    text = resp.text.strip()
    if not text:
        raise ValueError(f"未获取到分时数据（非交易时段或代码错误）：{symbol}")

    # 剥离 JSONP 包装：/*...*/ \n var_d([...]) 或 var_d(null)
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if not m:
        raise ValueError(f"分时数据为空（可能非交易时段）：{symbol}")

    try:
        data = json.loads(m.group())
    except json.JSONDecodeError as exc:
        raise ValueError(f"分时数据格式异常：{exc}") from exc

    bars = []
    for item in data:
        if not isinstance(item, dict):
            continue
        bars.append({
            "time": item.get("day", ""),
            "open": _parse_number(item.get("open")),
            "high": _parse_number(item.get("high")),
            "low": _parse_number(item.get("low")),
            "close": _parse_number(item.get("close")),
            "volume": _parse_number(item.get("volume")),
        })

    # 取当日昨收（用于分时图画昨收参考线）
    prev_close = None
    with _cache_lock:
        spot = _cache["spot_df"]
    if spot is not None:
        code_col = spot["代码"].astype(str)
        row = spot.loc[code_col == full]
        if not row.empty:
            prev_close = _parse_number(row.iloc[0].get("昨收"))

    return {
        "symbol": normalize_symbol(symbol),
        "scale": scale,
        "prev_close": prev_close,
        "bars": bars,
    }
