"""技术指标计算模块。

从日 K 线 OHLCV 序列计算 A 股短线常用指标：MA、MACD、KDJ、RSI、BOLL、量价关系，
并按 `docs/trading-knowledge.md` 中的规则给出综合多空打分与信号摘要。

纯 Python 实现（不依赖 numpy/pandas），便于独立测试与复用。所有函数对数据不足、
含 None 的情况做了容错，缺数据时返回 None 而非抛异常。

输入：history 为按日期升序排列的列表，每项 dict 含 open/high/low/close/volume。
输出：summarize_indicators() 返回结构化指标 + signals(信号列表) + score(多空分) + text(摘要)。
"""
from typing import List, Optional, Dict


def _closes(history: List[dict]) -> List[float]:
    return [h["close"] for h in history if h.get("close") is not None]


def _ema(values: List[float], period: int) -> List[Optional[float]]:
    """指数移动平均，返回与 values 等长的序列（前期用累计均值预热）。"""
    if not values:
        return []
    k = 2 / (period + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def _sma_wilder(values: List[float], period: int) -> List[float]:
    """Wilder 平滑（SMMA，权重1），通达信 RSI 所用。"""
    if not values:
        return []
    out = [values[0]]
    for v in values[1:]:
        out.append((out[-1] * (period - 1) + v) / period)
    return out


def moving_averages(history: List[dict]) -> Dict[str, Optional[float]]:
    closes = _closes(history)
    res = {}
    for n in (5, 10, 20, 60):
        res[f"ma{n}"] = round(sum(closes[-n:]) / n, 3) if len(closes) >= n else None
    return res


def ma_signal(history: List[dict]) -> Dict[str, object]:
    closes = _closes(history)
    ma = moving_averages(history)
    out = {"arrangement": None, "above_ma20": None, "cross": None, **ma}
    if len(closes) < 20:
        return out
    ma5, ma10, ma20 = ma["ma5"], ma["ma10"], ma["ma20"]
    ma60 = ma["ma60"]
    if None not in (ma5, ma10, ma20):
        longs = [ma5, ma10, ma20] + ([ma60] if ma60 is not None else [])
        if all(longs[i] > longs[i + 1] for i in range(len(longs) - 1)):
            out["arrangement"] = "bull"   # 多头排列
        elif all(longs[i] < longs[i + 1] for i in range(len(longs) - 1)):
            out["arrangement"] = "bear"   # 空头排列
        else:
            out["arrangement"] = "mixed"
        out["above_ma20"] = closes[-1] > ma20
    # MA5/MA10 金叉死叉（比较今日与昨日的相对位置）
    if len(closes) >= 11:
        prev5 = sum(closes[-6:-1]) / 5
        prev10 = sum(closes[-11:-1]) / 10
        if prev5 <= prev10 and ma5 > ma10:
            out["cross"] = "golden"
        elif prev5 >= prev10 and ma5 < ma10:
            out["cross"] = "dead"
    return out


def macd(history: List[dict]) -> Dict[str, object]:
    closes = _closes(history)
    out = {"dif": None, "dea": None, "hist": None, "cross": None, "above_zero": None}
    if len(closes) < 26:
        return out
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    dif = [a - b for a, b in zip(ema12, ema26)]
    dea = _ema(dif, 9)
    hist = [2 * (d - e) for d, e in zip(dif, dea)]
    out["dif"] = round(dif[-1], 4)
    out["dea"] = round(dea[-1], 4)
    out["hist"] = round(hist[-1], 4)
    out["above_zero"] = dif[-1] > 0
    if len(hist) >= 2:
        if hist[-2] <= 0 < hist[-1]:
            out["cross"] = "golden"
        elif hist[-2] >= 0 > hist[-1]:
            out["cross"] = "dead"
        elif hist[-1] > hist[-2]:
            out["cross"] = "hist_up"    # 柱放大（动能增强）
        elif hist[-1] < hist[-2]:
            out["cross"] = "hist_down"
    return out


def kdj(history: List[dict], n: int = 9) -> Dict[str, object]:
    rows = [h for h in history if None not in (h.get("high"), h.get("low"), h.get("close"))]
    out = {"k": None, "d": None, "j": None, "cross": None, "state": None}
    if len(rows) < n:
        return out
    k, d = 50.0, 50.0
    ks, ds = [], []
    for i in range(n - 1, len(rows)):
        window = rows[i - n + 1:i + 1]
        low = min(r["low"] for r in window)
        high = max(r["high"] for r in window)
        close = rows[i]["close"]
        rsv = 0.0 if high == low else (close - low) / (high - low) * 100
        k = 2 / 3 * k + 1 / 3 * rsv
        d = 2 / 3 * d + 1 / 3 * k
        ks.append(k)
        ds.append(d)
    j = 3 * ks[-1] - 2 * ds[-1]
    out["k"], out["d"], out["j"] = round(ks[-1], 2), round(ds[-1], 2), round(j, 2)
    if ks[-1] > 80 or j > 100:
        out["state"] = "overbought"
    elif ks[-1] < 20 or j < 0:
        out["state"] = "oversold"
    if len(ks) >= 2:
        if ks[-2] <= ds[-2] and ks[-1] > ds[-1]:
            out["cross"] = "golden"
        elif ks[-2] >= ds[-2] and ks[-1] < ds[-1]:
            out["cross"] = "dead"
    return out


def rsi(history: List[dict], periods=(6, 12, 24)) -> Dict[str, object]:
    closes = _closes(history)
    out = {f"rsi{p}": None for p in periods}
    out["state"] = None
    if len(closes) < max(periods) + 1:
        return out
    diffs = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0) for d in diffs]
    absd = [abs(d) for d in diffs]
    for p in periods:
        up = _sma_wilder(gains, p)
        tot = _sma_wilder(absd, p)
        out[f"rsi{p}"] = round(100 * up[-1] / tot[-1], 2) if tot[-1] else 50.0
    r6 = out.get("rsi6")
    if r6 is not None:
        if r6 > 80:
            out["state"] = "overbought"
        elif r6 < 20:
            out["state"] = "oversold"
    return out


def boll(history: List[dict], n: int = 20, k: float = 2.0) -> Dict[str, object]:
    closes = _closes(history)
    out = {"mid": None, "upper": None, "lower": None, "position": None, "bandwidth": None}
    if len(closes) < n:
        return out
    window = closes[-n:]
    mid = sum(window) / n
    var = sum((c - mid) ** 2 for c in window) / n
    std = var ** 0.5
    upper, lower = mid + k * std, mid - k * std
    out["mid"], out["upper"], out["lower"] = round(mid, 3), round(upper, 3), round(lower, 3)
    out["bandwidth"] = round((upper - lower) / mid * 100, 2) if mid else None
    last = closes[-1]
    if last >= upper:
        out["position"] = "above_upper"
    elif last <= lower:
        out["position"] = "below_lower"
    elif last >= mid:
        out["position"] = "upper_half"
    else:
        out["position"] = "lower_half"
    return out


def volume_price(history: List[dict]) -> Dict[str, object]:
    """量价关系：最近一日量能 vs 近5日均量，结合价格方向判定模式。"""
    rows = [h for h in history if h.get("volume") is not None and h.get("close") is not None]
    out = {"vol_ratio": None, "pattern": None}
    if len(rows) < 6:
        return out
    last = rows[-1]
    avg5 = sum(r["volume"] for r in rows[-6:-1]) / 5
    if avg5 <= 0:
        return out
    ratio = last["volume"] / avg5
    out["vol_ratio"] = round(ratio, 2)
    price_up = last["close"] >= rows[-2]["close"]
    enlarged = ratio >= 1.5
    shrunk = ratio <= 0.7
    if enlarged and price_up:
        out["pattern"] = "量增价涨"
    elif enlarged and not price_up:
        out["pattern"] = "量增价跌"
    elif shrunk and price_up:
        out["pattern"] = "量缩价涨"
    elif shrunk and not price_up:
        out["pattern"] = "量缩价跌"
    else:
        out["pattern"] = "量价平稳"
    return out


# ============================================================
# 综合：多空打分 + 信号摘要（依据 trading-knowledge.md 第八节）
# ============================================================

def summarize_indicators(history: List[dict]) -> Optional[Dict[str, object]]:
    if not history or len(_closes(history)) < 20:
        return None
    m = ma_signal(history)
    mac = macd(history)
    k = kdj(history)
    r = rsi(history)
    b = boll(history)
    vp = volume_price(history)

    score = 0
    signals = []

    # 1) 均线
    if m["arrangement"] == "bull":
        score += 1; signals.append("均线多头排列")
    elif m["arrangement"] == "bear":
        score -= 1; signals.append("均线空头排列")
    if m["cross"] == "golden":
        score += 1; signals.append("MA5上穿MA10(金叉)")
    elif m["cross"] == "dead":
        score -= 1; signals.append("MA5下穿MA10(死叉)")

    # 2) MACD
    if mac["cross"] == "golden":
        score += 1; signals.append("MACD金叉" + ("(零轴上)" if mac["above_zero"] else ""))
    elif mac["cross"] == "dead":
        score -= 1; signals.append("MACD死叉")
    elif mac["cross"] == "hist_up":
        score += 1; signals.append("MACD红柱放大")
    elif mac["cross"] == "hist_down":
        score -= 1; signals.append("MACD绿柱放大")

    # 3) KDJ
    if k["cross"] == "golden" and (k["k"] or 50) < 50:
        score += 1; signals.append("KDJ低位金叉")
    elif k["cross"] == "golden":
        score += 1; signals.append("KDJ金叉")
    elif k["cross"] == "dead":
        score -= 1; signals.append("KDJ死叉")
    if k["state"] == "overbought":
        score -= 1; signals.append("KDJ超买")
    elif k["state"] == "oversold":
        score += 1; signals.append("KDJ超卖")

    # 4) RSI
    if r["state"] == "overbought":
        score -= 1; signals.append("RSI超买")
    elif r["state"] == "oversold":
        score += 1; signals.append("RSI超卖")

    # 5) BOLL
    if b["position"] == "below_lower":
        score += 1; signals.append("跌破布林下轨(超跌)")
    elif b["position"] == "above_upper":
        score -= 1; signals.append("突破布林上轨(超买)")

    # 6) 量价
    if vp["pattern"] == "量增价涨":
        score += 1; signals.append("量增价涨")
    elif vp["pattern"] in ("量增价跌", "量缩价涨"):
        score -= 1; signals.append(vp["pattern"] + "(量价背离)")

    if score >= 2:
        bias = "偏多"
    elif score <= -2:
        bias = "偏空"
    else:
        bias = "中性"

    text = f"技术面{bias}(评分{score:+d})：" + "、".join(signals) if signals else f"技术面{bias}(评分{score:+d})"

    return {
        "ma": m, "macd": mac, "kdj": k, "rsi": r, "boll": b, "volume_price": vp,
        "score": score, "bias": bias, "signals": signals, "text": text,
    }
