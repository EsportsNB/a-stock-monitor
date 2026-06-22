"""AI 股票涨跌预测模块。

流程：每个交易日收盘后，从全市场选出候选股票（放量异动 + 涨幅榜 + 成交活跃榜），
用智谱 GLM-4-Flash（免费、支持联网搜索）结合量价技术面 + 联网搜到的公司新闻/公告，
预测下一交易日涨跌，挑出预测涨幅高的作为推荐；次日收盘后自动复盘对错并滚动统计准确率。

依赖环境变量 ZHIPU_API_KEY（去 https://open.bigmodel.cn 免费注册获取）。
可选 ZHIPU_MODEL 覆盖模型名（默认 glm-4-flash）。
"""
import os
import re
import json
import time
import requests
from datetime import datetime
from typing import List, Optional

ZHIPU_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
ZHIPU_MODEL = os.environ.get("ZHIPU_MODEL", "glm-4-flash")

PRED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "predictions")
SUMMARY_PATH = os.path.join(PRED_DIR, "SUMMARY.md")

BATCH_SIZE = 10          # 每次喂给模型的股票数（控制单次 token）
CANDIDATE_LIMIT = 50     # 候选池上限
FLAT_THRESHOLD = 0.5     # 实际涨跌幅在 ±该值(%) 内视为“平”


# ============================================================
# 智谱 GLM 客户端（直接 HTTP，避免额外 SDK 依赖）
# ============================================================

def zhipu_available() -> bool:
    return bool(os.environ.get("ZHIPU_API_KEY"))


def _call_glm(messages, use_web_search=True, temperature=0.3, timeout=120) -> str:
    api_key = os.environ.get("ZHIPU_API_KEY")
    if not api_key:
        raise ValueError("未配置 ZHIPU_API_KEY 环境变量，无法调用 AI 模型。请到 https://open.bigmodel.cn 免费获取后设置。")
    payload = {"model": ZHIPU_MODEL, "messages": messages, "temperature": temperature}
    if use_web_search:
        payload["tools"] = [{
            "type": "web_search",
            "web_search": {"enable": True, "search_engine": "search_std", "search_result": True},
        }]
    resp = requests.post(
        ZHIPU_ENDPOINT,
        json=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ============================================================
# 候选池：放量异动 + 涨幅榜 + 成交活跃榜 + 放量榜，去重
# ============================================================

def select_candidates(limit: int = CANDIDATE_LIMIT) -> List[dict]:
    import pandas as pd
    from .stock_monitor import get_spot_data, _build_stock_item

    spot = get_spot_data()
    if spot is None or spot.empty:
        return []
    df = spot
    code_col = df["代码"].astype(str)
    pct = pd.to_numeric(df.get("涨跌幅"), errors="coerce")
    amount = pd.to_numeric(df.get("成交额"), errors="coerce")
    ratio = pd.to_numeric(df.get("放量倍数"), errors="coerce") if "放量倍数" in df.columns else None

    picked = []
    # 1) 放量异动票（最优先）
    if "异动" in df.columns:
        picked += code_col[df["异动"].notna()].tolist()
    # 2) 涨幅榜 Top20
    picked += code_col[pct.sort_values(ascending=False).index[:20]].tolist()
    # 3) 成交额活跃榜 Top20
    picked += code_col[amount.sort_values(ascending=False).index[:20]].tolist()
    # 4) 放量倍数榜 Top15
    if ratio is not None:
        picked += code_col[ratio.sort_values(ascending=False, na_position="last").index[:15]].tolist()

    ordered = list(dict.fromkeys(picked))[:limit]   # 去重保序
    items = []
    for code in ordered:
        row = df.loc[code_col == code]
        if not row.empty:
            items.append(_build_stock_item(row.iloc[0]))
    return items


# ============================================================
# 预测：分批调用 GLM
# ============================================================

def _build_messages(batch: List[dict]):
    lines = []
    for s in batch:
        lines.append(
            f"- {s['symbol']} {s['name']}：现价{s['current_price']} 涨跌幅{s['change_percent']}% "
            f"今开{s.get('open')} 最高{s.get('high')} 最低{s.get('low')} 成交量{s.get('volume')} "
            f"放量倍数{s.get('volume_ratio')} 异动{s.get('alert')} 趋势{s.get('trend')}"
        )
    stock_block = "\n".join(lines)
    system = (
        "你是严谨的A股短线分析助手。请结合每只股票的量价技术面，并联网搜索该公司最近一周的新闻、"
        "公告、行业政策与资金动向，综合预测【下一个交易日】的涨跌。"
        "务必只输出一个JSON数组，禁止输出任何多余文字或解释。数组每个元素格式："
        '{"symbol":"股票代码","direction":"up或down或flat","expected_pct":预测涨跌幅(数字,如3.5或-2.0),'
        '"confidence":0到1之间的置信度,"reason":"30字内理由，含搜到的关键消息"}'
    )
    user = f"请分析以下股票并预测下一交易日涨跌，只返回JSON数组：\n{stock_block}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_predictions(text: str) -> List[dict]:
    if not text:
        return []
    m = re.search(r"\[.*\]", text.strip(), re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
        return [p for p in arr if isinstance(p, dict)]
    except Exception:
        return []


def predict_candidates(candidates: List[dict], use_web_search: bool = True) -> List[dict]:
    predictions = []
    for i in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[i:i + BATCH_SIZE]
        try:
            text = _call_glm(_build_messages(batch), use_web_search=use_web_search)
            preds = _parse_predictions(text)
        except Exception as exc:
            print(f"AI 预测批次失败（{i}）：{exc}")
            preds = []
        by_symbol = {str(p.get("symbol", "")).strip(): p for p in preds}
        for s in batch:
            p = by_symbol.get(s["symbol"], {})
            exp = p.get("expected_pct")
            predictions.append({
                "symbol": s["symbol"],
                "name": s["name"],
                "ref_price": s["current_price"],
                "ref_change_percent": s["change_percent"],
                "direction": p.get("direction"),
                "expected_pct": exp if isinstance(exp, (int, float)) else None,
                "confidence": p.get("confidence"),
                "reason": p.get("reason"),
            })
        time.sleep(1)   # 轻微限速，避免触发免费额度的 RPM 限制
    return predictions


# ============================================================
# 日志：每日预测 JSON + 可读 Markdown
# ============================================================

def _ensure_dir():
    os.makedirs(PRED_DIR, exist_ok=True)


def _pred_path(date_str: str) -> str:
    return os.path.join(PRED_DIR, f"{date_str}.json")


def _md_path(date_str: str) -> str:
    return os.path.join(PRED_DIR, f"{date_str}.md")


def load_prediction(date_str: str) -> Optional[dict]:
    try:
        with open(_pred_path(date_str), encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _save_prediction(date_str: str, record: dict):
    _ensure_dir()
    with open(_pred_path(date_str), "w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    _write_markdown(date_str, record)


def _fmt(v, suffix=""):
    return "--" if v is None else f"{v}{suffix}"


def _write_markdown(date_str: str, record: dict):
    lines = [
        f"# {date_str} AI 涨跌预测",
        "",
        f"- 生成时间：{record.get('generated_at')}",
        f"- 模型：{record.get('model')} ｜ 联网搜索：{'是' if record.get('web_search') else '否'}",
        f"- 候选股票数：{record.get('candidate_count')}",
    ]
    if record.get("reviewed"):
        lines.append(f"- **复盘**：{record.get('review_at')} ｜ 方向命中 {record.get('hit_count')}/{record.get('pred_count')} ｜ 准确率 **{_fmt_pct(record.get('accuracy'))}**")
    recs = record.get("recommendations", [])
    if recs:
        lines += ["", f"## 🔮 推荐（预测涨幅 Top {len(recs)}）", ""]
        by_sym = {p["symbol"]: p for p in record["predictions"]}
        lines.append("| 代码 | 名称 | 预测涨幅 | 置信度 | 理由 |" + ("  实际 | 命中 |" if record.get("reviewed") else ""))
        lines.append("|------|------|---------|--------|------|" + ("------|------|" if record.get("reviewed") else ""))
        for sym in recs:
            p = by_sym.get(sym, {})
            row = f"| {sym} | {p.get('name','')} | {_fmt(p.get('expected_pct'),'%')} | {_fmt(p.get('confidence'))} | {p.get('reason','')} |"
            if record.get("reviewed"):
                hit = p.get("hit")
                row += f" {_fmt(p.get('actual_pct'),'%')} | {'✅' if hit else '❌' if hit is not None else '—'} |"
            lines.append(row)
    # 全部预测明细
    lines += ["", "## 全部候选预测", ""]
    header = "| 代码 | 名称 | 方向 | 预测涨幅 | 置信度 | 理由 |"
    sep = "|------|------|------|---------|--------|------|"
    if record.get("reviewed"):
        header += " 实际涨幅 | 命中 |"
        sep += "---------|------|"
    lines += [header, sep]
    for p in record["predictions"]:
        row = f"| {p['symbol']} | {p.get('name','')} | {_fmt(p.get('direction'))} | {_fmt(p.get('expected_pct'),'%')} | {_fmt(p.get('confidence'))} | {p.get('reason','')} |"
        if record.get("reviewed"):
            hit = p.get("hit")
            row += f" {_fmt(p.get('actual_pct'),'%')} | {'✅' if hit else '❌' if hit is not None else '—'} |"
        lines.append(row)
    with open(_md_path(date_str), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def _fmt_pct(v):
    return "--" if v is None else f"{round(v * 100, 1)}%"


# ============================================================
# 主流程：当日预测
# ============================================================

def run_daily_prediction(date_str: Optional[str] = None, use_web_search: bool = True, top_n: int = 10) -> dict:
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    candidates = select_candidates()
    if not candidates:
        raise ValueError("无法获取候选股票（行情未就绪），请确认服务已加载行情数据。")
    predictions = predict_candidates(candidates, use_web_search=use_web_search)

    ups = [p for p in predictions if p.get("direction") == "up" and isinstance(p.get("expected_pct"), (int, float))]
    ups.sort(key=lambda x: x["expected_pct"], reverse=True)
    recommendations = [p["symbol"] for p in ups[:top_n]]

    record = {
        "date": date_str,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": ZHIPU_MODEL,
        "web_search": use_web_search,
        "candidate_count": len(candidates),
        "predictions": predictions,
        "recommendations": recommendations,
        "reviewed": False,
    }
    _save_prediction(date_str, record)
    return record


# ============================================================
# 复盘：用次日实际涨跌对比预测，判定对错
# ============================================================

def review_prediction(date_str: str) -> dict:
    """对 date_str 这天的预测做复盘。应在【次日收盘后】调用：此时实时行情的涨跌幅
    正是预测目标日的实际涨跌幅。判定方向命中并回写日志、刷新总结。"""
    from .stock_monitor import get_spot_data, _build_stock_item, _symbol_to_full_code

    record = load_prediction(date_str)
    if not record:
        raise ValueError(f"未找到 {date_str} 的预测记录")
    spot = get_spot_data()
    if spot is None or spot.empty:
        raise ValueError("行情未就绪，无法复盘")
    code_col = spot["代码"].astype(str)

    correct = 0
    total = 0
    for p in record["predictions"]:
        full = _symbol_to_full_code(p["symbol"])
        row = spot.loc[code_col == full]
        if row.empty:
            p["hit"] = None
            continue
        actual = _build_stock_item(row.iloc[0])["change_percent"]
        p["actual_pct"] = actual
        actual_dir = "up" if actual > FLAT_THRESHOLD else ("down" if actual < -FLAT_THRESHOLD else "flat")
        p["actual_direction"] = actual_dir
        pred_dir = p.get("direction")
        if pred_dir in ("up", "down", "flat"):
            total += 1
            hit = pred_dir == actual_dir
            p["hit"] = hit
            if hit:
                correct += 1
        else:
            p["hit"] = None

    record["reviewed"] = True
    record["review_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record["pred_count"] = total
    record["hit_count"] = correct
    record["accuracy"] = round(correct / total, 4) if total else None
    _save_prediction(date_str, record)
    update_summary()
    return record


# ============================================================
# 滚动总结：累计准确率 + 易/难预测股票
# ============================================================

def update_summary() -> dict:
    _ensure_dir()
    files = sorted(f for f in os.listdir(PRED_DIR) if re.fullmatch(r"\d{4}-\d{2}-\d{2}\.json", f))
    daily = []
    per_stock = {}
    grand_total = 0
    grand_hit = 0
    for fn in files:
        try:
            with open(os.path.join(PRED_DIR, fn), encoding="utf-8") as f:
                rec = json.load(f)
        except Exception:
            continue
        if not rec.get("reviewed"):
            continue
        daily.append((rec["date"], rec.get("hit_count", 0), rec.get("pred_count", 0), rec.get("accuracy")))
        grand_total += rec.get("pred_count", 0)
        grand_hit += rec.get("hit_count", 0)
        for p in rec["predictions"]:
            if p.get("hit") is None:
                continue
            st = per_stock.setdefault(p["symbol"], {"name": p.get("name", ""), "total": 0, "hit": 0})
            st["total"] += 1
            if p["hit"]:
                st["hit"] += 1

    ranked = [
        (sym, d["name"], d["hit"], d["total"], d["hit"] / d["total"])
        for sym, d in per_stock.items() if d["total"] >= 2
    ]
    easy = sorted(ranked, key=lambda x: (-x[4], -x[3]))[:15]
    hard = sorted(ranked, key=lambda x: (x[4], -x[3]))[:15]
    overall = round(grand_hit / grand_total, 4) if grand_total else None

    lines = [
        "# AI 预测准确率总结",
        "",
        f"最后更新：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"累计已复盘预测：方向命中 **{grand_hit}/{grand_total}**，总准确率 **{_fmt_pct(overall)}**",
        f"已复盘交易日：{len(daily)} 天",
        "",
        "## 每日准确率",
        "",
        "| 日期 | 命中/总数 | 准确率 |",
        "|------|----------|--------|",
    ]
    for date_str, hit, tot, acc in daily:
        lines.append(f"| {date_str} | {hit}/{tot} | {_fmt_pct(acc)} |")

    lines += ["", "## ✅ 容易预测的股票（命中率高，预测≥2次）", "",
              "| 代码 | 名称 | 命中/次数 | 命中率 |", "|------|------|----------|--------|"]
    for sym, name, hit, tot, rate in easy:
        lines.append(f"| {sym} | {name} | {hit}/{tot} | {_fmt_pct(rate)} |")

    lines += ["", "## ⚠️ 难预测的股票（命中率低，预测≥2次）", "",
              "| 代码 | 名称 | 命中/次数 | 命中率 |", "|------|------|----------|--------|"]
    for sym, name, hit, tot, rate in hard:
        lines.append(f"| {sym} | {name} | {hit}/{tot} | {_fmt_pct(rate)} |")

    with open(SUMMARY_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return {"overall_accuracy": overall, "reviewed_days": len(daily), "total_predictions": grand_total}


def review_pending() -> Optional[dict]:
    """复盘最近一个尚未复盘的预测（其预测目标日应为今天收盘）。
    供调度在每个交易日收盘后调用：自动找到上一个交易日的预测，用今天的实际涨跌复盘。
    返回被复盘的记录，没有可复盘的则返回 None。"""
    if not os.path.isdir(PRED_DIR):
        return None
    today = datetime.now().strftime("%Y-%m-%d")
    files = sorted(f for f in os.listdir(PRED_DIR) if re.fullmatch(r"\d{4}-\d{2}-\d{2}\.json", f))
    for fn in reversed(files):
        date_str = fn[:-5]
        if date_str >= today:
            continue
        rec = load_prediction(date_str)
        if rec and not rec.get("reviewed"):
            return review_prediction(date_str)
    return None
