# A股实时监控服务 — 项目上下文

## 项目概况

FastAPI + uvicorn 的 A 股实时监控 Web 服务，运行在 `http://localhost:8000`。

功能：全市场实时行情（沪深北）、即时成交量 / 大户异动检测、关注列表、五档盘口、日 K 线 + 当日分时图、AI 次日涨跌预测。

## 启动与部署

```powershell
# 前台启动（开发调试）
cd C:\Project\a-stock-monitor
py -m uvicorn app.main:app --port 8000 --reload

# 后台无窗口启动（生产，开机自启同路径）
Start-Process pythonw -ArgumentList "run_server.py" -WorkingDirectory "C:\Project\a-stock-monitor" -WindowStyle Hidden

# 停止服务
Stop-Process -Name python, pythonw -Force -ErrorAction SilentlyContinue
```

开机自启：注册表 `HKCU\Software\Microsoft\Windows\CurrentVersion\Run` + 启动文件夹，均指向 `pythonw.exe run_server.py`。

## 数据源

- **实时行情**：新浪 `hq.sinajs.cn`（GBK 编码，需 Referer 头），akshare 1.18.64
- **分时 K 线**：新浪 JSONP `quotes.sina.cn/cn/api/jsonp_v2.php/CN_MarketDataService.getKLineData`
- **东方财富**：此机器完全不可达（RemoteDisconnected），所有行情走新浪
- **代理**：Clash 127.0.0.1:7890，启动时 `disable_proxy_for_china_data()` 清除代理环境变量，直连国内源

## 关键文件

| 文件 | 说明 |
|------|------|
| `app/stock_monitor.py` | 核心：行情拉取、缓存、异动检测、关注列表、分时接口 |
| `app/main.py` | FastAPI 路由 |
| `app/static/index.html` | 单页前端（Chart.js，无构建工具） |
| `app/ai_predictor.py` | AI 预测模块（智谱 GLM-4-Flash）；预测前用 indicators 预计算技术面喂模型并参与排序 |
| `app/indicators.py` | 技术指标计算：MA/MACD/KDJ/RSI/BOLL/量价 + 综合多空评分（纯 Python） |
| `docs/trading-knowledge.md` | 短线技术分析知识库，indicators 与预测提示词的规则依据 |
| `run_server.py` | 无窗口启动入口，重定向 stdout/stderr 到 logs/service.log |
| `predict_daily.ps1` | 每日 15:30 调度脚本（复盘 + 预测） |
| `watchlist.json` | 关注列表持久化（gitignore） |
| `predictions/` | AI 预测日志（gitignore） |

## 架构要点

- **全市场缓存**：30s 刷新一次，`_cache_lock`（RLock）只在写入时持锁，网络请求在锁外
- **关注列表高频通道**：独立 8s 轮询，独立状态（`_watch_prev_volume` 等），不干扰全市场基线
- `get_watchlist_data()` 每次按 watchlist 文件对齐缓存：新加入的立即从全市场快照补全，删除的立即剔除
- akshare 1.18.x 代码带市场前缀：`sh600519`、`sz000001`、`bj920000`

## AI 预测

- 模型：智谱 GLM-4-Flash（免费），联网搜索 `search_std` 引擎
- API Key：环境变量 `ZHIPU_API_KEY`（已持久化到用户环境变量，**不提交 git**）
- 候选池：异动股 + 涨幅榜 Top20 + 换手率榜 + 放量倍数榜，约 50 只
- 日志：`predictions/YYYY-MM-DD.json/.md` + `SUMMARY.md`
- 每日联网搜索约 5 次调用（按次计费）

## 注意事项

- Python 命令用 `py`，不是 `python`
- `watchlist.json`、`predictions/`、`logs/` 均在 `.gitignore`，不提交
- `ZHIPU_API_KEY` 必须在目标机器的用户环境变量里单独配置
