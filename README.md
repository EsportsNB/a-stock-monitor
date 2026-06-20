# A股实时监控服务

这是一个可部署在本地的中国大陆股市实时监控服务。

## 目录结构

- `app/main.py` - FastAPI 服务入口
- `app/stock_monitor.py` - 使用 akshare 抓取股票和指数数据并生成趋势总结
- `app/schemas.py` - Pydantic 数据模型
- `app/static/index.html` - 简单前端页面
- `requirements.txt` - 依赖列表

## 运行步骤

1. 安装 Python 依赖

```bash
cd C:\Project\a-stock-monitor
python -m pip install -r requirements.txt
```

2. 启动服务

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

3. 浏览器访问

`http://127.0.0.1:8000`

## 使用说明

- 在输入框中输入股票代码，支持 `600519.SH`、`000001.SZ` 或 `600519`、`000001` 自动补全。
- 点击查询后会显示当前价格、涨跌幅、趋势判断和文本摘要。
- 页面加载时会自动获取上证、深证和创业板主要指数数据。
