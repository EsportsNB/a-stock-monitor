"""无窗口启动入口（供 pythonw.exe 调用，用于开机自启）。

pythonw 没有控制台，标准输出句柄无效，直接用会让 uvicorn 日志和 print 报错。
这里在启动 uvicorn 之前先把 stdout/stderr 重定向到日志文件，保证无窗口运行稳定且有日志。
"""
import sys
import os
import socket

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)                 # 确保 app/static 等相对路径正确
sys.path.insert(0, BASE)

# 开机自启用了「注册表 Run + 启动文件夹」双保险；若 8000 已有实例在跑则本次直接退出
try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _probe:
        if _probe.connect_ex(("127.0.0.1", 8000)) == 0:
            sys.exit(0)
except SystemExit:
    raise
except Exception:
    pass

log_dir = os.path.join(BASE, "logs")
os.makedirs(log_dir, exist_ok=True)
try:
    _logf = open(os.path.join(log_dir, "service.log"), "a", encoding="utf-8", buffering=1)
    sys.stdout = _logf
    sys.stderr = _logf
except Exception:
    pass

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, log_level="info")
