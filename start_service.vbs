' A股实时监控服务 - 无窗口后台启动器
' 由开机自启快捷方式调用；以隐藏窗口方式运行 uvicorn，并将日志追加到 logs\service.log
Option Explicit
Dim sh, cmd, projDir, pyExe, logFile, q
q = Chr(34)   ' 双引号字符，避免字符串内引号转义歧义
projDir = "C:\Project\a-stock-monitor"
pyExe   = "C:\Users\Esp\AppData\Local\Programs\Python\Python313\python.exe"
logFile = projDir & "\logs\service.log"

Set sh = CreateObject("WScript.Shell")
cmd = "cmd /c cd /d " & projDir & " && " & q & pyExe & q & _
      " -m uvicorn app.main:app --host 0.0.0.0 --port 8000 >> " & q & logFile & q & " 2>&1"
' 第二个参数 0 = 隐藏窗口；第三个参数 False = 不等待子进程退出
sh.Run cmd, 0, False
