# A股 AI 预测 - 每日调度脚本
# 在【交易日收盘后】运行（建议周一~周五 15:30）。两步：
#   1) 复盘上一交易日的预测（用今天的实际收盘涨跌打分）
#   2) 触发今天的预测（预测下一交易日，后台运行约几分钟）
#
# 挂到 Windows 计划任务（每天会自动跳过周末/节假日——非交易日没有上一条预测可复盘、
# 当天预测虽会生成但目标日是下个交易日，复盘时会自然对齐）：
#   触发器：每天 15:30
#   操作：powershell -ExecutionPolicy Bypass -File "C:\Project\a-stock-monitor\predict_daily.ps1"

$base = "http://127.0.0.1:8000"

Write-Host "[1/2] 复盘上一交易日预测..."
try {
    $r = Invoke-RestMethod -Method Post "$base/api/predict/review-pending" -TimeoutSec 180
    Write-Host "复盘结果: $($r | ConvertTo-Json -Depth 2 -Compress)"
} catch {
    Write-Host "复盘跳过/失败: $_"
}

Write-Host "[2/2] 触发今日预测（后台运行）..."
try {
    $r = Invoke-RestMethod -Method Post "$base/api/predict/run?web_search=true" -TimeoutSec 30
    Write-Host "预测已触发: $($r.message)"
} catch {
    Write-Host "预测触发失败: $_"
}
