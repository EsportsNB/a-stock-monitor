param(
    [switch]$InstallPython
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

function Find-Python {
    $candidates = @('python.exe', 'py.exe')
    foreach ($name in $candidates) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) {
            return $cmd.Path
        }
    }
    return $null
}

$pythonPath = Find-Python
if (-not $pythonPath) {
    Write-Host "未找到 Python 解释器。"
    if ($InstallPython) {
        if (Get-Command winget -ErrorAction SilentlyContinue) {
            Write-Host "尝试通过 winget 安装 Python 3.11..."
            winget install --id Python.Python.3 --exact --accept-source-agreements --accept-package-agreements
            $pythonPath = Find-Python
        }
    }
}

if (-not $pythonPath) {
    Write-Host "请先安装 Python 3.11+，或确认 python/py 可执行文件已加入 PATH。"
    Write-Host "推荐命令：winget install --id Python.Python.3 --exact --accept-source-agreements --accept-package-agreements"
    exit 1
}

Write-Host "使用 Python：$pythonPath"
& "$pythonPath" -m pip install --upgrade pip
& "$pythonPath" -m pip install -r requirements.txt
Write-Host "依赖安装完成。正在启动服务..."
& "$pythonPath" -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
