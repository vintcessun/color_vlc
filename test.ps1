# 获取输入的 ID（如 01, 02...）
$id = $args[0]
if (-not $id) { 
    Write-Host "错误: 请输入测试编号 (例如: .\test.ps1 01)" -ForegroundColor Red
    exit 
}

# 定义路径
$exePath = ".\decoder_run.exe"  # 假设你的 exe 在 dist 目录下
$mp4Path = ".\testcase\$id.phone.mp4"
$outPath = ".\testcase\$id.exe.out"
$voutPath = ".\testcase\$id.exe.vout"

# 检查文件是否存在
if (-not (Test-Path $exePath)) { Write-Host "找不到 $exePath" -ForegroundColor Red; exit }
if (-not (Test-Path $mp4Path)) { Write-Host "找不到视频文件 $mp4Path" -ForegroundColor Red; exit }

Write-Host "------------------------------------" -ForegroundColor Cyan
Write-Host "正在启动 Color-VLC 解码器 (ID: $id)..." -ForegroundColor Yellow
Write-Host "注意：由于打包模块较多，首次启动解压可能需要 10-20 秒，请耐心等待。" -ForegroundColor Gray

# 记录开始时间
$startTime = Get-Date

# 执行编译后的 EXE
# 直接传递参数：视频路径 输出路径 校验路径
& $exePath $mp4Path $outPath $voutPath

# 记录结束时间
$endTime = Get-Date
$duration = ($endTime - $startTime).TotalSeconds

Write-Host "------------------------------------" -ForegroundColor Cyan
Write-Host "测试完成！" -ForegroundColor Green
Write-Host "输出文件: $outPath"
Write-Host "耗时: $duration 秒"