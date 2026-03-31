1..10 | ForEach-Object {
    $num = $_.ToString("00")
    Write-Host ">>> 正在处理第 $num 组数据..." -ForegroundColor Magenta
    .\test.ps1 $num
}