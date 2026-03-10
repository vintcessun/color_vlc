@echo off
set "input=%~1"
set "output=cyrene.play.mkv"

echo 正在处理文件: %input%
echo 正在添加白幕并转码为视频容器...

:: 修正点：
:: 1. 移除了音频处理逻辑（因为原片没有音频）
:: 2. 使用 vcodec=libx264 或保持编码一致（这里使用 x264 保证兼容性）
:: 3. 统一滤镜处理逻辑
ffmpeg -f lavfi -i color=c=white:s=716x716:d=2 -i "%input%" -filter_complex "[0:v]format=yuv444p,fps=25[v0];[1:v]format=yuv444p[v1];[v0][v1]concat=n=2:v=1:a=0[outv]" -map "[outv]" -c:v ffv1 -level 3 "%output%"

echo 处理完成！文件已保存为 %output%
pause