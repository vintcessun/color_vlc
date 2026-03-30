@echo off
set "input=%~1"
set "output=%~n1.play.mkv"

ffmpeg -f lavfi -i color=c=white:s=556x556:d=2 -i "%input%" -filter_complex "[0:v]format=yuv444p,fps=25[v0];[1:v]format=yuv444p[v1];[v0][v1]concat=n=2:v=1:a=0[outv]" -map "[outv]" -c:v ffv1 -level 3 "%output%"