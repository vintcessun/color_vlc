import sys
import os

id = sys.argv[1]

os.system(
    rf"D:\Scripts\color-vlc\.venv\Scripts\python.exe decoder.py .\testcase\{id}.phone.mp4 .\testcase\{id}.out .\testcase\{id}.vout"
)
