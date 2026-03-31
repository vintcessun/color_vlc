import os
import subprocess
import glob


def main():
    testcase_dir = "./testcase"
    # 获取所有 .bin 文件并按名称排序，保证顺序执行
    bins = sorted(glob.glob(os.path.join(testcase_dir, "*.bin")))

    for bin_file in bins:
        # 获取文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(bin_file))[0]
        mkv_file = os.path.join(testcase_dir, f"{base_name}.mkv")
        play_mkv_file = os.path.join(testcase_dir, f"{base_name}.play.mkv")

        # 1. 执行 encoder
        print(f"\n>>> Encoding: {bin_file} to {mkv_file}")
        cargo_cmd = [
            "cargo",
            "run",
            "--release",
            "--bin",
            "encoder",
            "--",
            bin_file,
            mkv_file,
            "1000",
        ]
        try:
            subprocess.run(
                cargo_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            print(f"Error running cargo for {bin_file}: {e}")
            continue

        # 2. 模拟 make_play.bat 的逻辑
        # ffmpeg -f lavfi -i color=c=white:s=556x556:d=2 -i "%input%" -filter_complex "[0:v]format=yuv444p,fps=25[v0];[1:v]format=yuv444p[v1];[v0][v1]concat=n=2:v=1:a=0[outv]" -map "[outv]" -c:v ffv1 -level 3 "%output%"
        print(f">>> Generating play mkv: {play_mkv_file}")
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=white:s=556x556:d=2",
            "-i",
            mkv_file,
            "-filter_complex",
            "[0:v]format=yuv444p,fps=25[v0];[1:v]format=yuv444p[v1];[v0][v1]concat=n=2:v=1:a=0[outv]",
            "-map",
            "[outv]",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            play_mkv_file,
        ]
        try:
            subprocess.run(
                ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            print(f"Error running ffmpeg for {mkv_file}: {e}")
            continue

        print(f">>> Try decode {mkv_file} to out.bin and vout.bin")
        decoder_cmd = [
            r"D:\Scripts\color-vlc\.venv\Scripts\python.exe",
            "decoder.py",
            mkv_file,
            "out.bin",
            "vout.bin",
        ]
        try:
            decoder_process = subprocess.run(
                decoder_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout = decoder_process.stdout.decode("utf-8")
            if stdout.find("Final: 60/60 chunks captured.") == -1:
                raise ValueError(
                    f"Decoder output does not indicate success for {mkv_file}:\n{stdout}"
                )
        except subprocess.CalledProcessError as e:
            print(f"Error running decoder for {mkv_file}: {e}")
            continue

    print("\nAll tasks completed!\n")


if __name__ == "__main__":
    main()
