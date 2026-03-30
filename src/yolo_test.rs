use anyhow::{Context, Result, anyhow, bail};
use color_vlc::decoder::yolo::YoloDetector;
use opencv::{
    core::{self, MatTraitConst},
    imgcodecs,
};
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

const OUTPUT_SIZE: i32 = 1074;
const FRAME_GLOB_WIDTH: usize = 8;
const OUTPUT_NAME_WIDTH: usize = 4;
const SUPPORTED_VIDEO_EXTS: &[&str] = &["mp4", "avi", "mov", "mkv", "wmv", "flv", "webm", "m4v"];

fn resolve_video_inputs(input: &Path) -> Result<Vec<PathBuf>> {
    if input.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }

    if !input.is_dir() {
        bail!("arg0 must be a video file or a directory containing video files");
    }

    let mut videos = Vec::new();
    for entry in fs::read_dir(input)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(ext) = path.extension().and_then(|s| s.to_str()) else {
            continue;
        };
        if SUPPORTED_VIDEO_EXTS
            .iter()
            .any(|candidate| ext.eq_ignore_ascii_case(candidate))
        {
            videos.push(path);
        }
    }

    videos.sort();
    if videos.is_empty() {
        bail!("no video files found under arg0 directory");
    }
    Ok(videos)
}

fn ensure_clean_dir(dir: &Path) -> Result<()> {
    if dir.exists() {
        fs::remove_dir_all(dir)?;
    }
    fs::create_dir_all(dir)?;
    Ok(())
}

fn extract_frames_with_ffmpeg(video_path: &Path, frame_dir: &Path) -> Result<Vec<PathBuf>> {
    ensure_clean_dir(frame_dir)?;

    let frame_pattern = frame_dir.join(format!("%0{}d.png", FRAME_GLOB_WIDTH));
    let status = Command::new("ffmpeg")
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-y")
        .arg("-i")
        .arg(video_path)
        .arg(&frame_pattern)
        .status()
        .with_context(|| format!("failed to launch ffmpeg for {}", video_path.display()))?;

    if !status.success() {
        bail!(
            "ffmpeg failed when extracting frames from {}",
            video_path.display()
        );
    }

    let mut frames = Vec::new();
    for entry in fs::read_dir(frame_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("png"))
        {
            frames.push(path);
        }
    }
    frames.sort();

    if frames.is_empty() {
        bail!("ffmpeg extracted no frames from {}", video_path.display());
    }

    Ok(frames)
}

fn make_blank_output() -> Result<core::Mat> {
    core::Mat::new_size_with_default(
        core::Size::new(OUTPUT_SIZE, OUTPUT_SIZE),
        core::CV_8UC3,
        core::Scalar::all(0.0),
    )
    .map_err(Into::into)
}

fn process_frame(detector: &mut YoloDetector, frame_path: &Path) -> Result<core::Mat> {
    let frame = imgcodecs::imread(
        frame_path
            .to_str()
            .ok_or_else(|| anyhow!("invalid frame path: {}", frame_path.display()))?,
        imgcodecs::IMREAD_COLOR,
    )?;

    let Some(detection) = detector.detect_stage1(&frame)? else {
        return make_blank_output();
    };

    let cropped = YoloDetector::crop_for_stage2(&frame, &detection)?;
    if cropped.rows() <= 0 || cropped.cols() <= 0 {
        return make_blank_output();
    }

    let stage2 = detector.detect_stage2(&cropped)?;
    if stage2.rectified.rows() != OUTPUT_SIZE || stage2.rectified.cols() != OUTPUT_SIZE {
        bail!(
            "stage2 output size mismatch for {}: got {}x{}",
            frame_path.display(),
            stage2.rectified.cols(),
            stage2.rectified.rows()
        );
    }

    Ok(stage2.rectified)
}

fn save_output_frame(output_dir: &Path, index: usize, image: &core::Mat) -> Result<()> {
    let out_path = output_dir.join(format!("{:0width$}.png", index, width = OUTPUT_NAME_WIDTH));
    imgcodecs::imwrite(
        out_path
            .to_str()
            .ok_or_else(|| anyhow!("invalid output path: {}", out_path.display()))?,
        image,
        &core::Vector::new(),
    )?;
    Ok(())
}

fn process_video(
    detector: &mut YoloDetector,
    video_path: &Path,
    output_dir: &Path,
    start_index: usize,
) -> Result<usize> {
    let temp_root = output_dir.join("_ffmpeg_frames_tmp");
    let video_stem = video_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("video");
    let frame_dir = temp_root.join(video_stem);

    println!("Extracting frames from {}...", video_path.display());
    let frames = extract_frames_with_ffmpeg(video_path, &frame_dir)?;
    println!("  Extracted {} frames", frames.len());

    let mut next_index = start_index;
    for frame_path in frames {
        let output = match process_frame(detector, &frame_path) {
            Ok(img) => img,
            Err(error) => {
                println!("  Frame {} failed: {}", frame_path.display(), error);
                make_blank_output()?
            }
        };
        save_output_frame(output_dir, next_index, &output)?;
        next_index += 1;
    }

    if frame_dir.exists() {
        fs::remove_dir_all(&frame_dir)?;
    }
    if temp_root.exists() && fs::read_dir(&temp_root)?.next().is_none() {
        fs::remove_dir_all(&temp_root)?;
    }

    Ok(next_index)
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args_os().collect();
    if args.len() != 3 {
        eprintln!("Usage: yolo_test <video-file-or-directory> <output-directory>");
        bail!("invalid arguments");
    }

    let input_path = PathBuf::from(&args[1]);
    let output_dir = PathBuf::from(&args[2]);
    fs::create_dir_all(&output_dir)?;

    let s1_path = Path::new("train/stage1.onnx");
    let s2_path = Path::new("train/stage2.onnx");
    if !s1_path.exists() || !s2_path.exists() {
        bail!("ONNX models not found at train/stage1.onnx and train/stage2.onnx");
    }

    let videos = resolve_video_inputs(&input_path)?;
    let mut detector = YoloDetector::new(s1_path, s2_path)?;

    let mut next_index = 1usize;
    for video in videos {
        next_index = process_video(&mut detector, &video, &output_dir, next_index)?;
    }

    println!(
        "Done. Wrote {} output frames to {}",
        next_index.saturating_sub(1),
        output_dir.display()
    );
    Ok(())
}
