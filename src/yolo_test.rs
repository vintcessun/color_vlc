use anyhow::Result;
use color_vlc::decoder::yolo::YoloDetector;
use opencv::{core, imgcodecs};

fn main() -> Result<()> {
    let model_path = "yolov8n-pose.onnx";
    if !std::path::Path::new(model_path).exists() {
        println!(
            "Error: {} not found. Please place the model in the current directory.",
            model_path
        );
        return Ok(());
    }

    let mut detector = YoloDetector::new(model_path)?;

    let test_img_path = "test.png";
    if !std::path::Path::new(test_img_path).exists() {
        println!("Error: {} not found.", test_img_path);
        return Ok(());
    }

    let frame = imgcodecs::imread(test_img_path, imgcodecs::IMREAD_COLOR)?;

    println!("Detecting QR code...");
    if let Some(detection) = detector.detect(&frame)? {
        println!("Detected 4 keypoints: {:?}", detection.keypoints);

        let warped = YoloDetector::get_warped_qr(&frame, &detection.keypoints)?;
        imgcodecs::imwrite("yolo_warped.png", &warped, &core::Vector::new())?;
        println!("Warped image saved to yolo_warped.png");

        let grid_n = 57; // 假设是 version 10: (10-1)*4 + 21 = 57
        let grid = YoloDetector::sample_grid(&warped, grid_n)?;
        println!("Sampled {}x{} grid colors.", grid.len(), grid[0].len());
    } else {
        println!("No QR code detected.");
    }

    Ok(())
}
