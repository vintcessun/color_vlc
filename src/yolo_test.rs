use anyhow::Result;
use color_vlc::decoder::qrcode::decode_color_blocks_v40m;
use color_vlc::decoder::yolo::YoloDetector;
use opencv::{
    core::{self, MatTrait, MatTraitConst},
    imgcodecs, imgproc,
};
use std::path::Path;

fn run_test(detector: &mut YoloDetector, input_path: &str, output_prefix: &str) -> Result<()> {
    if !Path::new(input_path).exists() {
        println!("Skipping {}: file not found", input_path);
        return Ok(());
    }

    println!("Processing {}...", input_path);
    let frame = imgcodecs::imread(input_path, imgcodecs::IMREAD_COLOR)?;

    // Stage 1
    if let Some(detection) = detector.detect_stage1(&frame)? {
        println!(
            "  Stage 1: Detected QR with confidence {:.2}",
            detection.confidence
        );

        // Save Stage 1 Crop
        let cropped = YoloDetector::crop_for_stage2(&frame, &detection)?;
        let stage1_out_path = format!("{}_stage1_crop.png", output_prefix);
        imgcodecs::imwrite(&stage1_out_path, &cropped, &core::Vector::new())?;
        println!("  Saved stage 1 crop to {}", stage1_out_path);

        // Stage 2
        let stage2_res = detector.detect_stage2(&cropped)?;
        println!("  Stage 2: Completed STN alignment");

        // Save Stage 2 Keypoints visualization on the crop
        let mut kpts_vis = cropped.clone();
        let src_h = cropped.rows() as f32;
        let src_w = cropped.cols() as f32;
        // 注意：这里的 size 应该是推理时的 800
        let in_size = 800.0f32;
        let scale = in_size / src_h.max(src_w);
        let pad_x = (in_size - src_w * scale) / 2.0;
        let pad_y = (in_size - src_h * scale) / 2.0;

        for p in &stage2_res.pred_pts {
            // 从归一化 [-1, 1] 转回 800x800 像素空间
            let px_800 = (p.x + 1.0) * (in_size - 1.0) / 2.0;
            let py_800 = (p.y + 1.0) * (in_size - 1.0) / 2.0;

            // 从 800x800 letterbox 转回原始裁剪图坐标
            let ox = (px_800 - pad_x) / scale;
            let oy = (py_800 - pad_y) / scale;

            imgproc::circle(
                &mut kpts_vis,
                core::Point::new(ox as i32, oy as i32),
                2,
                core::Scalar::new(0.0, 0.0, 255.0, 0.0),
                -1,
                imgproc::LINE_8,
                0,
            )?;
        }
        let stage2_kpts_path = format!("{}_stage2_kpts.png", output_prefix);
        imgcodecs::imwrite(&stage2_kpts_path, &kpts_vis, &core::Vector::new())?;
        println!(
            "  Saved stage 2 keypoints visualization to {}",
            stage2_kpts_path
        );

        // Save Final rectified image
        let out_img_path = format!("{}_result.png", output_prefix);
        imgcodecs::imwrite(&out_img_path, &stage2_res.rectified, &core::Vector::new())?;
        println!("  Saved rectified image to {}", out_img_path);

        // Save grid visualization
        let grid_img = YoloDetector::visualize_grid(&stage2_res.rectified, 40, 4)?;
        let out_grid_path = format!("{}_result_grid.png", output_prefix);
        imgcodecs::imwrite(&out_grid_path, &grid_img, &core::Vector::new())?;
        println!("  Saved grid visualization to {}", out_grid_path);

        // Save adaptive grid visualization
        let adaptive_grid_img = detector.visualize_adaptive_grid(&stage2_res.rectified, 40, 4)?;
        let out_adaptive_grid_path = format!("{}_result_adaptive_grid.png", output_prefix);
        imgcodecs::imwrite(
            &out_adaptive_grid_path,
            &adaptive_grid_img,
            &core::Vector::new(),
        )?;
        println!(
            "  Saved adaptive grid visualization to {}",
            out_adaptive_grid_path
        );

        // --- NEW: Sample and Decode ---
        let version = 40;
        let blocks = detector.sample_grid(&stage2_res.rectified, version)?;
        println!("  Sampled {}x{} grid", blocks.len(), blocks[0].len());

        // Export Sampled Bits as BMP for debugging
        let size = blocks.len();
        let mut img_a = core::Mat::new_size_with_default(
            core::Size::new(size as i32, size as i32),
            core::CV_8UC1,
            core::Scalar::all(255.0),
        )?;
        let mut img_b = core::Mat::new_size_with_default(
            core::Size::new(size as i32, size as i32),
            core::CV_8UC1,
            core::Scalar::all(255.0),
        )?;

        let is_finder_pattern_dark = |x: usize, y: usize| -> bool {
            let is_in_finder = |lx: usize, ly: usize| -> bool {
                (lx == 0 || lx == 6 || ly == 0 || ly == 6)
                    || (lx >= 2 && lx <= 4 && ly >= 2 && ly <= 4)
            };
            if x < 7 && y < 7 {
                return is_in_finder(x, y);
            }
            if x >= size - 7 && y < 7 {
                return is_in_finder(x - (size - 7), y);
            }
            if x < 7 && y >= size - 7 {
                return is_in_finder(x, y - (size - 7));
            }
            false
        };

        for y in 0..size {
            for x in 0..size {
                use color_vlc::shared::QRCodeBlock;
                let block = &blocks[y][x];
                let is_f = is_finder_pattern_dark(x, y);

                // Stream A: Red or Blue, OR Finder Pattern
                if is_f || matches!(block, QRCodeBlock::Red | QRCodeBlock::Blue) {
                    *img_a.at_2d_mut::<u8>(y as i32, x as i32)? = 0;
                }
                // Stream B: Green or Blue, OR Finder Pattern
                if is_f || matches!(block, QRCodeBlock::Green | QRCodeBlock::Blue) {
                    *img_b.at_2d_mut::<u8>(y as i32, x as i32)? = 0;
                }
            }
        }
        imgcodecs::imwrite("debug_sample_a.bmp", &img_a, &core::Vector::new())?;
        imgcodecs::imwrite("debug_sample_b.bmp", &img_b, &core::Vector::new())?;
        println!("  Saved debug sampling maps to debug_sample_a.bmp and debug_sample_b.bmp");

        match decode_color_blocks_v40m(&blocks) {
            Ok((data_a, data_b)) => {
                println!("  Successfully decoded color blocks with fixed Version 40 and ECC M!");
                println!("  Stream A (hex): {:02X?}", data_a);
                println!("  Stream B (hex): {:02X?}", data_b);
            }
            Err(e) => {
                println!("  Failed to decode color blocks: {}", e);
            }
        }
    } else {
        println!("  Stage 1: No QR code detected in {}", input_path);
    }

    Ok(())
}

fn main() -> Result<()> {
    let s1_path = "train/stage1.onnx";
    let s2_path = "train/stage2.onnx";

    if !Path::new(s1_path).exists() || !Path::new(s2_path).exists() {
        println!("Error: ONNX models not found. Please run export_onnx.py first.");
        return Ok(());
    }

    let mut detector = YoloDetector::new(s1_path, s2_path)?;

    run_test(&mut detector, "test.jpg", "test")?;

    Ok(())
}
