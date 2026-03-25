use anyhow::Result;
use color_vlc::decoder::qrcode::{
    decode_color_blocks,
    decode_color_blocks_robust,
    decode_color_blocks_v40m,
};
use color_vlc::decoder::yolo::YoloDetector;
use color_vlc::shared::QRCodeBlock;
use opencv::{
    core::{self, MatTrait, MatTraitConst},
    imgcodecs, imgproc,
};
use std::path::Path;

fn sample_blocks_from_ideal_image(
    img: &core::Mat,
    version: i32,
) -> Result<Option<Vec<Vec<QRCodeBlock>>>> {
    let m = (version - 1) * 4 + 21;
    let n = m + 2;

    if img.cols() % n != 0 || img.rows() % n != 0 {
        return Ok(None);
    }

    let box_w = img.cols() / n;
    let box_h = img.rows() / n;
    if box_w <= 0 || box_h <= 0 || box_w != box_h {
        return Ok(None);
    }

    let mut blocks = vec![vec![QRCodeBlock::White; m as usize]; m as usize];

    for y in 0..m {
        for x in 0..m {
            // border=1: module (x,y) maps to cell (x+1,y+1)
            let cx = (x + 1) * box_w + box_w / 2;
            let cy = (y + 1) * box_h + box_h / 2;
            let p = img.at_2d::<core::Vec3b>(cy, cx)?;
            let b = p[0] as i32;
            let g = p[1] as i32;
            let r = p[2] as i32;

            let block = if r >= 200 && g <= 80 && b <= 80 {
                QRCodeBlock::Red
            } else if g >= 200 && r <= 80 && b <= 80 {
                QRCodeBlock::Green
            } else if b >= 200 && r <= 80 && g <= 80 {
                QRCodeBlock::Blue
            } else {
                QRCodeBlock::White
            };
            blocks[y as usize][x as usize] = block;
        }
    }

    Ok(Some(blocks))
}

fn sample_blocks_with_params(
    img: &core::Mat,
    version: i32,
    start_ratio: f32,
    end_ratio: f32,
    white_delta: f32,
    white_min: f32,
    dom_margin: f32,
) -> Result<Vec<Vec<QRCodeBlock>>> {
    let m = (version - 1) * 4 + 21;
    let n = m + 2;
    let cw = img.cols() as f32 / n as f32;
    let ch = img.rows() as f32 / n as f32;
    let mut blocks = vec![vec![QRCodeBlock::White; m as usize]; m as usize];

    for y in 0..m {
        for x in 0..m {
            let mx = (x + 1) as f32;
            let my = (y + 1) as f32;
            let px_start = (mx * cw + cw * start_ratio) as i32;
            let px_end = (mx * cw + cw * end_ratio) as i32;
            let py_start = (my * ch + ch * start_ratio) as i32;
            let py_end = (my * ch + ch * end_ratio) as i32;

            let mut votes = [0i32; 4];
            for sy in py_start..py_end {
                for sx in px_start..px_end {
                    if sx < 0 || sx >= img.cols() || sy < 0 || sy >= img.rows() {
                        continue;
                    }
                    let p = img.at_2d::<core::Vec3b>(sy, sx)?;
                    let b = p[0] as f32;
                    let g = p[1] as f32;
                    let r = p[2] as f32;
                    let cmax = r.max(g).max(b);
                    let cmin = r.min(g).min(b);
                    let delta = cmax - cmin;

                    if delta < white_delta && cmin > white_min {
                        votes[3] += 1;
                    } else if r > g + dom_margin && r > b + dom_margin {
                        votes[0] += 1;
                    } else if g > r + dom_margin && g > b + dom_margin {
                        votes[1] += 1;
                    } else if b > r + dom_margin && b > g + dom_margin {
                        votes[2] += 1;
                    }
                }
            }

            let mut best = 3usize;
            let mut best_v = votes[3];
            for (i, &v) in votes.iter().enumerate().take(3) {
                if v > best_v {
                    best_v = v;
                    best = i;
                }
            }
            blocks[y as usize][x as usize] = match best {
                0 => QRCodeBlock::Red,
                1 => QRCodeBlock::Green,
                2 => QRCodeBlock::Blue,
                _ => QRCodeBlock::White,
            };
        }
    }

    Ok(blocks)
}

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

            // Save pred_pts and crop dimensions for local-homography analysis
            {
                let mut pts_content = format!(
                    "crop_size: {}x{}\n",
                    cropped.cols(),
                    cropped.rows()
                );
                for (i, pt) in stage2_res.pred_pts.iter().enumerate() {
                    pts_content.push_str(&format!("{}: {},{}\n", i, pt.x, pt.y));
                }
                std::fs::write(format!("{}_pred_pts.txt", output_prefix), &pts_content)?;
                println!("  Saved pred_pts to {}_pred_pts.txt", output_prefix);
            }

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

        match decode_color_blocks(&blocks) {
            Ok((data_a, data_b)) => {
                println!("  Successfully decoded color blocks with auto metadata!");
                println!("  Stream A (hex): {:02X?}", data_a);
                println!("  Stream B (hex): {:02X?}", data_b);
            }
            Err(e) => {
                println!("  Auto metadata decode failed: {}", e);
            }
        }

        match decode_color_blocks_robust(&blocks) {
            Ok((data_a, data_b)) => {
                println!("  Robust decode success!");
                println!("  Stream A (hex): {:02X?}", data_a);
                println!("  Stream B (hex): {:02X?}", data_b);
            }
            Err(e) => {
                println!("  Robust decode failed: {}", e);
            }
        }

        match decode_color_blocks_v40m(&blocks) {
            Ok((data_a, data_b)) => {
                println!("  Successfully decoded color blocks with fixed Version 40 and ECC M!");
                println!("  Stream A (hex): {:02X?}", data_a);
                println!("  Stream B (hex): {:02X?}", data_b);
            }
            Err(e) => {
                println!("  Failed to decode color blocks: {}", e);

                // Fallback: search small sampling phase offsets on rectified image.
                let mut recovered = false;
                for dy in -2..=2 {
                    for dx in -2..=2 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }

                        let m = core::Mat::from_slice_2d(&[
                            [1.0f64, 0.0f64, dx as f64],
                            [0.0f64, 1.0f64, dy as f64],
                        ])?;

                        let mut shifted = core::Mat::default();
                        imgproc::warp_affine(
                            &stage2_res.rectified,
                            &mut shifted,
                            &m,
                            core::Size::new(stage2_res.rectified.cols(), stage2_res.rectified.rows()),
                            imgproc::INTER_NEAREST,
                            core::BORDER_REPLICATE,
                            core::Scalar::default(),
                        )?;

                        let shifted_blocks = detector.sample_grid(&shifted, version)?;
                        if let Ok((data_a, data_b)) = decode_color_blocks_robust(&shifted_blocks) {
                            println!("  Recovered by phase search: dx={}, dy={}", dx, dy);
                            println!("  Stream A (hex): {:02X?}", data_a);
                            println!("  Stream B (hex): {:02X?}", data_b);
                            recovered = true;
                            break;
                        }
                    }
                    if recovered {
                        break;
                    }
                }
                if !recovered {
                    // Last resort: brute-force sampling thresholds on rectified image.
                    'search: for &(start_ratio, end_ratio) in &[(0.30, 0.70), (0.25, 0.75), (0.35, 0.65)] {
                        for &white_delta in &[20.0, 26.0, 32.0, 40.0] {
                            for &white_min in &[80.0, 95.0, 110.0] {
                                for &dom_margin in &[10.0, 16.0, 22.0, 28.0] {
                                    let brute_blocks = sample_blocks_with_params(
                                        &stage2_res.rectified,
                                        version,
                                        start_ratio,
                                        end_ratio,
                                        white_delta,
                                        white_min,
                                        dom_margin,
                                    )?;
                                    if let Ok((data_a, data_b)) = decode_color_blocks_robust(&brute_blocks) {
                                        println!(
                                            "  Recovered by threshold search: win=({:.2},{:.2}) wd={:.1} wm={:.1} dm={:.1}",
                                            start_ratio,
                                            end_ratio,
                                            white_delta,
                                            white_min,
                                            dom_margin
                                        );
                                        println!("  Stream A (hex): {:02X?}", data_a);
                                        println!("  Stream B (hex): {:02X?}", data_b);
                                        recovered = true;
                                        break 'search;
                                    }
                                }
                            }
                        }
                    }

                    if !recovered {
                        println!("  Phase-search + threshold-search did not recover test image.");
                    }
                }
            }
        }
    } else {
        println!("  Stage 1: No QR code detected in {}", input_path);
    }

    Ok(())
}

fn run_direct_grid_test(detector: &mut YoloDetector, input_path: &str) -> Result<()> {
    if !Path::new(input_path).exists() {
        println!("Skipping direct-grid test {}: file not found", input_path);
        return Ok(());
    }

    println!("Direct-grid test on {}...", input_path);
    let rectified = imgcodecs::imread(input_path, imgcodecs::IMREAD_COLOR)?;

    if let Some(ideal_blocks) = sample_blocks_from_ideal_image(&rectified, 40)? {
        match decode_color_blocks(&ideal_blocks) {
            Ok((data_a, data_b)) => {
                println!("  Ideal-sampler decode success (auto metadata)");
                println!("  Stream A bytes: {}", data_a.len());
                println!("  Stream B bytes: {}", data_b.len());
            }
            Err(e) => {
                println!("  Ideal-sampler auto decode failed: {}", e);
            }
        }

        match decode_color_blocks_v40m(&ideal_blocks) {
            Ok((data_a, data_b)) => {
                println!("  Ideal-sampler decode success (v40m)");
                println!("  Stream A bytes: {}", data_a.len());
                println!("  Stream B bytes: {}", data_b.len());
            }
            Err(e) => {
                println!("  Ideal-sampler v40m decode failed: {}", e);
            }
        }
    }

    let blocks = detector.sample_grid(&rectified, 40)?;

    match decode_color_blocks(&blocks) {
        Ok((data_a, data_b)) => {
            println!("  Direct-grid decode success (auto metadata)");
            println!("  Stream A bytes: {}", data_a.len());
            println!("  Stream B bytes: {}", data_b.len());
        }
        Err(e) => {
            println!("  Direct-grid auto decode failed: {}", e);
        }
    }

    match decode_color_blocks_robust(&blocks) {
        Ok((data_a, data_b)) => {
            println!("  Direct-grid robust decode success");
            println!("  Stream A bytes: {}", data_a.len());
            println!("  Stream B bytes: {}", data_b.len());
        }
        Err(e) => {
            println!("  Direct-grid robust decode failed: {}", e);
        }
    }

    match decode_color_blocks_v40m(&blocks) {
        Ok((data_a, data_b)) => {
            println!("  Direct-grid decode success (v40m)");
            println!("  Stream A bytes: {}", data_a.len());
            println!("  Stream B bytes: {}", data_b.len());
        }
        Err(e) => {
            println!("  Direct-grid v40m decode failed: {}", e);
        }
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

    run_direct_grid_test(&mut detector, "test_qr40_color.png")?;
    run_test(&mut detector, "test.jpg", "test")?;

    Ok(())
}
