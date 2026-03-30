use anyhow::{Result, anyhow, bail};
use color_vlc::decoder::yolo::YoloDetector;
use opencv::{
    core::{self, Mat, Point, Point2f, Scalar, Size},
    imgcodecs, imgproc,
    prelude::*,
    wechat_qrcode,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use video_rs::{Decoder, Error as VideoError};

const CHUNK_SIZE: usize = 2940;
const STAGE1_BORDER_PX: i32 = 20;
const WARP_MARGIN_PX: i32 = 40;
const QR_SIZE_PX: i32 = 556;

struct FinderTemplates {
    size: i32,
    tl: Mat,
    tr: Mat,
    bl: Mat,
}

#[derive(Clone, Copy)]
struct RefColors {
    r: [f32; 3],
    g: [f32; 3],
    b: [f32; 3],
    w: [f32; 3],
}

fn frame_to_bgr_mat(frame: &video_rs::Frame) -> Result<Mat> {
    let (h, _w, c) = frame.dim();
    if c != 3 {
        bail!("video-rs frame channel mismatch: expected 3, got {c}");
    }

    let data = frame
        .as_slice_memory_order()
        .ok_or_else(|| anyhow!("video-rs frame is not contiguous"))?;

    let row_major = Mat::from_slice(data)?;
    let rgb = row_major.reshape(3, h as i32)?;

    let mut bgr = Mat::default();
    imgproc::cvt_color(
        &rgb,
        &mut bgr,
        imgproc::COLOR_RGB2BGR,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    Ok(bgr)
}

fn draw_finder_pattern(
    img: &mut Mat,
    r_module: i32,
    c_module: i32,
    box_size: i32,
    margin: i32,
    color: Scalar,
    white: Scalar,
) -> Result<()> {
    let r = r_module * box_size + margin;
    let c = c_module * box_size + margin;
    let s = 7 * box_size;

    imgproc::rectangle(
        img,
        core::Rect::new(c, r, s, s),
        color,
        -1,
        imgproc::LINE_8,
        0,
    )?;

    let r1 = r + box_size;
    let c1 = c + box_size;
    let s1 = 5 * box_size;
    imgproc::rectangle(
        img,
        core::Rect::new(c1, r1, s1, s1),
        white,
        -1,
        imgproc::LINE_8,
        0,
    )?;

    let r2 = r + 2 * box_size;
    let c2 = c + 2 * box_size;
    let s2 = 3 * box_size;
    imgproc::rectangle(
        img,
        core::Rect::new(c2, r2, s2, s2),
        color,
        -1,
        imgproc::LINE_8,
        0,
    )?;

    Ok(())
}

fn build_templates() -> Result<Vec<FinderTemplates>> {
    let mut out = Vec::new();

    for size in (8..=120).rev().step_by(4) {
        let m = i32::max(1, (size as f32 / 7.0).round() as i32);
        let m2 = 2 * m;

        let make_tpl = |color: Scalar| -> Result<Mat> {
            let mut tpl = Mat::new_size_with_default(
                Size::new(size, size),
                core::CV_8UC3,
                Scalar::new(255.0, 255.0, 255.0, 0.0),
            )?;
            imgproc::rectangle(
                &mut tpl,
                core::Rect::new(0, 0, size, size),
                color,
                -1,
                imgproc::LINE_8,
                0,
            )?;
            imgproc::rectangle(
                &mut tpl,
                core::Rect::new(m, m, size - 2 * m, size - 2 * m),
                Scalar::new(255.0, 255.0, 255.0, 0.0),
                -1,
                imgproc::LINE_8,
                0,
            )?;
            imgproc::rectangle(
                &mut tpl,
                core::Rect::new(m2, m2, size - 2 * m2, size - 2 * m2),
                color,
                -1,
                imgproc::LINE_8,
                0,
            )?;
            Ok(tpl)
        };

        out.push(FinderTemplates {
            size,
            tl: make_tpl(Scalar::new(0.0, 0.0, 255.0, 0.0))?,
            tr: make_tpl(Scalar::new(0.0, 255.0, 0.0, 0.0))?,
            bl: make_tpl(Scalar::new(255.0, 0.0, 0.0, 0.0))?,
        });
    }

    Ok(out)
}

fn find_finder_patterns(
    frame: &Mat,
    templates: &[FinderTemplates],
) -> Result<Option<[Point2f; 3]>> {
    let mut best_overall = -1.0f64;
    let mut best_pts: Option<[Point2f; 3]> = None;

    for t in templates {
        if frame.rows() < t.size || frame.cols() < t.size {
            continue;
        }

        let mut res_tl = Mat::default();
        let mut res_tr = Mat::default();
        let mut res_bl = Mat::default();

        imgproc::match_template(
            frame,
            &t.tl,
            &mut res_tl,
            imgproc::TM_CCOEFF_NORMED,
            &Mat::default(),
        )?;
        imgproc::match_template(
            frame,
            &t.tr,
            &mut res_tr,
            imgproc::TM_CCOEFF_NORMED,
            &Mat::default(),
        )?;
        imgproc::match_template(
            frame,
            &t.bl,
            &mut res_bl,
            imgproc::TM_CCOEFF_NORMED,
            &Mat::default(),
        )?;

        let mut max_val_tl = 0.0;
        let mut max_val_tr = 0.0;
        let mut max_val_bl = 0.0;
        let mut max_loc_tl = Point::new(0, 0);
        let mut max_loc_tr = Point::new(0, 0);
        let mut max_loc_bl = Point::new(0, 0);

        core::min_max_loc(
            &res_tl,
            None,
            Some(&mut max_val_tl),
            None,
            Some(&mut max_loc_tl),
            &Mat::default(),
        )?;
        core::min_max_loc(
            &res_tr,
            None,
            Some(&mut max_val_tr),
            None,
            Some(&mut max_loc_tr),
            &Mat::default(),
        )?;
        core::min_max_loc(
            &res_bl,
            None,
            Some(&mut max_val_bl),
            None,
            Some(&mut max_loc_bl),
            &Mat::default(),
        )?;

        let score = max_val_tl + max_val_tr + max_val_bl;
        if score > best_overall && max_val_tl > 0.4 && max_val_tr > 0.4 && max_val_bl > 0.4 {
            let offset = t.size as f32 / 2.0;
            best_overall = score;
            best_pts = Some([
                Point2f::new(max_loc_tl.x as f32 + offset, max_loc_tl.y as f32 + offset),
                Point2f::new(max_loc_tr.x as f32 + offset, max_loc_tr.y as f32 + offset),
                Point2f::new(max_loc_bl.x as f32 + offset, max_loc_bl.y as f32 + offset),
            ]);

            if score > 2.7 {
                break;
            }
        }
    }

    Ok(best_pts)
}

fn get_warped_frame(
    frame: &Mat,
    templates: &[FinderTemplates],
) -> Result<Option<(Mat, RefColors)>> {
    let Some([pt_tl, pt_tr, pt_bl]) = find_finder_patterns(frame, templates)? else {
        return Ok(None);
    };

    let target_size = QR_SIZE_PX + 2 * WARP_MARGIN_PX;
    let p1 = 18.0f32 + WARP_MARGIN_PX as f32;
    let p2 = 538.0f32 + WARP_MARGIN_PX as f32;

    let src_pts = core::Vector::<Point2f>::from_iter([pt_tl, pt_tr, pt_bl]);
    let dst_pts = core::Vector::<Point2f>::from_iter([
        Point2f::new(p1, p1),
        Point2f::new(p2, p1),
        Point2f::new(p1, p2),
    ]);

    let m = imgproc::get_affine_transform(&src_pts, &dst_pts)?;

    let mut warped = Mat::default();
    imgproc::warp_affine(
        frame,
        &mut warped,
        &m,
        Size::new(target_size, target_size),
        imgproc::INTER_CUBIC,
        core::BORDER_CONSTANT,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
    )?;

    let ip1 = p1 as i32;
    let ip2 = p2 as i32;
    let pr = *warped.at_2d::<core::Vec3b>(ip1, ip1)?;
    let pg = *warped.at_2d::<core::Vec3b>(ip1, ip2)?;
    let pb = *warped.at_2d::<core::Vec3b>(ip2, ip1)?;

    let ref_colors = RefColors {
        r: [pr[0] as f32, pr[1] as f32, pr[2] as f32],
        g: [pg[0] as f32, pg[1] as f32, pg[2] as f32],
        b: [pb[0] as f32, pb[1] as f32, pb[2] as f32],
        w: [255.0, 255.0, 255.0],
    };

    let total_margin = 4 + WARP_MARGIN_PX;
    draw_finder_pattern(
        &mut warped,
        0,
        0,
        4,
        total_margin,
        Scalar::new(0.0, 0.0, 0.0, 0.0),
        Scalar::new(255.0, 255.0, 255.0, 0.0),
    )?;
    draw_finder_pattern(
        &mut warped,
        0,
        130,
        4,
        total_margin,
        Scalar::new(0.0, 0.0, 0.0, 0.0),
        Scalar::new(255.0, 255.0, 255.0, 0.0),
    )?;
    draw_finder_pattern(
        &mut warped,
        130,
        0,
        4,
        total_margin,
        Scalar::new(0.0, 0.0, 0.0, 0.0),
        Scalar::new(255.0, 255.0, 255.0, 0.0),
    )?;

    Ok(Some((warped, ref_colors)))
}

fn extract_qr_bits(warped: &Mat, refs: RefColors) -> Result<(Mat, Mat)> {
    let h = warped.rows();
    let w = warped.cols();

    let mut img_a = Mat::new_size_with_default(
        Size::new(w, h),
        core::CV_8UC1,
        Scalar::new(255.0, 0.0, 0.0, 0.0),
    )?;
    let mut img_b = Mat::new_size_with_default(
        Size::new(w, h),
        core::CV_8UC1,
        Scalar::new(255.0, 0.0, 0.0, 0.0),
    )?;

    for y in 0..h {
        for x in 0..w {
            let p = *warped.at_2d::<core::Vec3b>(y, x)?;
            let fv = [p[0] as f32, p[1] as f32, p[2] as f32];

            let dist = |a: [f32; 3], b: [f32; 3]| -> f32 {
                let d0 = a[0] - b[0];
                let d1 = a[1] - b[1];
                let d2 = a[2] - b[2];
                (d0 * d0 + d1 * d1 + d2 * d2).sqrt()
            };

            let dr = dist(fv, refs.r);
            let dg = dist(fv, refs.g);
            let db = dist(fv, refs.b);
            let dw = dist(fv, refs.w);

            let is_r = dr < dg && dr < db && dr < dw;
            let is_g = dg < dr && dg < db && dg < dw;
            let is_b = db < dr && db < dg && db < dw;

            if is_r || is_b {
                *img_a.at_2d_mut::<u8>(y, x)? = 0;
            }
            if is_g || is_b {
                *img_b.at_2d_mut::<u8>(y, x)? = 0;
            }
        }
    }

    let total_margin = 4 + WARP_MARGIN_PX;
    draw_finder_pattern(
        &mut img_a,
        0,
        0,
        4,
        total_margin,
        Scalar::new(0.0, 0.0, 0.0, 0.0),
        Scalar::new(255.0, 255.0, 255.0, 0.0),
    )?;
    draw_finder_pattern(
        &mut img_a,
        0,
        130,
        4,
        total_margin,
        Scalar::new(0.0, 0.0, 0.0, 0.0),
        Scalar::new(255.0, 255.0, 255.0, 0.0),
    )?;
    draw_finder_pattern(
        &mut img_a,
        130,
        0,
        4,
        total_margin,
        Scalar::new(0.0, 0.0, 0.0, 0.0),
        Scalar::new(255.0, 255.0, 255.0, 0.0),
    )?;

    draw_finder_pattern(
        &mut img_b,
        0,
        0,
        4,
        total_margin,
        Scalar::new(0.0, 0.0, 0.0, 0.0),
        Scalar::new(255.0, 255.0, 255.0, 0.0),
    )?;
    draw_finder_pattern(
        &mut img_b,
        0,
        130,
        4,
        total_margin,
        Scalar::new(0.0, 0.0, 0.0, 0.0),
        Scalar::new(255.0, 255.0, 255.0, 0.0),
    )?;
    draw_finder_pattern(
        &mut img_b,
        130,
        0,
        4,
        total_margin,
        Scalar::new(0.0, 0.0, 0.0, 0.0),
        Scalar::new(255.0, 255.0, 255.0, 0.0),
    )?;

    let kernel =
        imgproc::get_structuring_element(imgproc::MORPH_RECT, Size::new(2, 2), Point::new(-1, -1))?;
    let mut out_a = Mat::default();
    let mut out_b = Mat::default();
    imgproc::morphology_ex(
        &img_a,
        &mut out_a,
        imgproc::MORPH_OPEN,
        &kernel,
        Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        Scalar::default(),
    )?;
    imgproc::morphology_ex(
        &img_b,
        &mut out_b,
        imgproc::MORPH_OPEN,
        &kernel,
        Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        Scalar::default(),
    )?;

    Ok((out_a, out_b))
}

fn parse_chunk(res: &str) -> Option<(u32, u32, Vec<u8>)> {
    let mut bytes = Vec::with_capacity(res.len());
    for ch in res.chars() {
        let v = ch as u32;
        if v > 0xFF {
            return None;
        }
        bytes.push(v as u8);
    }

    if bytes.len() < 10 {
        return None;
    }

    let idx = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let total = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let payload_len = u16::from_be_bytes([bytes[8], bytes[9]]) as usize;
    if 10 + payload_len > bytes.len() {
        return None;
    }

    Some((idx, total, bytes[10..10 + payload_len].to_vec()))
}

fn crop_stage1_with_border(
    frame: &Mat,
    det: &color_vlc::decoder::yolo::YoloDetection,
) -> Result<Mat> {
    let mut x1 = det.bbox.x - STAGE1_BORDER_PX;
    let mut y1 = det.bbox.y - STAGE1_BORDER_PX;
    let mut x2 = det.bbox.x + det.bbox.width + STAGE1_BORDER_PX;
    let mut y2 = det.bbox.y + det.bbox.height + STAGE1_BORDER_PX;

    x1 = x1.clamp(0, frame.cols().saturating_sub(1));
    y1 = y1.clamp(0, frame.rows().saturating_sub(1));
    x2 = x2.clamp(1, frame.cols());
    y2 = y2.clamp(1, frame.rows());

    if x2 <= x1 || y2 <= y1 {
        bail!("invalid expanded stage1 ROI");
    }

    let roi = core::Rect::new(x1, y1, x2 - x1, y2 - y1);
    let view = Mat::roi(frame, roi)?;
    let mut out = Mat::default();
    view.copy_to(&mut out)?;
    Ok(out)
}

pub fn decode_video(video: &Path, out_bin: &Path, vout_bin: &Path, workspace: &Path) -> Result<()> {
    let s1_path = workspace.join("train/stage1.onnx");
    let s2_path = workspace.join("train/stage2.onnx");
    if !s1_path.exists() || !s2_path.exists() {
        bail!("ONNX models not found at train/stage1.onnx and train/stage2.onnx");
    }

    let mut detector = YoloDetector::new(&s1_path, &s2_path)?;
    let templates = build_templates()?;

    let wechat_dir = workspace.join("app_data/wechat_qrcode");
    let detect_prototxt = wechat_dir.join("detect.prototxt");
    let detect_model = wechat_dir.join("detect.caffemodel");
    let sr_prototxt = wechat_dir.join("sr.prototxt");
    let sr_model = wechat_dir.join("sr.caffemodel");
    for p in [&detect_prototxt, &detect_model, &sr_prototxt, &sr_model] {
        if !p.exists() {
            bail!("wechat_qrcode model file missing: {}", p.display());
        }
    }

    let mut wechat = wechat_qrcode::WeChatQRCode::new(
        detect_prototxt
            .to_str()
            .ok_or_else(|| anyhow!("invalid detect.prototxt path"))?,
        detect_model
            .to_str()
            .ok_or_else(|| anyhow!("invalid detect.caffemodel path"))?,
        sr_prototxt
            .to_str()
            .ok_or_else(|| anyhow!("invalid sr.prototxt path"))?,
        sr_model
            .to_str()
            .ok_or_else(|| anyhow!("invalid sr.caffemodel path"))?,
    )?;

    if cfg!(debug_assertions) {
        let _ = fs::create_dir_all(workspace.join("debug"));
    }

    video_rs::init().map_err(|e| anyhow!("video-rs init failed: {e}"))?;
    let mut decoder =
        Decoder::new(video).map_err(|e| anyhow!("open video failed {}: {e}", video.display()))?;

    let mut frame_idx = 0usize;
    let mut decoded_chunks: BTreeMap<u32, Vec<u8>> = BTreeMap::new();
    let mut total_chunks: Option<u32> = None;

    println!("Starting decode: {}", video.display());

    loop {
        let (_ts, frame) = match decoder.decode() {
            Ok(v) => v,
            Err(VideoError::DecodeExhausted) => break,
            Err(e) => return Err(anyhow!("video-rs decode failed at frame {frame_idx}: {e}")),
        };

        let bgr = frame_to_bgr_mat(&frame)?;

        let Some(s1) = detector.detect_stage1(&bgr)? else {
            println!("Frame {frame_idx}: No QR code detected. 1");
            println!(
                "Frame {frame_idx}: Decoded {} chunks...",
                decoded_chunks.len()
            );
            frame_idx += 1;
            continue;
        };

        let stage1_frame = crop_stage1_with_border(&bgr, &s1)?;
        if stage1_frame.rows() <= 0 || stage1_frame.cols() <= 0 {
            println!("Frame {frame_idx}: No QR code detected. 2");
            println!(
                "Frame {frame_idx}: Decoded {} chunks...",
                decoded_chunks.len()
            );
            frame_idx += 1;
            continue;
        }

        if cfg!(debug_assertions) {
            let fp = workspace
                .join("debug")
                .join(format!("frame_{frame_idx:06}.png"));
            let _ = imgcodecs::imwrite(
                fp.to_str().ok_or_else(|| anyhow!("invalid debug path"))?,
                &stage1_frame,
                &core::Vector::new(),
            );
        }

        let Some((warped, refs)) = get_warped_frame(&stage1_frame, &templates)? else {
            println!("Frame {frame_idx}: No QR code detected. 3");
            println!(
                "Frame {frame_idx}: Decoded {} chunks...",
                decoded_chunks.len()
            );
            frame_idx += 1;
            continue;
        };

        let (img_a, img_b) = extract_qr_bits(&warped, refs)?;

        if cfg!(debug_assertions) {
            let wp = workspace
                .join("debug")
                .join(format!("wrapped_{frame_idx:06}.png"));
            let ap = workspace
                .join("debug")
                .join(format!("a_{frame_idx:06}.png"));
            let bp = workspace
                .join("debug")
                .join(format!("b_{frame_idx:06}.png"));
            let _ = imgcodecs::imwrite(
                wp.to_str().ok_or_else(|| anyhow!("invalid debug path"))?,
                &warped,
                &core::Vector::new(),
            );
            let _ = imgcodecs::imwrite(
                ap.to_str().ok_or_else(|| anyhow!("invalid debug path"))?,
                &img_a,
                &core::Vector::new(),
            );
            let _ = imgcodecs::imwrite(
                bp.to_str().ok_or_else(|| anyhow!("invalid debug path"))?,
                &img_b,
                &core::Vector::new(),
            );
        }

        let mut hit = false;
        for (img, channel) in [(&img_a, "A"), (&img_b, "B")] {
            let mut rgb = Mat::default();
            imgproc::cvt_color(
                img,
                &mut rgb,
                imgproc::COLOR_GRAY2BGR,
                0,
                core::AlgorithmHint::ALGO_HINT_DEFAULT,
            )?;

            match wechat.detect_and_decode_def(&rgb) {
                Ok(decoded_list) => {
                    for res in decoded_list.iter() {
                        if let Some((idx, total, payload)) = parse_chunk(&res)
                            && let std::collections::btree_map::Entry::Vacant(v) =
                                decoded_chunks.entry(idx)
                        {
                            v.insert(payload);
                            total_chunks = Some(total);
                            hit = true;
                            println!(
                                "Frame {frame_idx} Channel {channel}: Decoded chunk index {idx}"
                            );
                        }
                    }
                }
                Err(e) => println!("Frame {frame_idx} Channel {channel}: Detect error: {e}"),
            }
        }

        if !hit {
            println!("Frame {frame_idx}: No QR code detected. 4");
        }
        println!(
            "Frame {frame_idx}: Decoded {} chunks...",
            decoded_chunks.len()
        );

        frame_idx += 1;
    }

    if decoded_chunks.is_empty() {
        println!("Failed to decode any data.");
        return Ok(());
    }

    let num_chunks = total_chunks.unwrap_or(decoded_chunks.keys().max().copied().unwrap_or(0) + 1);
    println!(
        "\nFinal: {}/{} chunks captured.",
        decoded_chunks.len(),
        num_chunks
    );

    let mut all_data = Vec::new();
    let mut validity = Vec::new();
    for i in 0..num_chunks {
        if let Some(data) = decoded_chunks.get(&i) {
            all_data.extend_from_slice(data);
            validity.extend(std::iter::repeat_n(0xFF, data.len()));
        } else {
            all_data.extend(std::iter::repeat_n(0x00, CHUNK_SIZE));
            validity.extend(std::iter::repeat_n(0x00, CHUNK_SIZE));
        }
    }

    fs::write(out_bin, &all_data)?;
    fs::write(vout_bin, &validity)?;
    println!("Saved to {}", out_bin.display());

    Ok(())
}
