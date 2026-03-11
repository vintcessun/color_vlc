use anyhow::{Result, anyhow};
use ndarray::{Array4, s};
use opencv::{
    core::{self, CV_32F, Point2f, Scalar, Size, Vector},
    imgproc,
    prelude::*,
};
use ort::{inputs, session::Session, value::Value};
use std::path::Path;

pub struct YoloDetector {
    session: Session,
}

#[derive(Debug, Clone)]
pub struct Detection {
    pub keypoints: Vec<Point2f>,
}

pub struct LetterboxInfo {
    pub ratio: f32,
    pub dw: f32,
    pub dh: f32,
}

impl YoloDetector {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let session = Session::builder()?.commit_from_file(model_path)?;
        Ok(Self { session })
    }

    /// Letterbox 预处理: 将原始图像缩放并填充至 640x640
    pub fn letterbox(src: &core::Mat, target_size: Size) -> Result<(core::Mat, LetterboxInfo)> {
        let src_size = src.size()?;
        let mut ratio = (target_size.width as f32 / src_size.width as f32)
            .min(target_size.height as f32 / src_size.height as f32);

        let new_unpad_w = (src_size.width as f32 * ratio).round() as i32;
        let new_unpad_h = (src_size.height as f32 * ratio).round() as i32;

        let dw = (target_size.width - new_unpad_w) as f32 / 2.0;
        let dh = (target_size.height - new_unpad_h) as f32 / 2.0;

        let mut resized = core::Mat::default();
        imgproc::resize(
            src,
            &mut resized,
            Size::new(new_unpad_w, new_unpad_h),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        let top = dh.round() as i32;
        let bottom = (target_size.height - new_unpad_h - top) as i32;
        let left = dw.round() as i32;
        let right = (target_size.width - new_unpad_w - left) as i32;

        let mut dst = core::Mat::default();
        core::copy_make_border(
            &resized,
            &mut dst,
            top,
            bottom,
            left,
            right,
            core::BORDER_CONSTANT,
            Scalar::new(114.0, 114.0, 114.0, 0.0),
        )?;

        Ok((
            dst,
            LetterboxInfo {
                ratio,
                dw: left as f32,
                dh: top as f32,
            },
        ))
    }

    /// 将 Mat 转换为 ndarray [1, 3, 640, 640] 并归一化
    fn mat_to_ndarray(img: &core::Mat) -> Result<Array4<f32>> {
        let mut rgb = core::Mat::default();
        imgproc::cvt_color(
            img,
            &mut rgb,
            imgproc::COLOR_BGR2RGB,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        let size = rgb.size()?;
        let mut float_img = core::Mat::default();
        rgb.convert_to(&mut float_img, CV_32F, 1.0 / 255.0, 0.0)?;

        let mut out = Array4::zeros((1, 3, size.height as usize, size.width as usize));

        // 优化：直接从 Mat 内存读取
        for y in 0..size.height {
            for x in 0..size.width {
                let pixel = float_img.at_2d::<core::Vec3f>(y, x)?;
                out[[0, 0, y as usize, x as usize]] = pixel[0];
                out[[0, 1, y as usize, x as usize]] = pixel[1];
                out[[0, 2, y as usize, x as usize]] = pixel[2];
            }
        }

        Ok(out)
    }

    pub fn detect(&mut self, frame: &core::Mat) -> Result<Option<Detection>> {
        let (input_img, info) = Self::letterbox(frame, Size::new(640, 640))?;
        let input_tensor = Self::mat_to_ndarray(&input_img)?;
        let input_value = Value::from_array(input_tensor)?;

        let input_values = inputs!["images" => input_value];

        let outputs = self
            .session
            .run(input_values)
            .map_err(|e| anyhow!("Failed to run YOLO session: {}", e))?;
        let (shape, output_data) = outputs["output0"].try_extract_tensor::<f32>()?;

        // YOLOv8-pose 输出维度通常是 [1, 56, 8400] (对于 640x640 输入)
        // 56 = 4 (box) + 1 (score) + 17 * 3 (keypoints: x, y, conf)
        // 我们需要找到 score 最高的检测结果

        let view = ndarray::ArrayView3::<f32>::from_shape(
            (shape[0] as usize, shape[1] as usize, shape[2] as usize),
            output_data,
        )?;
        let num_anchors = shape[2];

        let mut best_score = 0.0;
        let mut best_idx = None;

        for i in 0..num_anchors {
            let score = view[[0, 4, i as usize]];
            if score > best_score {
                best_score = score;
                best_idx = Some(i);
            }
        }

        if let Some(idx) = best_idx {
            if best_score < 0.5 {
                return Ok(None);
            }

            // 提取关键点 (假设我们只需要前 4 个关键点作为二维码的 4 个角)
            // YOLOv8-pose 关键点从索引 5 开始，每 3 个一组 (x, y, conf)
            let mut kpts = Vec::new();
            for k in 0..4 {
                let kpt_x = view[[0, 5 + k * 3, idx as usize]];
                let kpt_y = view[[0, 5 + k * 3 + 1, idx as usize]];

                // 坐标逆映射
                let raw_x = (kpt_x - info.dw) / info.ratio;
                let raw_y = (kpt_y - info.dh) / info.ratio;
                kpts.push(Point2f::new(raw_x, raw_y));
            }

            return Ok(Some(Detection { keypoints: kpts }));
        }

        Ok(None)
    }

    /// 高清校正: 使用透视变换输出 400x400 的标准图
    pub fn get_warped_qr(src: &core::Mat, kpts: &[Point2f]) -> Result<core::Mat> {
        if kpts.len() < 4 {
            return Err(anyhow::anyhow!(
                "Need 4 keypoints for perspective transform"
            ));
        }

        let dst_pts = [
            Point2f::new(0.0, 0.0),
            Point2f::new(399.0, 0.0),
            Point2f::new(399.0, 399.0),
            Point2f::new(0.0, 399.0),
        ];

        let src_vec = Vector::<Point2f>::from_slice(&kpts[0..4]);
        let dst_vec = Vector::<Point2f>::from_slice(&dst_pts);

        let m = imgproc::get_perspective_transform(&src_vec, &dst_vec, 0)?;
        let mut warped = core::Mat::default();
        imgproc::warp_perspective(
            src,
            &mut warped,
            &m,
            Size::new(400, 400),
            imgproc::INTER_LINEAR,
            core::BORDER_CONSTANT,
            Scalar::default(),
        )?;

        Ok(warped)
    }

    /// 网格化色彩采样
    pub fn sample_grid(warped: &core::Mat, grid_n: usize) -> Result<Vec<Vec<[u8; 3]>>> {
        let cell_size = 400.0 / grid_n as f32;
        let mut grid = vec![vec![[0u8; 3]; grid_n]; grid_n];

        for y in 0..grid_n {
            for x in 0..grid_n {
                let center_x = (x as f32 + 0.5) * cell_size;
                let center_y = (y as f32 + 0.5) * cell_size;

                let pixel = warped.at_2d::<core::Vec3b>(center_y as i32, center_x as i32)?;
                grid[y][x] = [pixel[2], pixel[1], pixel[0]]; // BGR to RGB
            }
        }

        Ok(grid)
    }
}
