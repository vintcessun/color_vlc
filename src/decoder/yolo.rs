use anyhow::Result;
use ndarray::{Array2, Array4};
use opencv::{
    core::{self, Point2f, Rect, Scalar, Size},
    imgproc,
    prelude::*,
};
use ort::{execution_providers::CUDAExecutionProvider, inputs, session::Session, value::Value};
use std::path::Path;

pub struct YoloDetection {
    pub bbox: Rect,
    pub confidence: f32,
    pub keypoints: Vec<Point2f>,
}

pub struct Stage2Result {
    pub pred_pts: Vec<Point2f>,
    pub rectified: core::Mat,
}

pub struct TPSTransformer {
    l_inv: Array2<f32>,
    q_grid: Array2<f32>,
    out_size: i32,
    num_pts: usize,
}

impl TPSTransformer {
    pub fn new(ctrl_pts: &[Point2f], out_size: i32) -> Result<Self> {
        let n = ctrl_pts.len();
        let reg = 1e-6f32;
        let mut l = Array2::<f32>::zeros((n + 3, n + 3));
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    l[[i, j]] = reg;
                } else {
                    let dx = ctrl_pts[i].x - ctrl_pts[j].x;
                    let dy = ctrl_pts[i].y - ctrl_pts[j].y;
                    let r2 = dx * dx + dy * dy;
                    l[[i, j]] = r2 * (r2 + 1e-12).ln();
                }
            }
        }
        for i in 0..n {
            l[[i, n]] = 1.0;
            l[[i, n + 1]] = ctrl_pts[i].x;
            l[[i, n + 2]] = ctrl_pts[i].y;
            l[[n, i]] = 1.0;
            l[[n + 1, i]] = ctrl_pts[i].x;
            l[[n + 2, i]] = ctrl_pts[i].y;
        }
        let mut l_mat = core::Mat::new_size_with_default(
            Size::new((n + 3) as i32, (n + 3) as i32),
            core::CV_32F,
            Scalar::default(),
        )?;
        for r in 0..(n + 3) {
            for c in 0..(n + 3) {
                *l_mat.at_2d_mut::<f32>(r as i32, c as i32)? = l[[r, c]];
            }
        }
        let mut l_inv_mat = core::Mat::default();
        core::invert(&l_mat, &mut l_inv_mat, core::DECOMP_LU)?;
        let mut l_inv = Array2::<f32>::zeros((n + 3, n + 3));
        for r in 0..(n + 3) {
            for c in 0..(n + 3) {
                l_inv[[r, c]] = *l_inv_mat.at_2d::<f32>(r as i32, c as i32)?;
            }
        }
        let h = out_size;
        let w = out_size;
        let mut q_grid = Array2::<f32>::zeros(((h * w) as usize, n + 3));
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                let gx = (x as f32 / (w as f32 - 1.0)) * 2.0 - 1.0;
                let gy = (y as f32 / (h as f32 - 1.0)) * 2.0 - 1.0;
                for i in 0..n {
                    let dx = gx - ctrl_pts[i].x;
                    let dy = gy - ctrl_pts[i].y;
                    let r2 = dx * dx + dy * dy;
                    q_grid[[idx, i]] = r2 * (r2 + 1e-12).ln();
                }
                q_grid[[idx, n]] = 1.0;
                q_grid[[idx, n + 1]] = gx;
                q_grid[[idx, n + 2]] = gy;
            }
        }
        Ok(Self {
            l_inv,
            q_grid,
            out_size,
            num_pts: n,
        })
    }

    pub fn transform(
        &self,
        src: &core::Mat,
        pred_pts: &[Point2f],
        src_in_size: i32,
    ) -> Result<core::Mat> {
        let n = self.num_pts;
        let mut rhs = Array2::<f32>::zeros((n + 3, 2));
        for i in 0..n {
            rhs[[i, 0]] = pred_pts[i].x;
            rhs[[i, 1]] = pred_pts[i].y;
        }
        let theta = self.l_inv.dot(&rhs);
        let grid = self.q_grid.dot(&theta);
        let mut map_x = core::Mat::new_size_with_default(
            Size::new(self.out_size, self.out_size),
            core::CV_32F,
            Scalar::default(),
        )?;
        let mut map_y = core::Mat::new_size_with_default(
            Size::new(self.out_size, self.out_size),
            core::CV_32F,
            Scalar::default(),
        )?;
        let src_h = src.rows() as f32;
        let src_w = src.cols() as f32;
        let scale = src_in_size as f32 / src_h.max(src_w);
        let pad_x = (src_in_size as f32 - src_w * scale) / 2.0;
        let pad_y = (src_in_size as f32 - src_h * scale) / 2.0;
        for y in 0..self.out_size {
            for x in 0..self.out_size {
                let idx = (y * self.out_size + x) as usize;
                let px_norm = (grid[[idx, 0]] + 1.0) * (src_in_size as f32 - 1.0) / 2.0;
                let py_norm = (grid[[idx, 1]] + 1.0) * (src_in_size as f32 - 1.0) / 2.0;
                *map_x.at_2d_mut::<f32>(y, x)? = (px_norm - pad_x) / scale;
                *map_y.at_2d_mut::<f32>(y, x)? = (py_norm - pad_y) / scale;
            }
        }
        let mut out = core::Mat::default();
        imgproc::remap(
            src,
            &mut out,
            &map_x,
            &map_y,
            imgproc::INTER_NEAREST,
            core::BORDER_CONSTANT,
            Scalar::default(),
        )?;
        Ok(out)
    }
}

pub struct YoloDetector {
    stage1: Session,
    stage2: Session,
    tps: TPSTransformer,
}

impl YoloDetector {
    pub fn new<P1: AsRef<Path>, P2: AsRef<Path>>(s1_path: P1, s2_path: P2) -> Result<Self> {
        let cuda = CUDAExecutionProvider::default();
        let stage1 = Session::builder()
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .with_execution_providers([cuda.clone().build()])
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .commit_from_file(s1_path)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let stage2 = Session::builder()
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .with_execution_providers([cuda.build()])
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .commit_from_file(s2_path)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        let canon_pts = Self::compute_canon_pts_norm_raw()?;
        // 179个cell（Version40含边框）按约6px/cell重采样，减小颜色串扰
        let tps = TPSTransformer::new(&canon_pts, 1074)?;
        Ok(Self {
            stage1,
            stage2,
            tps,
        })
    }

    fn compute_canon_pts_norm_raw() -> Result<Vec<Point2f>> {
        let qr_mc = 177;
        let qr_bs = 4;
        let qr_br = 1;
        let m_c = |r: i32, c: i32| -> Point2f {
            Point2f::new(
                ((c + qr_br) * qr_bs + qr_bs / 2) as f32,
                ((r + qr_br) * qr_bs + qr_bs / 2) as f32,
            )
        };
        let mut base = Vec::new();
        base.push(m_c(3, 3));
        base.push(m_c(3, qr_mc - 4));
        base.push(m_c(qr_mc - 4, 3));
        let aps = [6, 30, 58, 86, 114, 142, 170];
        let f_ov = [(6, 6), (6, 170), (170, 6)];
        for &i in &aps {
            for &j in &aps {
                if f_ov.contains(&(i, j))
                    || (i <= 8 && j <= 8)
                    || (i <= 8 && j >= qr_mc - 8)
                    || (i >= qr_mc - 8 && j <= 8)
                {
                    continue;
                }
                base.push(m_c(i, j));
            }
        }
        let end = (qr_mc + 2 * qr_br) * qr_bs - 1;
        base.push(Point2f::new(0.0, 0.0));
        base.push(Point2f::new(end as f32, 0.0));
        base.push(Point2f::new(end as f32, end as f32));
        base.push(Point2f::new(0.0, end as f32));
        let src = [base[49], base[50], base[51], base[52]];
        let s = 799.0f32;
        let dst = [
            Point2f::new(0.0, 0.0),
            Point2f::new(s, 0.0),
            Point2f::new(s, s),
            Point2f::new(0.0, s),
        ];
        let h = imgproc::get_perspective_transform(
            &core::Mat::from_slice(&src)?,
            &core::Mat::from_slice(&dst)?,
            0,
        )?;
        let mut transformed = core::Vector::<Point2f>::new();
        core::perspective_transform(
            &core::Vector::<Point2f>::from_iter(base),
            &mut transformed,
            &h,
        )?;
        let half = 399.5f32;
        Ok(transformed
            .into_iter()
            .map(|p| Point2f::new((p.x / half) - 1.0, (p.y / half) - 1.0))
            .collect())
    }

    pub fn detect_stage1(&mut self, frame: &core::Mat) -> Result<Option<YoloDetection>> {
        let imgsz = 960;
        let (img, ratio, pad) = self.preprocess_letterbox(frame, imgsz)?;
        let mut inp = Array4::<f32>::zeros((1, 3, imgsz as usize, imgsz as usize));
        for y in 0..imgsz {
            for x in 0..imgsz {
                let px = img.at_2d::<core::Vec3b>(y, x)?;
                inp[[0, 0, y as usize, x as usize]] = px[2] as f32 / 255.0;
                inp[[0, 1, y as usize, x as usize]] = px[1] as f32 / 255.0;
                inp[[0, 2, y as usize, x as usize]] = px[0] as f32 / 255.0;
            }
        }
        let out = {
            let val = Value::from_array(inp).map_err(|e| anyhow::anyhow!(e.to_string()))?;
            let res = self
                .stage1
                .run(inputs![val])
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            let (sh, d) = res[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            (sh[2] as usize, d.to_vec())
        };
        let n = out.0;
        let data = out.1;
        let mut b_conf = 0.0;
        let mut b_idx = 0;
        for i in 0..n {
            let conf = data[4 * n + i];
            if conf > b_conf {
                b_conf = conf;
                b_idx = i;
            }
        }
        let cx = data[b_idx];
        let cy = data[n + b_idx];
        let w = data[2 * n + b_idx];
        let h = data[3 * n + b_idx];
        let x1 = (cx - w / 2.0 - pad.0 as f32) / ratio;
        let y1 = (cy - h / 2.0 - pad.1 as f32) / ratio;
        let x2 = (cx + w / 2.0 - pad.0 as f32) / ratio;
        let y2 = (cy + h / 2.0 - pad.1 as f32) / ratio;
        let mut kpts = Vec::new();
        for i in 0..4 {
            let kx = (data[(5 + i * 3) * n + b_idx] - pad.0 as f32) / ratio;
            let ky = (data[(5 + i * 3 + 1) * n + b_idx] - pad.1 as f32) / ratio;
            kpts.push(Point2f::new(kx, ky));
        }
        Ok(Some(YoloDetection {
            bbox: Rect::new(x1 as i32, y1 as i32, (x2 - x1) as i32, (y2 - y1) as i32),
            confidence: b_conf,
            keypoints: kpts,
        }))
    }

    pub fn detect_stage2(&mut self, cropped: &core::Mat) -> Result<Stage2Result> {
        let size = 800;
        let (img, ratio, pad) = self.preprocess_letterbox(cropped, size)?;
        let mut inp = Array4::<f32>::zeros((1, 3, size as usize, size as usize));
        for y in 0..size {
            for x in 0..size {
                let pixel = img.at_2d::<core::Vec3b>(y, x)?;
                inp[[0, 0, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
                inp[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
                inp[[0, 2, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
            }
        }
        let pts_data = {
            let val = Value::from_array(inp).map_err(|e| anyhow::anyhow!(e.to_string()))?;
            let res = self
                .stage2
                .run(inputs![val])
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            let (_, d) = res[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            d.to_vec()
        };
        let mut pred_pts_800 = Vec::new();
        for i in 0..53 {
            let half = 399.5f32;
            pred_pts_800.push(Point2f::new(
                (pts_data[i * 2] + 1.0) * half,
                (pts_data[i * 2 + 1] + 1.0) * half,
            ));
        }

        // --- 第一步：使用四个角点进行基础变换 ---
        let mut src_corners = Vec::new();
        for i in [49, 50, 51, 52] {
            let ox = (pred_pts_800[i].x - pad.0 as f32) / ratio;
            let oy = (pred_pts_800[i].y - pad.1 as f32) / ratio;
            src_corners.push(Point2f::new(ox, oy));
        }
        let s = size as f32 - 1.0;
        let dst_corners = [
            Point2f::new(0.0, 0.0),
            Point2f::new(s, 0.0),
            Point2f::new(s, s),
            Point2f::new(0.0, s),
        ];
        let h_mat = imgproc::get_perspective_transform(
            &core::Mat::from_slice(&src_corners)?,
            &core::Mat::from_slice(&dst_corners)?,
            0,
        )?;
        let mut rectified = core::Mat::default();
        imgproc::warp_perspective(
            cropped,
            &mut rectified,
            &h_mat,
            Size::new(size, size),
            imgproc::INTER_NEAREST,
            core::BORDER_CONSTANT,
            Scalar::default(),
        )?;

        // --- 第二步：探测 3 个定位中心并校准 ---
        let qr_mc = 177;
        let qr_bs = 4;
        let qr_br = 1;
        let m_c_v40 = |r: i32, c: i32| -> Point2f {
            let px = ((c + qr_br) * qr_bs + qr_bs / 2) as f32;
            let py = ((r + qr_br) * qr_bs + qr_bs / 2) as f32;
            let scale_v40 = 800.0 / 716.0;
            Point2f::new(px * scale_v40, py * scale_v40)
        };
        let standard_centers = [m_c_v40(3, 3), m_c_v40(3, qr_mc - 4), m_c_v40(qr_mc - 4, 3)];
        let mut actual_centers = Vec::new();
        for (idx, &std_center) in standard_centers.iter().enumerate() {
            if let Ok(actual) = self.detect_color_centroid(&rectified, std_center, idx) {
                actual_centers.push(actual);
            } else {
                actual_centers.push(std_center);
            }
        }
        let src_tri = core::Mat::from_slice(&actual_centers)?;
        let dst_tri = core::Mat::from_slice(&standard_centers)?;
        let affine_m = imgproc::get_affine_transform(&src_tri, &dst_tri)?;

        let mut final_rectified = core::Mat::default();
        imgproc::warp_affine(
            &rectified,
            &mut final_rectified,
            &affine_m,
            Size::new(size, size),
            imgproc::INTER_NEAREST,
            core::BORDER_CONSTANT,
            Scalar::default(),
        )?;

        // --- 第三步：针对辅助定位点 (Idx 3..48) 进行精准探测与终极 TPS 矫正 ---
        let mut h_inv = core::Mat::default();
        core::invert(&h_mat, &mut h_inv, core::DECOMP_LU)?;

        let mut affine_inv = core::Mat::default();
        imgproc::invert_affine_transform(&affine_m, &mut affine_inv)?;

        let half = 399.5f32;

        let map_rect_to_norm = |p_rect: Point2f| -> Result<Point2f> {
            let mut p_rect_vec_in = core::Vector::<Point2f>::new();
            p_rect_vec_in.push(p_rect);
            let mut p_crop_vec = core::Vector::<Point2f>::new();
            core::perspective_transform(&p_rect_vec_in, &mut p_crop_vec, &h_inv)?;
            let p_crop = p_crop_vec.get(0)?;
            let lx = p_crop.x * ratio + pad.0 as f32;
            let ly = p_crop.y * ratio + pad.1 as f32;
            Ok(Point2f::new((lx / half) - 1.0, (ly / half) - 1.0))
        };

        let map_final_to_norm = |p_final: Point2f| -> Result<Point2f> {
            let mut p_final_vec = core::Vector::<Point2f>::new();
            p_final_vec.push(p_final);
            let mut p_rect_vec = core::Vector::<Point2f>::new();
            core::transform(&p_final_vec, &mut p_rect_vec, &affine_inv)?;
            let p_rect = p_rect_vec.get(0)?;
            map_rect_to_norm(p_rect)
        };

        let map_norm_to_final = |p_norm: Point2f| -> Result<Point2f> {
            let lx = (p_norm.x + 1.0) * half;
            let ly = (p_norm.y + 1.0) * half;
            let px = (lx - pad.0 as f32) / ratio;
            let py = (ly - pad.1 as f32) / ratio;
            let mut p_crop_vec = core::Vector::<Point2f>::new();
            p_crop_vec.push(Point2f::new(px, py));
            let mut p_rect_vec = core::Vector::<Point2f>::new();
            core::perspective_transform(&p_crop_vec, &mut p_rect_vec, &h_mat)?;
            let p_rect = p_rect_vec.get(0)?;
            let mut p_rect_vec_in = core::Vector::<Point2f>::new();
            p_rect_vec_in.push(p_rect);
            let mut p_final_vec = core::Vector::<Point2f>::new();
            core::transform(&p_rect_vec_in, &mut p_final_vec, &affine_m)?;
            let p_final = p_final_vec.get(0)?;
            Ok(p_final)
        };

        let mut ultimate_pts_norm = Vec::new();
        for idx in 0..53 {
            let p_norm = Point2f::new(pts_data[idx * 2], pts_data[idx * 2 + 1]);

            if (0..=2).contains(&idx) {
                // Main finders: use detected centers in rectified image
                if let Ok(p_norm_actual) = map_rect_to_norm(actual_centers[idx]) {
                    ultimate_pts_norm.push(p_norm_actual);
                } else {
                    ultimate_pts_norm.push(p_norm);
                }
            } else if (3..=48).contains(&idx) {
                // Alignment patterns: map prediction to final_rectified and search
                if let Ok(p_final_expected) = map_norm_to_final(p_norm) {
                    if let Ok(actual_px) =
                        self.detect_alignment_pattern_center(&mut final_rectified, p_final_expected)
                    {
                        if let Ok(p_norm_actual) = map_final_to_norm(actual_px) {
                            ultimate_pts_norm.push(p_norm_actual);
                        } else {
                            ultimate_pts_norm.push(p_norm);
                        }
                    } else {
                        ultimate_pts_norm.push(p_norm);
                    }
                } else {
                    ultimate_pts_norm.push(p_norm);
                }
            } else {
                // Corners (49..=52): keep original predictions since they form h_mat
                ultimate_pts_norm.push(p_norm);
            }
        }

        if cfg!(debug_assertions) {
            let _ = opencv::imgcodecs::imwrite(
                "debug_alignment_patterns.png",
                &final_rectified,
                &core::Vector::new(),
            );
        }

        let ultimate_rectified = self.tps.transform(cropped, &ultimate_pts_norm, size)?;

        Ok(Stage2Result {
            pred_pts: ultimate_pts_norm,
            rectified: ultimate_rectified,
        })
    }

    fn detect_alignment_pattern_center(
        &self,
        img: &mut core::Mat,
        center: Point2f,
    ) -> Result<Point2f> {
        let max_search_range = 15;
        let center_x = center.x.round() as i32;
        let center_y = center.y.round() as i32;

        let cols = img.cols();
        let rows = img.rows();

        if center_x < 0 || center_x >= cols || center_y < 0 || center_y >= rows {
            return Ok(center);
        }

        let is_blue =
            |b: f32, g: f32, r: f32| -> bool { b > 100.0 && b > r + 30.0 && b > g + 30.0 };

        let mut center_is_blue = false;

        // 检查预测点是否落在蓝色上
        if let Ok(p) = img.at_2d::<core::Vec3b>(center_y, center_x) {
            let b = p[0] as f32;
            let g = p[1] as f32;
            let r = p[2] as f32;
            if is_blue(b, g, r) {
                center_is_blue = true;
            }
        }

        let mut final_x = center.x;
        let mut final_y = center.y;

        let is_white = |b: f32, g: f32, r: f32| -> bool {
            b > 120.0 && g > 120.0 && r > 120.0 && (r.max(g).max(b) - r.min(g).min(b)) < 40.0
        };

        if center_is_blue {
            // 已经在蓝色上了，进行小范围的局部质心计算以微调
            let radius = 3;
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut sum_weight = 0.0;

            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let nx = center_x + dx;
                    let ny = center_y + dy;
                    if nx >= 0
                        && nx < cols
                        && ny >= 0
                        && ny < rows
                        && let Ok(p) = img.at_2d::<core::Vec3b>(ny, nx)
                    {
                        let b = p[0] as f32;
                        let g = p[1] as f32;
                        let r = p[2] as f32;
                        let blue_val = b - r.max(g);
                        if blue_val > 10.0 {
                            sum_x += nx as f32 * blue_val;
                            sum_y += ny as f32 * blue_val;
                            sum_weight += blue_val;
                        }
                    }
                }
            }
            if sum_weight > 0.0 {
                final_x = sum_x / sum_weight;
                final_y = sum_y / sum_weight;
            }
        } else {
            // 没有落在蓝色上（可能是白色或者杂色）
            let roi_x = (center_x - max_search_range).max(0);
            let roi_y = (center_y - max_search_range).max(0);
            let roi_w = (max_search_range * 2).min(cols - roi_x);
            let roi_h = (max_search_range * 2).min(rows - roi_y);

            if roi_w > 0 && roi_h > 0 {
                let roi = core::Mat::roi(img, Rect::new(roi_x, roi_y, roi_w, roi_h))?;

                // 找到白色区域
                let mut white_mask = core::Mat::new_size_with_default(
                    Size::new(roi_w, roi_h),
                    core::CV_8U,
                    Scalar::all(0.0),
                )?;
                let mut blue_response = core::Mat::new_size_with_default(
                    Size::new(roi_w, roi_h),
                    core::CV_32F,
                    Scalar::all(0.0),
                )?;

                for y in 0..roi_h {
                    for x in 0..roi_w {
                        if let Ok(p) = roi.at_2d::<core::Vec3b>(y, x) {
                            let b = p[0] as f32;
                            let g = p[1] as f32;
                            let r = p[2] as f32;
                            if is_white(b, g, r) {
                                *white_mask.at_2d_mut::<u8>(y, x)? = 255;
                            }
                            // 即使不是白色，如果有蓝色特征也保存响应，以防蓝色区域扩张
                            if is_blue(b, g, r) {
                                *blue_response.at_2d_mut::<f32>(y, x)? = b - r.max(g);
                            }
                        }
                    }
                }

                // 通过寻找距离预测点最近的白色连通域，然后再在这个连通域周围找蓝色质心
                let mut white_contours = core::Vector::<core::Vector<core::Point>>::new();
                imgproc::find_contours(
                    &white_mask,
                    &mut white_contours,
                    imgproc::RETR_EXTERNAL,
                    imgproc::CHAIN_APPROX_SIMPLE,
                    core::Point::new(0, 0),
                )?;

                let local_center_x = center.x - roi_x as f32;
                let local_center_y = center.y - roi_y as f32;

                let mut best_white_dist = f32::MAX;
                let mut best_white_contour_idx = -1;

                for (i, contour) in white_contours.iter().enumerate() {
                    let rect = imgproc::bounding_rect(&contour)?;
                    let cx = rect.x as f32 + rect.width as f32 / 2.0;
                    let cy = rect.y as f32 + rect.height as f32 / 2.0;
                    let dist = (cx - local_center_x).powi(2) + (cy - local_center_y).powi(2);
                    if dist < best_white_dist {
                        best_white_dist = dist;
                        best_white_contour_idx = i as i32;
                    }
                }

                if best_white_contour_idx != -1 {
                    // 我们找到了预测点所在的或最近的白色连通块
                    // 因为蓝色核心一定在白色连通块内部（或非常靠近中心），我们直接在这个白色块所在的 bbox 内部及其附近寻找蓝色极大值
                    let best_contour = white_contours.get(best_white_contour_idx as usize)?;
                    let rect = imgproc::bounding_rect(&best_contour)?;

                    let mut sum_x = 0.0;
                    let mut sum_y = 0.0;
                    let mut sum_w = 0.0;

                    // 在稍微扩大一点的范围内寻找蓝点（以防蓝点完全隔绝了白点）
                    let core_x_start = (rect.x - 2).max(0);
                    let core_x_end = (rect.x + rect.width + 2).min(roi_w);
                    let core_y_start = (rect.y - 2).max(0);
                    let core_y_end = (rect.y + rect.height + 2).min(roi_h);

                    for y in core_y_start..core_y_end {
                        for x in core_x_start..core_x_end {
                            if let Ok(val) = blue_response.at_2d::<f32>(y, x)
                                && *val > 5.0
                            {
                                sum_x += x as f32 * *val;
                                sum_y += y as f32 * *val;
                                sum_w += *val;
                            }
                        }
                    }

                    if sum_w > 0.0 {
                        final_x = (sum_x / sum_w) + roi_x as f32;
                        final_y = (sum_y / sum_w) + roi_y as f32;
                    }
                }
            }
        }

        // --- 终极蓝色峰值校准 ---
        let micro_radius = 5;
        let mut sum_x_micro = 0.0;
        let mut sum_y_micro = 0.0;
        let mut sum_w_micro = 0.0;
        let mut max_blue = 0.0;

        for dy in -micro_radius..=micro_radius {
            for dx in -micro_radius..=micro_radius {
                let nx = final_x.round() as i32 + dx;
                let ny = final_y.round() as i32 + dy;
                if nx >= 0
                    && nx < cols
                    && ny >= 0
                    && ny < rows
                    && let Ok(p) = img.at_2d::<core::Vec3b>(ny, nx)
                {
                    let b = p[0] as f32;
                    let g = p[1] as f32;
                    let r = p[2] as f32;
                    let blue_val = b - r.max(g);
                    if blue_val > max_blue {
                        max_blue = blue_val;
                    }
                    let dist_sq = (nx as f32 - final_x).powi(2) + (ny as f32 - final_y).powi(2);
                    let weight = blue_val * (-dist_sq / 8.0).exp();
                    if blue_val > 0.0 {
                        sum_x_micro += nx as f32 * weight;
                        sum_y_micro += ny as f32 * weight;
                        sum_w_micro += weight;
                    }
                }
            }
        }

        if max_blue > 20.0 && sum_w_micro > 0.0 {
            final_x = sum_x_micro / sum_w_micro;
            final_y = sum_y_micro / sum_w_micro;
        }

        if cfg!(debug_assertions) {
            // Draw original center (green)
            imgproc::circle(
                img,
                core::Point::new(center.x.round() as i32, center.y.round() as i32),
                1,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                -1,
                imgproc::LINE_8,
                0,
            )?;
            // Draw detected centroid (yellow/red depending on fallback)
            let color = if center_is_blue {
                Scalar::new(0.0, 255.0, 255.0, 0.0) // Yellow = micro adjusted
            } else {
                Scalar::new(0.0, 0.0, 255.0, 0.0) // Red = fallback search
            };
            imgproc::circle(
                img,
                core::Point::new(final_x.round() as i32, final_y.round() as i32),
                1,
                color,
                -1,
                imgproc::LINE_8,
                0,
            )?;
        }

        Ok(Point2f::new(final_x, final_y))
    }

    fn detect_color_centroid(
        &self,
        img: &core::Mat,
        center: Point2f,
        idx: usize,
    ) -> Result<Point2f> {
        let search_range = 30;
        let roi_x = (center.x as i32 - search_range).max(0);
        let roi_y = (center.y as i32 - search_range).max(0);
        let roi_w = (search_range * 2).min(img.cols() - roi_x);
        let roi_h = (search_range * 2).min(img.rows() - roi_y);
        if roi_w <= 10 || roi_h <= 10 {
            return Ok(center);
        }

        let roi = core::Mat::roi(img, Rect::new(roi_x, roi_y, roi_w, roi_h))?;
        let seed_x = (center.x - roi_x as f32).clamp(0.0, (roi_w - 1) as f32);
        let seed_y = (center.y - roi_y as f32).clamp(0.0, (roi_h - 1) as f32);

        let mut samples: Vec<(f32, f32, f32)> = Vec::new();
        let mut best_resp = -1e9f32;
        let mut best_xy = (seed_x, seed_y);

        for y in 0..roi_h {
            for x in 0..roi_w {
                let p = roi.at_2d::<core::Vec3b>(y, x)?;
                let b = p[0] as f32;
                let g = p[1] as f32;
                let r = p[2] as f32;

                let (target, o1, o2) = match idx {
                    0 => (r, g, b),
                    1 => (g, r, b),
                    _ => (b, r, g),
                };
                let resp = target - 0.5 * o1 - 0.5 * o2;
                let dom = target - o1.max(o2);

                if resp > 0.0 && dom > 8.0 {
                    samples.push((x as f32, y as f32, resp));
                    if resp > best_resp {
                        best_resp = resp;
                        best_xy = (x as f32, y as f32);
                    }
                }
            }
        }

        if samples.len() < 20 {
            return Ok(center);
        }

        // 使用固定迭代次数的 2-means，避免任何不收敛重试导致的重复运行。
        let mut c0 = (seed_x, seed_y);
        let mut c1 = best_xy;
        const KM_ITERS: usize = 8;

        for _ in 0..KM_ITERS {
            let (mut s0x, mut s0y, mut w0) = (0.0f32, 0.0f32, 0.0f32);
            let (mut s1x, mut s1y, mut w1) = (0.0f32, 0.0f32, 0.0f32);

            for &(x, y, w) in &samples {
                let d0 = (x - c0.0).powi(2) + (y - c0.1).powi(2);
                let d1 = (x - c1.0).powi(2) + (y - c1.1).powi(2);
                if d0 <= d1 {
                    s0x += x * w;
                    s0y += y * w;
                    w0 += w;
                } else {
                    s1x += x * w;
                    s1y += y * w;
                    w1 += w;
                }
            }

            if w0 > 1e-6 {
                c0 = (s0x / w0, s0y / w0);
            }
            if w1 > 1e-6 {
                c1 = (s1x / w1, s1y / w1);
            }
        }

        let dist0 = (c0.0 - seed_x).powi(2) + (c0.1 - seed_y).powi(2);
        let dist1 = (c1.0 - seed_x).powi(2) + (c1.1 - seed_y).powi(2);
        let chosen = if dist0 <= dist1 { c0 } else { c1 };

        Ok(Point2f::new(
            chosen.0 + roi_x as f32,
            chosen.1 + roi_y as f32,
        ))
    }

    fn preprocess_letterbox(
        &self,
        frame: &core::Mat,
        size: i32,
    ) -> Result<(core::Mat, f32, (i32, i32))> {
        let h = frame.rows();
        let w = frame.cols();
        let scale = size as f32 / h.max(w) as f32;
        let nw = (w as f32 * scale).round() as i32;
        let nh = (h as f32 * scale).round() as i32;
        let mut resized = core::Mat::default();
        imgproc::resize(
            frame,
            &mut resized,
            Size::new(nw, nh),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;
        let mut canvas = core::Mat::new_size_with_default(
            Size::new(size, size),
            core::CV_8UC3,
            Scalar::new(0.0, 0.0, 0.0, 0.0),
        )?;
        let dx = (size - nw) / 2;
        let dy = (size - nh) / 2;
        let mut roi = core::Mat::roi_mut(&mut canvas, Rect::new(dx, dy, nw, nh))?;
        resized.copy_to(&mut roi)?;
        Ok((canvas, scale, (dx, dy)))
    }

    pub fn crop_for_stage2(frame: &core::Mat, detection: &YoloDetection) -> Result<core::Mat> {
        let x1 = detection.bbox.x as f32;
        let y1 = detection.bbox.y as f32;
        let x2 = (detection.bbox.x + detection.bbox.width) as f32;
        let y2 = (detection.bbox.y + detection.bbox.height) as f32;
        let img_w = frame.cols() as f32;
        let img_h = frame.rows() as f32;
        let bw = (x2 - x1).max(1.0);
        let bh = (y2 - y1).max(1.0);
        let side = bw.max(bh);
        let cx = 0.5 * (x1 + x2);
        let cy = 0.5 * (y1 + y2);
        let mut nx1 = cx - side * 0.5;
        let mut ny1 = cy - side * 0.5;
        let mut nx2 = cx + side * 0.5;
        let mut ny2 = cy + side * 0.5;
        if nx1 < 0.0 {
            nx2 -= nx1;
            nx1 = 0.0;
        }
        if ny1 < 0.0 {
            ny2 -= ny1;
            ny1 = 0.0;
        }
        if nx2 > img_w {
            nx1 -= nx2 - img_w;
            nx2 = img_w;
        }
        if ny2 > img_h {
            ny1 -= ny2 - img_h;
            ny2 = img_h;
        }
        let pad = 20.0f32;
        let sx1 = (nx1 - pad).max(0.0).round() as i32;
        let sy1 = (ny1 - pad).max(0.0).round() as i32;
        let sx2 = (nx2 + pad).min(img_w).round() as i32;
        let sy2 = (ny2 + pad).min(img_h).round() as i32;
        let rect = Rect::new(sx1, sy1, sx2 - sx1, sy2 - sy1);
        let mut cropped = core::Mat::default();
        if rect.width > 0 && rect.height > 0 {
            let roi = core::Mat::roi(frame, rect)?;
            roi.copy_to(&mut cropped)?;
        }
        Ok(cropped)
    }

    pub fn visualize_grid(rectified: &core::Mat, version: i32, box_size: i32) -> Result<core::Mat> {
        let n = (version - 1) * 4 + 21 + 2;
        let total = n * box_size;
        let mut visual = core::Mat::default();
        imgproc::resize(
            rectified,
            &mut visual,
            Size::new(total, total),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;
        for i in 0..=n {
            let p = i * box_size;
            imgproc::line(
                &mut visual,
                core::Point::new(p, 0),
                core::Point::new(p, total),
                Scalar::new(0.0, 0.0, 0.0, 0.0),
                1,
                imgproc::LINE_8,
                0,
            )?;
            imgproc::line(
                &mut visual,
                core::Point::new(0, p),
                core::Point::new(total, p),
                Scalar::new(0.0, 0.0, 0.0, 0.0),
                1,
                imgproc::LINE_8,
                0,
            )?;
        }
        Ok(visual)
    }
}
