use anyhow::Result;
use opencv::{
    calib3d,
    core::{self, Point2f, Rect, Scalar, Size},
    dnn_superres, imgproc,
    prelude::*,
};

pub struct QRDecoderStandardizer {
    templates: Vec<FinderTemplate>,
    sr: Option<std::sync::Mutex<dnn_superres::DnnSuperResImpl>>,
}

struct FinderTemplate {
    size: i32,
    tl: core::Mat,
    tr: core::Mat,
    bl: core::Mat,
    alignment: core::Mat,
}

impl QRDecoderStandardizer {
    pub fn new() -> Result<Self> {
        let mut sr = None;
        // 尝试加载 EDSR 模型以进行 GAN/超分辨率修复
        if std::path::Path::new("EDSR_x4.pb").exists() {
            let mut model = dnn_superres::DnnSuperResImpl::new("edsr", 4)?;
            model.read_model("EDSR_x4.pb")?;
            model.set_model("edsr", 4)?;
            sr = Some(std::sync::Mutex::new(model));
        }

        let mut decoder = Self {
            templates: Vec::new(),
            sr,
        };
        decoder.generate_templates()?;
        Ok(decoder)
    }

    fn generate_templates(&mut self) -> Result<()> {
        // 生成不同尺寸的模板 (从 120x120 到 8x8)
        for size in (8..=120).rev().step_by(4) {
            let mut scale_templates = FinderTemplate {
                size,
                tl: core::Mat::default(),
                tr: core::Mat::default(),
                bl: core::Mat::default(),
                alignment: core::Mat::default(),
            };

            let m = (size as f32 / 7.0).round().max(1.0) as i32;
            let m2 = 2 * m;

            // BGR 颜色
            let colors = [
                ("TL", Scalar::new(0.0, 0.0, 255.0, 0.0)), // 红
                ("TR", Scalar::new(0.0, 255.0, 0.0, 0.0)), // 绿
                ("BL", Scalar::new(255.0, 0.0, 0.0, 0.0)), // 蓝
            ];

            for (name, color) in colors {
                let mut tpl = core::Mat::new_size_with_default(
                    Size::new(size, size),
                    core::CV_8UC3,
                    Scalar::new(255.0, 255.0, 255.0, 0.0),
                )?;

                // 1. 外层 7x7
                imgproc::rectangle(
                    &mut tpl,
                    Rect::new(0, 0, size, size),
                    color,
                    -1,
                    imgproc::LINE_8,
                    0,
                )?;
                // 2. 内部 5x5 (白色)
                imgproc::rectangle(
                    &mut tpl,
                    Rect::new(m, m, size - 2 * m, size - 2 * m),
                    Scalar::new(255.0, 255.0, 255.0, 0.0),
                    -1,
                    imgproc::LINE_8,
                    0,
                )?;
                // 3. 最中心 3x3
                imgproc::rectangle(
                    &mut tpl,
                    Rect::new(m2, m2, size - 2 * m2, size - 2 * m2),
                    color,
                    -1,
                    imgproc::LINE_8,
                    0,
                )?;

                match name {
                    "TL" => scale_templates.tl = tpl,
                    "TR" => scale_templates.tr = tpl,
                    "BL" => scale_templates.bl = tpl,
                    _ => unreachable!(),
                }
            }

            // 对齐标记模板 (5x5 模块)
            let a_size = (size as f32 * 5.0 / 7.0).round() as i32;
            let mut align_tpl = core::Mat::new_size_with_default(
                Size::new(a_size, a_size),
                core::CV_8UC3,
                Scalar::new(255.0, 255.0, 255.0, 0.0),
            )?;
            let am = (a_size as f32 / 5.0).round().max(1.0) as i32;
            let blue = Scalar::new(255.0, 0.0, 0.0, 0.0);

            imgproc::rectangle(
                &mut align_tpl,
                Rect::new(0, 0, a_size, a_size),
                blue,
                -1,
                imgproc::LINE_8,
                0,
            )?;
            imgproc::rectangle(
                &mut align_tpl,
                Rect::new(am, am, a_size - 2 * am, a_size - 2 * am),
                Scalar::new(255.0, 255.0, 255.0, 0.0),
                -1,
                imgproc::LINE_8,
                0,
            )?;
            imgproc::rectangle(
                &mut align_tpl,
                Rect::new(2 * am, 2 * am, a_size - 4 * am, a_size - 4 * am),
                blue,
                -1,
                imgproc::LINE_8,
                0,
            )?;
            scale_templates.alignment = align_tpl;

            self.templates.push(scale_templates);
        }
        Ok(())
    }

    pub fn find_finder_patterns(
        &self,
        frame: &core::Mat,
    ) -> Result<Option<(Point2f, Point2f, Point2f)>> {
        let mut best_overall_val = -1.0;
        let mut best_pts = None;

        for template in &self.templates {
            if frame.cols() < template.size || frame.rows() < template.size {
                continue;
            }

            let mut res_tl = core::Mat::default();
            let mut res_tr = core::Mat::default();
            let mut res_bl = core::Mat::default();

            imgproc::match_template(
                frame,
                &template.tl,
                &mut res_tl,
                imgproc::TM_CCOEFF_NORMED,
                &core::no_array(),
            )?;
            imgproc::match_template(
                frame,
                &template.tr,
                &mut res_tr,
                imgproc::TM_CCOEFF_NORMED,
                &core::no_array(),
            )?;
            imgproc::match_template(
                frame,
                &template.bl,
                &mut res_bl,
                imgproc::TM_CCOEFF_NORMED,
                &core::no_array(),
            )?;

            let mut max_val_tl = 0.0;
            let mut max_loc_tl = core::Point::default();
            core::min_max_loc(
                &res_tl,
                None,
                Some(&mut max_val_tl),
                None,
                Some(&mut max_loc_tl),
                &core::no_array(),
            )?;

            let mut max_val_tr = 0.0;
            let mut max_loc_tr = core::Point::default();
            core::min_max_loc(
                &res_tr,
                None,
                Some(&mut max_val_tr),
                None,
                Some(&mut max_loc_tr),
                &core::no_array(),
            )?;

            let mut max_val_bl = 0.0;
            let mut max_loc_bl = core::Point::default();
            core::min_max_loc(
                &res_bl,
                None,
                Some(&mut max_val_bl),
                None,
                Some(&mut max_loc_bl),
                &core::no_array(),
            )?;

            let score = max_val_tl + max_val_tr + max_val_bl;

            if score > best_overall_val && max_val_tl > 0.4 && max_val_tr > 0.4 && max_val_bl > 0.4
            {
                // 亚像素精细化 (Parabolic Interpolation)
                let refine_point = |res: &core::Mat, p: core::Point| -> Point2f {
                    let mut px = p.x as f32;
                    let mut py = p.y as f32;

                    if p.x > 0 && p.x < res.cols() - 1 {
                        let v_m1 = *res.at_2d::<f32>(p.y, p.x - 1).unwrap_or(&0.0);
                        let v_0 = *res.at_2d::<f32>(p.y, p.x).unwrap_or(&0.0);
                        let v_p1 = *res.at_2d::<f32>(p.y, p.x + 1).unwrap_or(&0.0);
                        let denom = 2.0 * v_0 - v_m1 - v_p1;
                        if denom.abs() > 1e-5 {
                            px += (v_p1 - v_m1) / (2.0 * denom);
                        }
                    }

                    if p.y > 0 && p.y < res.rows() - 1 {
                        let v_m1 = *res.at_2d::<f32>(p.y - 1, p.x).unwrap_or(&0.0);
                        let v_0 = *res.at_2d::<f32>(p.y, p.x).unwrap_or(&0.0);
                        let v_p1 = *res.at_2d::<f32>(p.y + 1, p.x).unwrap_or(&0.0);
                        let denom = 2.0 * v_0 - v_m1 - v_p1;
                        if denom.abs() > 1e-5 {
                            py += (v_p1 - v_m1) / (2.0 * denom);
                        }
                    }
                    Point2f::new(px, py)
                };

                let sub_tl = refine_point(&res_tl, max_loc_tl);
                let sub_tr = refine_point(&res_tr, max_loc_tr);
                let sub_bl = refine_point(&res_bl, max_loc_bl);

                let offset = template.size as f32 / 2.0;
                let pt_tl = Point2f::new(sub_tl.x + offset, sub_tl.y + offset);
                let pt_tr = Point2f::new(sub_tr.x + offset, sub_tr.y + offset);
                let pt_bl = Point2f::new(sub_bl.x + offset, sub_bl.y + offset);

                best_overall_val = score;
                best_pts = Some((pt_tl, pt_tr, pt_bl));

                if score > 2.7 {
                    break;
                }
            }
        }

        Ok(best_pts)
    }

    /// 根据二维码版本获取定位点的标准坐标（模块坐标）
    pub fn get_standard_finder_centers(version: i32) -> (Point2f, Point2f, Point2f) {
        let n = (version - 1) * 4 + 21;
        let n_f = n as f32;

        // 每个定位块是 7x7 模块，中心在 3.5
        let tl = Point2f::new(3.5, 3.5);
        let tr = Point2f::new(n_f - 3.5, 3.5);
        let bl = Point2f::new(3.5, n_f - 3.5);

        (tl, tr, bl)
    }

    pub fn get_warped_frame(
        &self,
        frame: &core::Mat,
        version: i32,
        box_size: i32,
    ) -> Result<Option<core::Mat>> {
        let pts = self.find_finder_patterns(frame)?;
        if let Some((pt_tl, pt_tr, pt_bl)) = pts {
            let n = (version - 1) * 4 + 21;
            let qr_pixel_size = n * box_size;

            let (s_tl, s_tr, s_bl) = Self::get_standard_finder_centers(version);

            // 1. 初步校准 (使用三个大角推导出的 BR)
            let pt_br_guess =
                Point2f::new(pt_tr.x + pt_bl.x - pt_tl.x, pt_tr.y + pt_bl.y - pt_tl.y);
            let s_br = Point2f::new(s_tr.x + s_bl.x - s_tl.x, s_tr.y + s_bl.y - s_tl.y);

            let transform_point = |p: Point2f| -> Point2f {
                Point2f::new(p.x * box_size as f32, p.y * box_size as f32)
            };

            let dst_tl = transform_point(s_tl);
            let dst_tr = transform_point(s_tr);
            let dst_bl = transform_point(s_bl);
            let dst_br = transform_point(s_br);

            let src = [pt_tl, pt_tr, pt_br_guess, pt_bl];
            let dst = [dst_tl, dst_tr, dst_br, dst_bl];
            let m = imgproc::get_perspective_transform(
                &core::Mat::from_slice(&src)?,
                &core::Mat::from_slice(&dst)?,
                0,
            )?;

            let mut warped = core::Mat::default();
            imgproc::warp_perspective(
                frame,
                &mut warped,
                &m,
                Size::new(qr_pixel_size, qr_pixel_size),
                imgproc::INTER_CUBIC,
                core::BORDER_CONSTANT,
                Scalar::new(255.0, 255.0, 255.0, 0.0),
            )?;

            // 2. 精细校准：寻找所有的对齐标记 (Alignment Patterns)
            let pos = crate::shared::qr_code_model::get_pattern_position(version);
            let mut src_points = vec![pt_tl, pt_tr, pt_bl];
            let mut dst_points = vec![dst_tl, dst_tr, dst_bl];

            // 创建反向变换，用于将校准后的点映射回原图
            let mut inv_m = core::Mat::default();
            core::invert(&m, &mut inv_m, core::DECOMP_LU)?;

            for &py in &pos {
                for &px in &pos {
                    // 跳过三个定位大角所在的区域
                    if (py < 10 && (px < 10 || px > n - 10)) || (px < 10 && py > n - 10) {
                        continue;
                    }

                    let target_pos_module = Point2f::new(px as f32, py as f32);
                    let target_pos_px = transform_point(target_pos_module);

                    // 在预期位置附近搜索对齐标记
                    let search_size = box_size * 8;
                    let roi_rect = Rect::new(
                        (target_pos_px.x - search_size as f32 / 2.0) as i32,
                        (target_pos_px.y - search_size as f32 / 2.0) as i32,
                        search_size,
                        search_size,
                    );

                    let safe_roi = Rect::new(0, 0, warped.cols(), warped.rows()) & roi_rect;
                    if safe_roi.width > 0 && safe_roi.height > 0 {
                        let roi = core::Mat::roi(&warped, safe_roi)?;
                        let roi_mat = roi.try_clone()?;
                        if let Some(p_in_roi) = self.find_alignment_in_roi(&roi_mat, box_size)? {
                            let p_in_warped = Point2f::new(
                                safe_roi.x as f32 + p_in_roi.x,
                                safe_roi.y as f32 + p_in_roi.y,
                            );

                            // 将此点映射回原图坐标
                            let mut pts_vec = core::Vector::<Point2f>::new();
                            pts_vec.push(p_in_warped);
                            let mut src_pts_vec = core::Vector::<Point2f>::new();
                            core::perspective_transform(&pts_vec, &mut src_pts_vec, &inv_m)?;

                            src_points.push(src_pts_vec.get(0)?);
                            dst_points.push(target_pos_px);
                        }
                    }
                }
            }

            // 使用所有找准的点进行全局优化校准 (至少需要 4 个点)
            if src_points.len() >= 4 {
                let src_mat = core::Mat::from_slice(&src_points)?;
                let dst_mat = core::Mat::from_slice(&dst_points)?;

                // find_homography 配合 RANSAC 能够过滤掉误匹配的对齐标记
                let m_final = calib3d::find_homography(
                    &src_mat,
                    &dst_mat,
                    &mut core::Mat::default(),
                    calib3d::RANSAC,
                    1.5,
                )?;

                // 为了消除右边和下边的白边，我们稍微增加输出画布的大小，然后进行裁剪
                // 或者直接稍微缩放变换矩阵
                let mut warped_large = core::Mat::default();
                imgproc::warp_perspective(
                    frame,
                    &mut warped_large,
                    &m_final,
                    Size::new(qr_pixel_size, qr_pixel_size),
                    imgproc::INTER_LINEAR,
                    core::BORDER_REPLICATE,
                    Scalar::default(),
                )?;

                // 显式裁剪掉右边和底部可能存在的白边 (通常 1-2 像素即可)
                let crop_rect = Rect::new(0, 0, qr_pixel_size - 2, qr_pixel_size - 2);
                let cropped = core::Mat::roi(&warped_large, crop_rect)?;
                warped = cropped.try_clone()?;
            }

            // --- GAN/Super-Resolution 修复环节 ---
            if let Some(sr_mutex) = &self.sr {
                println!("Applying GAN-based Super Resolution for reconstruction...");
                let mut sr_upsampled = core::Mat::default();
                if let Ok(mut sr) = sr_mutex.lock() {
                    sr.upsample(&warped, &mut sr_upsampled)?;
                }
                // 回缩到原始 box_size 对应的尺寸，但保留锐化后的细节
                let mut refined = core::Mat::default();
                imgproc::resize(
                    &sr_upsampled,
                    &mut refined,
                    Size::new(qr_pixel_size, qr_pixel_size),
                    0.0,
                    0.0,
                    imgproc::INTER_AREA,
                )?;
                warped = refined;
            } else {
                // 传统锐化降噪作为 fallback
                let temp_warped = warped.try_clone()?;
                let mut blurred = core::Mat::default();
                imgproc::gaussian_blur(
                    &temp_warped,
                    &mut blurred,
                    Size::new(0, 0),
                    3.0,
                    3.0,
                    core::BORDER_DEFAULT,
                    // AlgorithmHint
                    opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
                )?;
                core::add_weighted(&temp_warped, 1.5, &blurred, -0.5, 0.0, &mut warped, -1)?;
            }

            Ok(Some(warped))
        } else {
            Ok(None)
        }
    }

    /// 在给定的 ROI 中寻找最匹配的对齐标记中心
    fn find_alignment_in_roi(&self, roi: &core::Mat, _box_size: i32) -> Result<Option<Point2f>> {
        let mut best_match_val = -1.0;
        let mut best_match_loc = core::Point::default();
        let mut best_template_size = 0;
        let mut found = false;

        for template in &self.templates {
            if roi.cols() < template.alignment.cols() || roi.rows() < template.alignment.rows() {
                continue;
            }
            let mut res = core::Mat::default();
            imgproc::match_template(
                roi,
                &template.alignment,
                &mut res,
                imgproc::TM_CCOEFF_NORMED,
                &core::no_array(),
            )?;
            let mut max_val = 0.0;
            let mut max_loc = core::Point::default();
            core::min_max_loc(
                &res,
                None,
                Some(&mut max_val),
                None,
                Some(&mut max_loc),
                &core::no_array(),
            )?;

            if max_val > 0.6 && max_val > best_match_val {
                best_match_val = max_val;
                best_match_loc = max_loc;
                best_template_size = template.alignment.cols();
                found = true;
            }
        }

        if found {
            let center_offset = best_template_size as f32 / 2.0;
            Ok(Some(Point2f::new(
                best_match_loc.x as f32 + center_offset,
                best_match_loc.y as f32 + center_offset,
            )))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::imgcodecs;

    #[test]
    fn test_finder_patterns() -> Result<()> {
        let frame = imgcodecs::imread("test.png", imgcodecs::IMREAD_COLOR)?;
        if frame.empty() {
            println!("test.png not found, skipping test.");
            return Err(anyhow::anyhow!("test.png not found"));
        }

        let decoder = QRDecoderStandardizer::new()?;
        let version = 40;
        let box_size = 4;

        if let Some(warped) = decoder.get_warped_frame(&frame, version, box_size)? {
            imgcodecs::imwrite("warped.png", &warped, &core::Vector::new())?;
            println!("Warped frame saved to warped.png");
        } else {
            panic!("Could not find finder patterns in test.png");
        }

        Ok(())
    }
}
