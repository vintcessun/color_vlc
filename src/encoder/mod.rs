pub mod qrcode;

use anyhow::{Result, anyhow};
use ffmpeg_next as ffmpeg;
use ndarray::Array3;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::encoder::qrcode::{QRCode, QRCodeBlock};
use crate::shared::qr_code_model::QRErrorCorrectLevel;

pub struct ColorEncoder {
    pub version: i32,
    pub error_correction: QRErrorCorrectLevel,
    pub box_size: i32,
    pub border: i32,
    capacity_table: HashMap<String, i32>,
}

impl ColorEncoder {
    pub fn new(
        version: i32,
        error_correction: QRErrorCorrectLevel,
        box_size: i32,
        border: i32,
    ) -> Self {
        let mut capacity_table = HashMap::new();

        let data = vec![
            (1, 17, 14, 11, 7),
            (2, 32, 26, 20, 14),
            (3, 53, 42, 32, 24),
            (4, 78, 62, 46, 34),
            (5, 106, 84, 60, 44),
            (6, 134, 106, 74, 58),
            (7, 154, 122, 86, 64),
            (8, 192, 152, 108, 84),
            (9, 230, 180, 130, 98),
            (10, 271, 213, 151, 119),
            (11, 321, 251, 177, 137),
            (12, 367, 287, 203, 155),
            (13, 425, 331, 241, 177),
            (14, 458, 362, 258, 194),
            (15, 520, 412, 292, 220),
            (16, 586, 450, 322, 250),
            (17, 644, 504, 364, 280),
            (18, 718, 560, 394, 310),
            (19, 792, 624, 442, 338),
            (20, 858, 666, 482, 382),
            (21, 929, 711, 509, 403),
            (22, 1003, 779, 565, 439),
            (23, 1091, 857, 611, 461),
            (24, 1171, 911, 661, 511),
            (25, 1273, 997, 715, 535),
            (26, 1367, 1059, 751, 593),
            (27, 1465, 1125, 805, 625),
            (28, 1528, 1190, 868, 658),
            (29, 1628, 1264, 908, 698),
            (30, 1732, 1370, 982, 742),
            (31, 1840, 1452, 1030, 790),
            (32, 1952, 1538, 1112, 842),
            (33, 2068, 1628, 1168, 898),
            (34, 2188, 1722, 1228, 958),
            (35, 2303, 1809, 1283, 983),
            (36, 2431, 1911, 1351, 1051),
            (37, 2563, 1989, 1423, 1093),
            (38, 2699, 2099, 1499, 1139),
            (39, 2809, 2213, 1579, 1219),
            (40, 2953, 2331, 1663, 1273),
        ];

        for (v, l, m, q, h) in data {
            capacity_table.insert(format!("{}_L", v), l);
            capacity_table.insert(format!("{}_M", v), m);
            capacity_table.insert(format!("{}_Q", v), q);
            capacity_table.insert(format!("{}_H", v), h);
        }

        Self {
            version,
            error_correction,
            box_size,
            border,
            capacity_table,
        }
    }

    pub fn get_ec_level_str(&self) -> &str {
        match self.error_correction {
            QRErrorCorrectLevel::L => "L",
            QRErrorCorrectLevel::M => "M",
            QRErrorCorrectLevel::Q => "Q",
            QRErrorCorrectLevel::H => "H",
        }
    }

    pub fn get_dynamic_chunk_size(&self) -> usize {
        let key = format!("{}_{}", self.version, self.get_ec_level_str());
        let max_bytes = *self.capacity_table.get(&key).unwrap_or(&1732) as usize;

        max_bytes.saturating_sub(15)
    }

    fn create_color_frame(&self, data_a: &[u8], data_b: &[u8]) -> Array3<u8> {
        let mut qr = QRCode::new();
        qr.type_number = self.version;
        qr.options.correct_level = self.error_correction;
        qr.make_code(data_a, data_b);

        let modules = qr.get_module_count();
        let size = (modules + 2 * self.border) * self.box_size;
        let mut frame = Array3::from_elem((size as usize, size as usize, 3), 255u8);

        for r in 0..modules {
            for c in 0..modules {
                let block = qr.get_block(r, c);
                let color_rgb = match block {
                    QRCodeBlock::Red => [255, 0, 0],
                    QRCodeBlock::Green => [0, 255, 0],
                    QRCodeBlock::Blue => [0, 0, 255],
                    QRCodeBlock::White => [255, 255, 255],
                };

                let row_start = (r + self.border) * self.box_size;
                let col_start = (c + self.border) * self.box_size;

                for dr in 0..self.box_size {
                    for dc in 0..self.box_size {
                        let y = (row_start + dr) as usize;
                        let x = (col_start + dc) as usize;
                        frame[[y, x, 0]] = color_rgb[0];
                        frame[[y, x, 1]] = color_rgb[1];
                        frame[[y, x, 2]] = color_rgb[2];
                    }
                }
            }
        }

        frame
    }

    pub fn encode(&self, input_bin: &Path, output_mkv: &Path, max_ms: u64) -> Result<()> {
        let mut f = File::open(input_bin)?;
        let mut data = Vec::new();
        f.read_to_end(&mut data)?;

        let chunk_size = self.get_dynamic_chunk_size();
        let chunks: Vec<&[u8]> = data.chunks(chunk_size).collect();

        let mut pairs = Vec::new();
        for i in (0..chunks.len()).step_by(2) {
            let a = chunks[i];
            let b = if i + 1 < chunks.len() {
                chunks[i + 1]
            } else {
                &[]
            };
            pairs.push((a, b));
        }

        let total_frames = pairs.len();
        let mut fps = 30.0;
        if (total_frames as f64 / fps) * 1000.0 > max_ms as f64 {
            fps = (total_frames as f64 / (max_ms as f64 / 1000.0)).ceil();
        }

        println!(
            "Encoding {} bytes into {} frames at {} FPS...",
            data.len(),
            total_frames,
            fps
        );

        // 使用 ffmpeg-next
        ffmpeg::init()?;

        let modules = self.version * 4 + 17;
        let img_size = ((modules + 2 * self.border) * self.box_size) as u32;

        let mut octx = ffmpeg::format::output(&output_mkv)
            .map_err(|e| anyhow!("Could not create output context: {}", e))?;

        let encoder_codec =
            ffmpeg::encoder::find_by_name("ffv1").ok_or_else(|| anyhow!("Codec FFV1 not found"))?;

        let mut encoder_ctx = ffmpeg::codec::context::Context::new_with_codec(encoder_codec);

        if octx
            .format()
            .flags()
            .contains(ffmpeg::format::flag::Flags::GLOBAL_HEADER)
        {
            encoder_ctx.set_flags(ffmpeg::codec::flag::Flags::GLOBAL_HEADER);
        }

        let mut stream = octx.add_stream(encoder_codec)?;
        let stream_index = stream.index();

        let mut video_ctx = encoder_ctx.encoder().video()?;

        video_ctx.set_width(img_size);
        video_ctx.set_height(img_size);
        // FFV1 不支持 rgb24, 我们使用 yuv444p
        video_ctx.set_format(ffmpeg::format::Pixel::YUV444P);
        video_ctx.set_time_base(ffmpeg::Rational::new(1, fps as i32));

        let mut encoder = video_ctx.open()?;
        stream.set_parameters(&encoder);
        stream.set_time_base(ffmpeg::Rational::new(1, fps as i32));
        stream.set_avg_frame_rate(ffmpeg::Rational::new(fps as i32, 1));

        octx.write_header()?;

        // 初始化 SwsContext 用于颜色空间转换 (RGB24 -> YUV444P)
        let mut sws = ffmpeg::software::scaling::context::Context::get(
            ffmpeg::format::Pixel::RGB24,
            img_size,
            img_size,
            ffmpeg::format::Pixel::YUV444P,
            img_size,
            img_size,
            ffmpeg::software::scaling::flag::Flags::BILINEAR,
        )?;

        for (position, (idx, (a, b))) in (0_i64..).zip(pairs.iter().enumerate()) {
            let total_chunks = chunks.len() as u32;

            let mut payload_a = Vec::with_capacity(10 + a.len());
            payload_a.extend_from_slice(&((idx * 2) as u32).to_be_bytes());
            payload_a.extend_from_slice(&total_chunks.to_be_bytes());
            payload_a.extend_from_slice(&(a.len() as u16).to_be_bytes());
            payload_a.extend_from_slice(a);

            let mut payload_b = Vec::with_capacity(10 + b.len());
            payload_b.extend_from_slice(&((idx * 2 + 1) as u32).to_be_bytes());
            payload_b.extend_from_slice(&total_chunks.to_be_bytes());
            payload_b.extend_from_slice(&(b.len() as u16).to_be_bytes());
            payload_b.extend_from_slice(b);

            let frame_data = self.create_color_frame(&payload_a, &payload_b);

            // 1. 创建原始 RGB24 frame
            let mut rgb_frame =
                ffmpeg::util::frame::Video::new(ffmpeg::format::Pixel::RGB24, img_size, img_size);

            {
                let stride = rgb_frame.stride(0);
                let dest = rgb_frame.data_mut(0);
                for y in 0..img_size as usize {
                    let start = y * stride;
                    let end = start + (img_size as usize * 3);
                    let row = &frame_data.slice(ndarray::s![y, .., ..]);
                    let row_flat = row.as_slice().unwrap();
                    dest[start..end].copy_from_slice(row_flat);
                }
            }

            // 2. 创建目标 YUV444P frame
            let mut yuv_frame =
                ffmpeg::util::frame::Video::new(ffmpeg::format::Pixel::YUV444P, img_size, img_size);
            yuv_frame.set_pts(Some(position));

            // 3. 执行转换
            sws.run(&rgb_frame, &mut yuv_frame)?;

            // 4. 编码并写入
            encoder.send_frame(&yuv_frame)?;
            receive_and_write_packets(&mut encoder, &mut octx, stream_index)?;
        }

        // Flush
        encoder.send_eof()?;
        receive_and_write_packets(&mut encoder, &mut octx, stream_index)?;

        octx.write_trailer()?;

        println!("Successfully generated {:?}", output_mkv);

        Ok(())
    }
}

fn receive_and_write_packets(
    encoder: &mut ffmpeg::codec::encoder::video::Video,
    octx: &mut ffmpeg::format::context::Output,
    stream_index: usize,
) -> Result<()> {
    let mut packet = ffmpeg::codec::packet::Packet::empty();
    while encoder.receive_packet(&mut packet).is_ok() {
        packet.set_stream(stream_index);
        packet.rescale_ts(
            encoder.time_base(),
            octx.stream(stream_index).unwrap().time_base(),
        );
        packet.write_interleaved(octx)?;
    }
    Ok(())
}
