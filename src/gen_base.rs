use color_vlc::encoder::qrcode::{QRCode, QRCodeBlock};
use color_vlc::shared::qr_code_model::QRErrorCorrectLevel;
use image::{Rgb, RgbImage};
use rand::RngExt;
use std::env;
use std::fs;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: gen_base <count> <output_dir>");
        return Ok(());
    }

    let count: usize = args[1].parse()?;
    let output_dir = Path::new(&args[2]);
    if !output_dir.exists() {
        fs::create_dir_all(output_dir)?;
    }

    let mut rng = rand::rng();

    for i in 0..count {
        let mut qr = QRCode::new();
        qr.type_number = 40;
        qr.options.correct_level = QRErrorCorrectLevel::L;

        // Version 40 L 的容量是 2953 字节。每帧包含 payload_a (A) 和 payload_b (B)。
        // 参考 src/encoder/mod.rs: get_dynamic_chunk_size() -> max_bytes - 15
        // payload 还会包含 10 字节的头信息。
        // 所以我们模拟生成的 raw 数据长度应该在 2900 左右。
        let data_len = 2900;
        let data_a: Vec<u8> = (0..data_len).map(|_| rng.random()).collect();
        let data_b: Vec<u8> = (0..data_len).map(|_| rng.random()).collect();

        // 模拟头信息封装
        let mut payload_a = Vec::with_capacity(10 + data_a.len());
        payload_a.extend_from_slice(&(i as u32).to_be_bytes()); // idx
        payload_a.extend_from_slice(&(count as u32).to_be_bytes()); // total
        payload_a.extend_from_slice(&(data_a.len() as u16).to_be_bytes()); // len
        payload_a.extend_from_slice(&data_a);

        let mut payload_b = Vec::with_capacity(10 + data_b.len());
        payload_b.extend_from_slice(&(i as u32).to_be_bytes());
        payload_b.extend_from_slice(&(count as u32).to_be_bytes());
        payload_b.extend_from_slice(&(data_b.len() as u16).to_be_bytes());
        payload_b.extend_from_slice(&data_b);

        qr.make_code(&payload_a, &payload_b);

        let module_count = qr.get_module_count();
        let box_size = 4;
        let border = 1;
        let img_size = (module_count + 2 * border) * box_size;

        let mut img = RgbImage::new(img_size as u32, img_size as u32);
        // 背景白色
        for pixel in img.pixels_mut() {
            *pixel = Rgb([255, 255, 255]);
        }

        for r in 0..module_count {
            for c in 0..module_count {
                let block = qr.get_block(r, c);
                let color = match block {
                    QRCodeBlock::Red => Rgb([255, 0, 0]),
                    QRCodeBlock::Green => Rgb([0, 255, 0]),
                    QRCodeBlock::Blue => Rgb([0, 0, 255]),
                    QRCodeBlock::White => Rgb([255, 255, 255]),
                };

                let row_start = (r + border) * box_size;
                let col_start = (c + border) * box_size;

                for dr in 0..box_size {
                    for dc in 0..box_size {
                        img.put_pixel((col_start + dc) as u32, (row_start + dr) as u32, color);
                    }
                }
            }
        }

        let filename = format!("base_{:04}.png", i);
        img.save(output_dir.join(filename))?;

        if (i + 1) % 50 == 0 {
            println!("Generated {}/{} base images", i + 1, count);
        }
    }

    Ok(())
}
