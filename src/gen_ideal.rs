use color_vlc::encoder::qrcode::QRCode;
use color_vlc::get_encoder;
use color_vlc::shared::QRCodeBlock;
use color_vlc::shared::qr_code_model::QRErrorCorrectLevel;
use image::{Rgb, RgbImage};
use std::env;
use std::path::Path;

fn parse_ec_level(s: &str) -> QRErrorCorrectLevel {
    match s.to_ascii_uppercase().as_str() {
        "L" => QRErrorCorrectLevel::L,
        "Q" => QRErrorCorrectLevel::Q,
        "H" => QRErrorCorrectLevel::H,
        _ => QRErrorCorrectLevel::M,
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: gen_ideal <out.png> [payload_len] [ec_level]");
        eprintln!("Example: gen_ideal test_qr40_color.png 2300 M");
        return Ok(());
    }

    let out_path = Path::new(&args[1]);
    let payload_len: usize = if args.len() >= 3 {
        args[2].parse()?
    } else {
        2300
    };

    let encoder = get_encoder();
    let ec_level = if args.len() >= 4 {
        parse_ec_level(&args[3])
    } else {
        encoder.error_correction
    };

    let mut qr = QRCode::new();
    qr.type_number = encoder.version;
    qr.options.correct_level = ec_level;

    // Build deterministic payloads using the same header format as encoder stream chunks.
    let data_a: Vec<u8> = (0..payload_len).map(|i| (i % 251) as u8).collect();
    let data_b: Vec<u8> = (0..payload_len)
        .map(|i| ((i * 3 + 17) % 251) as u8)
        .collect();

    let mut payload_a = Vec::with_capacity(10 + data_a.len());
    payload_a.extend_from_slice(&0u32.to_be_bytes());
    payload_a.extend_from_slice(&2u32.to_be_bytes());
    payload_a.extend_from_slice(&(data_a.len() as u16).to_be_bytes());
    payload_a.extend_from_slice(&data_a);

    let mut payload_b = Vec::with_capacity(10 + data_b.len());
    payload_b.extend_from_slice(&1u32.to_be_bytes());
    payload_b.extend_from_slice(&2u32.to_be_bytes());
    payload_b.extend_from_slice(&(data_b.len() as u16).to_be_bytes());
    payload_b.extend_from_slice(&data_b);

    qr.make_code(&payload_a, &payload_b);

    let module_count = qr.get_module_count();
    let box_size = encoder.box_size;
    let border = encoder.border;
    let img_size = (module_count + 2 * border) * box_size;

    let mut img = RgbImage::new(img_size as u32, img_size as u32);
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

    img.save(out_path)?;
    println!(
        "Saved ideal color QR to {} (version={}, ec={:?}, size={}x{})",
        out_path.display(),
        encoder.version,
        ec_level,
        img_size,
        img_size
    );

    Ok(())
}
