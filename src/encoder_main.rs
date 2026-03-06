use color_vlc::encoder::ColorEncoder;
use color_vlc::shared::qr_code_model::QRErrorCorrectLevel;
use std::env;
use std::path::Path;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 4 {
        eprintln!("Usage: encode <in.bin> <out.mkv> <max_ms>");
        process::exit(1);
    }

    let input_bin = Path::new(&args[1]);
    let output_mkv = Path::new(&args[2]);
    let max_ms: u64 = args[3].parse().expect("max_ms must be a number");

    // 默认使用 Version 40, EC Level L, box_size 4, border 1
    let encoder = ColorEncoder::new(40, QRErrorCorrectLevel::L, 4, 1);

    if let Err(e) = encoder.encode(input_bin, output_mkv, max_ms) {
        eprintln!("Error during encoding: {}", e);
        process::exit(1);
    }
}
