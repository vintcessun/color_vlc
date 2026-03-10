use color_vlc::get_encoder;
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

    let encoder = get_encoder();

    if let Err(e) = encoder.encode(input_bin, output_mkv, max_ms) {
        eprintln!("Error during encoding: {}", e);
        process::exit(1);
    }
}
