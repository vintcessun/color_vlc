mod decoder_bridge;

use std::env;
use std::path::Path;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: decoder <video> <out.bin> <vout.bin>");
        process::exit(1);
    }

    let video = Path::new(&args[1]);
    let out_bin = Path::new(&args[2]);
    let vout_bin = Path::new(&args[3]);
    let workspace = Path::new(".");

    if let Err(e) = decoder_bridge::decode_video(video, out_bin, vout_bin, workspace) {
        eprintln!("Decode failed: {e}");
        process::exit(1);
    }
}
