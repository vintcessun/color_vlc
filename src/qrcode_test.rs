use color_vlc::decoder::qrcode::decode_color_blocks;
use color_vlc::encoder::qrcode::{QRCode, QRCodeOptions, QRErrorCorrectLevel};

fn main() {
    println!("Starting QRCode encoder/decoder consistency test...");

    // 1. 准备测试数据
    let data_a = b"Hello Stream A";
    let data_b = b"Hello Stream B";

    // 2. 生成二维码 (彩色模式)
    let mut qr = QRCode::with_options(QRCodeOptions {
        width: 256,
        height: 256,
        correct_level: QRErrorCorrectLevel::H,
    });
    qr.type_number = 4;
    qr.make_code(data_a, data_b);

    let count = qr.get_module_count();
    println!(
        "Generated QR code version {}, size {}x{}",
        qr.type_number, count, count
    );

    // 3. 提取 QRCodeBlock 矩阵
    let mut blocks = Vec::new();
    for y in 0..count {
        let mut row = Vec::new();
        for x in 0..count {
            row.push(qr.get_block(y, x));
        }
        blocks.push(row);
    }

    // 4. 使用新的解码函数解码
    match decode_color_blocks(&blocks) {
        Ok((decoded_a, decoded_b)) => {
            println!("Decoded successfully!");
            println!("Decoded A: {:?}", decoded_a);
            println!("Decoded B: {:?}", decoded_b);

            // 5. 验证一致性
            let original_a = String::from_utf8_lossy(data_a);
            let original_b = String::from_utf8_lossy(data_b);

            let mut success = true;
            if decoded_a == original_a.as_bytes() {
                println!("SUCCESS: Stream A matches!");
            } else {
                println!("FAILURE: Stream A mismatch!");
                println!("Original: {:?}", original_a);
                println!("Decoded:  {:?}", decoded_a);
                success = false;
            }

            if decoded_b == original_b.as_bytes() {
                println!("SUCCESS: Stream B matches!");
            } else {
                println!("FAILURE: Stream B mismatch!");
                println!("Original: {:?}", original_b);
                println!("Decoded:  {:?}", decoded_b);
                success = false;
            }

            if !success {
                std::process::exit(1);
            }
        }
        Err(e) => {
            println!("FAILED to decode: {}", e);
            std::process::exit(1);
        }
    }
}
