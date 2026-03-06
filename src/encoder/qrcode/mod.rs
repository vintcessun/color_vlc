pub mod qr_code;

pub use qr_code::{QRCode, QRCodeBlock, QRCodeOptions};

pub use crate::shared::{
    qr_8bit_byte::QR8bitByte,
    qr_bit_buffer::BitBuffer,
    qr_code_model::{PATTERN_POSITION_TABLE, QRErrorCorrectLevel, QRMode, get_type_number},
    qr_math::QRMath,
    qr_polynomial::Polynomial,
    qr_rs_block::{QRRSBlock, get_rs_blocks},
    qr_util::{get_bch_digit, get_length_in_bits},
};

pub struct QRCodeNative {
    qr: QRCode,
}

impl QRCodeNative {
    pub fn new(data_a: &[u8], data_b: &[u8], correct_level: QRErrorCorrectLevel) -> Self {
        let mut qr = QRCode::with_options(QRCodeOptions {
            width: 256,
            height: 256,
            correct_level,
        });
        qr.make_code(data_a, data_b);
        QRCodeNative { qr }
    }

    pub fn module_count(&self) -> i32 {
        self.qr.get_module_count()
    }

    pub fn get_module_count(&self) -> i32 {
        self.qr.get_module_count()
    }

    pub fn get_block(&self, row: i32, col: i32) -> QRCodeBlock {
        self.qr.get_block(row, col)
    }
}

impl Default for QRCodeNative {
    fn default() -> Self {
        Self::new(b"", b"", QRErrorCorrectLevel::H)
    }
}
