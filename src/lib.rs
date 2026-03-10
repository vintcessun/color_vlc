pub mod decoder;
pub mod encoder;
pub mod shared;

pub const VERSION: i32 = 40;

pub fn get_encoder() -> encoder::ColorEncoder {
    encoder::ColorEncoder::new(VERSION, shared::qr_code_model::QRErrorCorrectLevel::L, 4, 1)
}
