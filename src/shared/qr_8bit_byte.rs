pub struct QR8bitByte {
    pub data: Vec<u8>,
}

impl QR8bitByte {
    pub fn new(data: &[u8]) -> Self {
        QR8bitByte {
            data: data.to_vec(),
        }
    }

    pub fn get_length(&self) -> usize {
        self.data.len()
    }

    pub fn write(&self, buffer: &mut crate::shared::qr_bit_buffer::BitBuffer) {
        for &byte in &self.data {
            buffer.put(byte as i32, 8);
        }
    }
}
