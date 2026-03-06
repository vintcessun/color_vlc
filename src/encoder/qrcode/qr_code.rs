use crate::shared::{
    qr_8bit_byte::QR8bitByte,
    qr_bit_buffer::BitBuffer,
    qr_code_model::{PATTERN_POSITION_TABLE, QRErrorCorrectLevel, QRMode, get_type_number},
    qr_polynomial::Polynomial,
    qr_rs_block::get_rs_blocks,
    qr_util::{get_bch_digit, get_length_in_bits},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum QRCodeBlock {
    Red,
    Green,
    Blue,
    White,
}

#[derive(Clone)]
pub struct QRCodeOptions {
    pub width: i32,
    pub height: i32,
    pub correct_level: QRErrorCorrectLevel,
}

impl Default for QRCodeOptions {
    fn default() -> Self {
        QRCodeOptions {
            width: 256,
            height: 256,
            correct_level: QRErrorCorrectLevel::H,
        }
    }
}

pub struct QRCode {
    pub options: QRCodeOptions,
    pub type_number: i32,
    pub module_count: i32,
    pub modules: Vec<Vec<Option<QRCodeBlock>>>,
    pub data_cache_a: Option<Vec<i32>>,
    pub data_cache_b: Option<Vec<i32>>,
    pub data_list_a: Vec<QR8bitByte>,
    pub data_list_b: Vec<QR8bitByte>,
}

impl QRCode {
    pub fn new() -> Self {
        QRCode {
            options: QRCodeOptions::default(),
            type_number: 0,
            module_count: 0,
            modules: Vec::new(),
            data_cache_a: None,
            data_cache_b: None,
            data_list_a: Vec::new(),
            data_list_b: Vec::new(),
        }
    }

    pub fn with_options(options: QRCodeOptions) -> Self {
        QRCode {
            options,
            type_number: 0,
            module_count: 0,
            modules: Vec::new(),
            data_cache_a: None,
            data_cache_b: None,
            data_list_a: Vec::new(),
            data_list_b: Vec::new(),
        }
    }

    pub fn add_data(&mut self, data_a: &[u8], data_b: &[u8]) {
        self.data_list_a.push(QR8bitByte::new(data_a));
        self.data_list_b.push(QR8bitByte::new(data_b));
        self.data_cache_a = None;
        self.data_cache_b = None;
    }

    pub fn get_block(&self, row: i32, col: i32) -> QRCodeBlock {
        if row < 0 || self.module_count <= row || col < 0 || self.module_count <= col {
            panic!("getBlock: out of range");
        }
        self.modules[row as usize][col as usize].unwrap_or(QRCodeBlock::White)
    }

    pub fn get_module_count(&self) -> i32 {
        self.module_count
    }

    pub fn get_modules(&self) -> Option<&Vec<Vec<Option<QRCodeBlock>>>> {
        if self.modules.is_empty() {
            None
        } else {
            Some(&self.modules)
        }
    }

    pub fn make_code(&mut self, data_a: &[u8], data_b: &[u8]) {
        self.data_list_a.clear();
        self.data_list_b.clear();
        self.add_data(data_a, data_b);
        self.make();
    }

    fn make(&mut self) {
        self.make_impl(false);
    }

    fn make_impl(&mut self, test: bool) {
        if self.type_number == 0 {
            let mut type_num = 1;
            for data in &self.data_list_a {
                type_num = type_num.max(get_type_number(&data.data, self.options.correct_level));
            }
            for data in &self.data_list_b {
                type_num = type_num.max(get_type_number(&data.data, self.options.correct_level));
            }
            self.type_number = type_num;
        }

        self.module_count = self.type_number * 4 + 17;
        self.modules = vec![vec![None; self.module_count as usize]; self.module_count as usize];

        // 1. 识别定位块并按照 RGB 赋予颜色
        self.setup_position_probe_pattern(0, 0, QRCodeBlock::Red); // Red
        self.setup_position_probe_pattern(0, self.module_count - 7, QRCodeBlock::Green); // Green
        self.setup_position_probe_pattern(self.module_count - 7, 0, QRCodeBlock::Blue); // Blue
        self.setup_position_adjust_pattern();
        self.setup_timing_pattern();
        self.setup_type_info(test);

        if self.type_number >= 7 {
            self.setup_type_number(test);
        }

        if self.data_cache_a.is_none() {
            self.data_cache_a = Some(self.create_data_for_list(&self.data_list_a));
        }
        if self.data_cache_b.is_none() {
            self.data_cache_b = Some(self.create_data_for_list(&self.data_list_b));
        }

        let data_a = self.data_cache_a.as_ref().unwrap().clone();
        let data_b = self.data_cache_b.as_ref().unwrap().clone();
        self.map_data(&data_a, &data_b);
    }

    fn setup_position_probe_pattern(&mut self, row: i32, col: i32, color: QRCodeBlock) {
        for r in -1..=7 {
            if row + r <= -1 || self.module_count <= row + r {
                continue;
            }
            for c in -1..=7 {
                if col + c <= -1 || self.module_count <= col + c {
                    continue;
                }
                let is_dark = ((0..=6).contains(&r) && (c == 0 || c == 6))
                    || ((0..=6).contains(&c) && (r == 0 || r == 6))
                    || ((2..=4).contains(&r) && (2..=4).contains(&c));
                self.modules[(row + r) as usize][(col + c) as usize] =
                    Some(if is_dark { color } else { QRCodeBlock::White });
            }
        }
    }

    fn setup_position_adjust_pattern(&mut self) {
        let pos = PATTERN_POSITION_TABLE[(self.type_number - 1) as usize];
        for i in 0..pos.len() {
            for j in 0..pos.len() {
                let row = pos[i];
                let col = pos[j];

                if row == 0 || col == 0 {
                    continue;
                }

                if row <= 8 && col <= 8 {
                    continue;
                }
                if row <= 8 && col >= self.module_count - 8 {
                    continue;
                }
                if row >= self.module_count - 8 && col <= 8 {
                    continue;
                }

                if self.modules[row as usize][col as usize].is_some() {
                    continue;
                }

                for r in -2..=2 {
                    for c in -2..=2 {
                        let new_row = row + r;
                        let new_col = col + c;
                        if new_row < 0
                            || new_row >= self.module_count
                            || new_col < 0
                            || new_col >= self.module_count
                        {
                            continue;
                        }
                        let is_dark = r == -2 || r == 2 || c == -2 || c == 2 || (r == 0 && c == 0);
                        // 对齐图案使用蓝色作为标识（也可以统一使用某种逻辑，这里先设为蓝色）
                        self.modules[new_row as usize][new_col as usize] = Some(if is_dark {
                            QRCodeBlock::Blue
                        } else {
                            QRCodeBlock::White
                        });
                    }
                }
            }
        }
    }

    fn setup_timing_pattern(&mut self) {
        for r in 8..self.module_count - 8 {
            if self.modules[r as usize][6].is_none() {
                let dark = r % 2 == 0;
                self.modules[r as usize][6] = Some(if dark {
                    QRCodeBlock::Blue
                } else {
                    QRCodeBlock::White
                });
            }
        }
        for c in 8..self.module_count - 8 {
            if self.modules[6][c as usize].is_none() {
                let dark = c % 2 == 0;
                self.modules[6][c as usize] = Some(if dark {
                    QRCodeBlock::Blue
                } else {
                    QRCodeBlock::White
                });
            }
        }
    }

    fn setup_type_info(&mut self, test: bool) {
        let g15 = (1 << 10) | (1 << 8) | (1 << 5) | (1 << 4) | (1 << 2) | (1 << 1) | (1 << 0);
        let g15_mask = (1 << 14) | (1 << 12) | (1 << 10) | (1 << 4) | (1 << 1);

        let mask_pattern = 0;
        let correct_level = match self.options.correct_level {
            QRErrorCorrectLevel::L => 1,
            QRErrorCorrectLevel::M => 0,
            QRErrorCorrectLevel::Q => 3,
            QRErrorCorrectLevel::H => 2,
        };

        let mut data = (correct_level << 3) | mask_pattern;
        let mut d = data << 10;

        while get_bch_digit(d) - get_bch_digit(g15) >= 0 {
            d ^= g15 << (get_bch_digit(d) - get_bch_digit(g15));
        }

        data = ((data << 10) | d) ^ g15_mask;

        // 垂直类型信息（第 8 列）- 从上到下
        for i in 0..15 {
            // 在 test 模式下，格式信息位设置为 false（白色）
            let bit = !test && ((data >> i) & 1) == 1;
            let block = if bit {
                QRCodeBlock::Blue
            } else {
                QRCodeBlock::White
            };

            if i < 6 {
                // 行 0-5
                self.modules[i as usize][8] = Some(block);
            } else if i < 8 {
                // 行 7-8（跳过第6行，因为它是定时图案）
                self.modules[(i + 1) as usize][8] = Some(block);
            } else {
                // 行 module_count-7 到 module_count-1
                self.modules[(self.module_count - 15 + i) as usize][8] = Some(block);
            }
        }

        // 水平类型信息（第 8 行）- 从右到左
        for i in 0..15 {
            let bit = !test && ((data >> i) & 1) == 1;
            let block = if bit {
                QRCodeBlock::Blue
            } else {
                QRCodeBlock::White
            };

            if i < 8 {
                // 列 module_count-1 到 module_count-8
                self.modules[8][(self.module_count - 1 - i) as usize] = Some(block);
            } else if i < 9 {
                // 列 7
                self.modules[8][(15 - i - 1 + 1) as usize] = Some(block);
            } else {
                // 列 5 到 0
                self.modules[8][(15 - i - 1) as usize] = Some(block);
            }
        }

        // 固定暗模块 (module_count-8, 8)
        self.modules[(self.module_count - 8) as usize][8] = Some(if !test {
            QRCodeBlock::Blue
        } else {
            QRCodeBlock::White
        });
    }

    fn setup_type_number(&mut self, _test: bool) {
        // 简化版本
    }

    fn map_data(&mut self, data_a: &[i32], data_b: &[i32]) {
        let mut inc = -1;
        let mut row = self.module_count - 1;
        let mut bit_index = 7;
        let mut byte_index = 0;

        let mut col = self.module_count - 1;
        while col > 0 {
            if col == 6 {
                col -= 1;
            }

            loop {
                for c in 0..2 {
                    let col_idx = col - c;
                    if col_idx < 0 || col_idx >= self.module_count {
                        continue;
                    }
                    if self.modules[row as usize][col_idx as usize].is_none() {
                        let mut dark_a = false;
                        if byte_index < data_a.len() {
                            dark_a = ((data_a[byte_index] >> bit_index) & 1) == 1;
                        }

                        let mut dark_b = false;
                        if byte_index < data_b.len() {
                            dark_b = ((data_b[byte_index] >> bit_index) & 1) == 1;
                        }

                        let mask = (row + col_idx) % 2 == 0;
                        if mask {
                            dark_a = !dark_a;
                            dark_b = !dark_b;
                        }

                        let color = match (dark_a, dark_b) {
                            (true, false) => QRCodeBlock::Red,
                            (false, true) => QRCodeBlock::Green,
                            (true, true) => QRCodeBlock::Blue,
                            (false, false) => QRCodeBlock::White,
                        };

                        self.modules[row as usize][col_idx as usize] = Some(color);

                        if bit_index == 0 {
                            bit_index = 7;
                            byte_index += 1;
                        } else {
                            bit_index -= 1;
                        }
                    }
                }

                row += inc;

                if row < 0 || self.module_count <= row {
                    row -= inc;
                    inc = -inc;
                    break;
                }
            }

            col -= 2;
        }
    }

    fn create_data_for_list(&self, data_list: &[QR8bitByte]) -> Vec<i32> {
        let rs_blocks = get_rs_blocks(self.type_number, self.options.correct_level);

        let mut buffer = BitBuffer::new();

        for data in data_list {
            buffer.put(QRMode::MODE_8BIT_BYTE, 4);
            buffer.put(
                data.get_length() as i32,
                get_length_in_bits(QRMode::MODE_8BIT_BYTE, self.type_number),
            );
            data.write(&mut buffer);
        }

        let mut total_data_count = 0;
        for block in &rs_blocks {
            total_data_count += block.data_count;
        }

        if buffer.length + 4 <= total_data_count as usize * 8 {
            buffer.put(0, 4);
        }

        while !buffer.length.is_multiple_of(8) {
            buffer.put_bit(false);
        }

        loop {
            if buffer.length >= total_data_count as usize * 8 {
                break;
            }
            buffer.put(0xEC, 8);
            if buffer.length >= total_data_count as usize * 8 {
                break;
            }
            buffer.put(0x11, 8);
        }

        #[cfg(test)]
        eprintln!(
            "DEBUG: buffer.length={}, buffer.buffer.len={}, total_data_count*8={}",
            buffer.length,
            buffer.buffer.len(),
            total_data_count as usize * 8
        );

        let data = buffer.buffer;

        let mut offset = 0;
        let max_dc_count = rs_blocks.iter().map(|b| b.data_count).max().unwrap_or(0);
        let max_ec_count = rs_blocks
            .iter()
            .map(|b| b.total_count - b.data_count)
            .max()
            .unwrap_or(0);

        let mut dcdata: Vec<Vec<i32>> = Vec::new();
        let mut ecdata: Vec<Vec<i32>> = Vec::new();

        for block in &rs_blocks {
            let dc_count = block.data_count;
            let ec_count = block.total_count - dc_count;

            dcdata.push(data[offset as usize..(offset + dc_count) as usize].to_vec());
            offset += dc_count;

            let rs_poly = Polynomial::generate_rs_poly(ec_count);

            let dc = dcdata.last().unwrap();
            let mut raw_coeff = dc.clone();
            raw_coeff.extend(std::iter::repeat_n(0, ec_count as usize));
            let raw_poly = Polynomial::new(raw_coeff, 0);

            let mod_poly = raw_poly.r#mod(&rs_poly);

            let mut ec: Vec<i32> = Vec::with_capacity(ec_count as usize);
            for i in 0..ec_count {
                let mod_index = i + mod_poly.len() as i32 - ec_count;
                let val = if mod_index >= 0 {
                    mod_poly.get(mod_index as usize)
                } else {
                    0
                };
                #[cfg(test)]
                if i < 5 {
                    eprintln!("DEBUG: ec[{}] = {} (mod_index={})", i, val, mod_index);
                }
                ec.push(val);
            }
            ecdata.push(ec);
        }

        let mut result: Vec<i32> = Vec::new();

        for i in 0..max_dc_count {
            for (_r, item) in dcdata.iter().enumerate().take(rs_blocks.len()) {
                if i < item.len() as i32 {
                    result.push(item[i as usize]);
                }
            }
        }

        for i in 0..max_ec_count {
            for (_r, item) in ecdata.iter().enumerate().take(rs_blocks.len()) {
                if i < item.len() as i32 {
                    result.push(item[i as usize]);
                }
            }
        }

        result
    }
}

impl Default for QRCode {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qrcode_v30_l() {
        let mut qr = QRCode::new();
        qr.type_number = 30;
        qr.options.correct_level = QRErrorCorrectLevel::L;

        // 构造一些随机或有特征的测试数据
        let data_a = (0..500).map(|i| (i % 256) as u8).collect::<Vec<_>>();
        let data_b = (0..500)
            .map(|i| (255 - (i % 256)) as u8)
            .collect::<Vec<_>>();

        qr.make_code(&data_a, &data_b);

        let count = qr.get_module_count();
        // Version 30: 4 * 30 + 17 = 137
        assert_eq!(count, 137);

        // 验证定位块颜色
        // 左上角 (0,0) 到 (6,6) 应该是红色相关
        assert_eq!(qr.get_block(0, 0), QRCodeBlock::Red);
        assert_eq!(qr.get_block(0, 6), QRCodeBlock::Red);
        assert_eq!(qr.get_block(6, 0), QRCodeBlock::Red);
        assert_eq!(qr.get_block(6, 6), QRCodeBlock::Red);

        // 右上角 (0, 130) 到 (6, 136) 应该是绿色相关
        assert_eq!(qr.get_block(0, 130), QRCodeBlock::Green);
        assert_eq!(qr.get_block(0, 136), QRCodeBlock::Green);
        assert_eq!(qr.get_block(6, 130), QRCodeBlock::Green);
        assert_eq!(qr.get_block(6, 136), QRCodeBlock::Green);

        // 左下角 (130, 0) 到 (136, 6) 应该是蓝色相关
        assert_eq!(qr.get_block(130, 0), QRCodeBlock::Blue);
        assert_eq!(qr.get_block(130, 6), QRCodeBlock::Blue);
        assert_eq!(qr.get_block(136, 0), QRCodeBlock::Blue);
        assert_eq!(qr.get_block(136, 6), QRCodeBlock::Blue);

        // 保存为 PNG
        use image::{Rgb, RgbImage};
        let scale = 4;
        let mut img = RgbImage::new((count * scale) as u32, (count * scale) as u32);

        for r in 0..count {
            for c in 0..count {
                let block = qr.get_block(r, c);
                let color = match block {
                    QRCodeBlock::Red => Rgb([255, 0, 0]),
                    QRCodeBlock::Green => Rgb([0, 255, 0]),
                    QRCodeBlock::Blue => Rgb([0, 0, 255]),
                    QRCodeBlock::White => Rgb([255, 255, 255]),
                };
                for dr in 0..scale {
                    for dc in 0..scale {
                        img.put_pixel((c * scale + dc) as u32, (r * scale + dr) as u32, color);
                    }
                }
            }
        }
        img.save("test_qrcode_v30.png").unwrap();

        println!(
            "Successfully generated Version 30 Color QRCode and saved to test_qrcode_v30.png!"
        );
    }
}
