use std::io::Write;
use std::mem;

use g2p::{GaloisField, g2p};

use super::version_db::{RSParameters, VERSION_DATA_BASE};
use super::{BitGrid, DeQRError, DeQRResult};

g2p!(GF16, 4, modulus: 0b1_0011);
g2p!(GF256, 8, modulus: 0b1_0001_1101);

pub const MAX_PAYLOAD_SIZE: usize = 8896;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct Version(pub usize);

impl Version {
    pub fn from_size(b: usize) -> DeQRResult<Self> {
        if b < 21 || !(b - 21).is_multiple_of(4) {
            return Err(DeQRError::InvalidGridSize);
        }
        let computed_version = (b - 17) / 4;

        if (1..=40).contains(&computed_version) {
            Ok(Version(computed_version))
        } else {
            Err(DeQRError::InvalidVersion)
        }
    }

    pub fn to_size(&self) -> usize {
        self.0 * 4 + 17
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MetaData {
    pub version: Version,
    pub ecc_level: u16,
    pub mask: u16,
}

#[derive(Clone)]
pub struct RawData {
    pub data: [u8; MAX_PAYLOAD_SIZE],
    pub len: usize,
}

impl RawData {
    pub fn push(&mut self, bit: bool) {
        assert!((self.len / 8) < MAX_PAYLOAD_SIZE);
        let bitpos = (self.len & 7) as u8;
        let bytepos = self.len >> 3;

        if bit {
            self.data[bytepos] |= 0x80_u8 >> bitpos;
        }
        self.len += 1;
    }
}

#[derive(Clone)]
pub struct CorrectedDataStream {
    data: [u8; MAX_PAYLOAD_SIZE],
    ptr: usize,
    bit_len: usize,
}

impl CorrectedDataStream {
    pub fn bits_remaining(&self) -> usize {
        assert!(self.bit_len >= self.ptr);
        self.bit_len - self.ptr
    }

    pub fn take_bits(&mut self, nbits: usize) -> usize {
        let mut ret = 0;
        let max_len = ::std::cmp::min(self.bits_remaining(), nbits);
        assert!(max_len <= mem::size_of::<usize>() * 8);
        for _ in 0..max_len {
            let b = self.data[self.ptr >> 3];
            let bitpos = self.ptr & 7;
            ret <<= 1;
            if 0 != (b << bitpos) & 0x80 {
                ret |= 1
            }
            self.ptr += 1;
        }

        ret
    }
}

pub fn decode<W>(code: &dyn BitGrid, writer: W) -> DeQRResult<MetaData>
where
    W: Write,
{
    fn _decode(c: &dyn BitGrid) -> DeQRResult<(MetaData, CorrectedDataStream)> {
        let (meta, raw) = get_raw(c, true)?;
        let stream = codestream_ecc(&meta, raw)?;
        Ok((meta, stream))
    }
    let (meta, stream) = match _decode(code) {
        Ok((meta, stream)) => (meta, stream),
        Err(original) => match _decode(&super::MirroredGrid(code)) {
            Ok((meta, stream)) => (meta, stream),
            Err(_) => return Err(original),
        },
    };

    decode_payload(&meta, stream, writer)?;
    Ok(meta)
}

pub fn get_raw(code: &dyn BitGrid, remove_masked: bool) -> DeQRResult<(MetaData, RawData)> {
    let meta = read_format(code)?;
    let raw = read_data(code, &meta, remove_masked);
    Ok((meta, raw))
}

fn decode_payload<W>(meta: &MetaData, mut ds: CorrectedDataStream, mut writer: W) -> DeQRResult<()>
where
    W: Write,
{
    while ds.bits_remaining() >= 4 {
        let ty = ds.take_bits(4);
        match ty {
            0 => break,
            1 => decode_numeric(meta, &mut ds, &mut writer),
            2 => decode_alpha(meta, &mut ds, &mut writer),
            4 => decode_byte(meta, &mut ds, &mut writer),
            8 => decode_kanji(meta, &mut ds, &mut writer),
            7 => decode_eci(meta, &mut ds, &mut writer),
            _ => Err(DeQRError::UnknownDataType)?,
        }?;
    }
    Ok(())
}

fn decode_eci<W>(_meta: &MetaData, ds: &mut CorrectedDataStream, mut _writer: W) -> DeQRResult<()>
where
    W: Write,
{
    if ds.bits_remaining() < 8 {
        Err(DeQRError::DataUnderflow)?
    }

    let mut _eci = ds.take_bits(8) as u32;
    if _eci & 0xc0 == 0x80 {
        if ds.bits_remaining() < 8 {
            Err(DeQRError::DataUnderflow)?
        }
        _eci = (_eci << 8) | (ds.take_bits(8) as u32)
    } else if _eci & 0xe0 == 0xc0 {
        if ds.bits_remaining() < 16 {
            Err(DeQRError::DataUnderflow)?
        }

        _eci = (_eci << 16) | (ds.take_bits(16) as u32)
    }
    Ok(())
}

fn decode_kanji<W>(meta: &MetaData, ds: &mut CorrectedDataStream, mut writer: W) -> DeQRResult<()>
where
    W: Write,
{
    let nbits = match meta.version {
        Version(0..=9) => 8,
        Version(10..=26) => 10,
        _ => 12,
    };

    let count = ds.take_bits(nbits);
    if ds.bits_remaining() < count * 13 {
        Err(DeQRError::DataUnderflow)?
    }

    for _ in 0..count {
        let d = ds.take_bits(13);
        let ms_b = d / 0xc0;
        let ls_b = d % 0xc0;
        let intermediate = (ms_b << 8) | ls_b;
        let sjw = if intermediate + 0x8140 <= 0x9ffc {
            /* bytes are in the range 0x8140 to 0x9FFC */
            (intermediate + 0x8140) as u16
        } else {
            (intermediate + 0xc140) as u16
        };
        writer
            .write_all(&[(sjw >> 8) as u8, (sjw & 0xff) as u8])
            .map_err(|_| DeQRError::IoError)?;
    }
    Ok(())
}

fn decode_byte<W>(meta: &MetaData, ds: &mut CorrectedDataStream, mut writer: W) -> DeQRResult<()>
where
    W: Write,
{
    let nbits = match meta.version {
        Version(0..=9) => 8,
        _ => 16,
    };

    let count = ds.take_bits(nbits);
    if ds.bits_remaining() < count * 8 {
        Err(DeQRError::DataUnderflow)?;
    }

    for _ in 0..count {
        let buf = &[ds.take_bits(8) as u8];
        writer.write_all(buf).map_err(|_| DeQRError::IoError)?;
    }
    Ok(())
}

fn decode_alpha<W>(meta: &MetaData, ds: &mut CorrectedDataStream, mut writer: W) -> DeQRResult<()>
where
    W: Write,
{
    let nbits = match meta.version {
        Version(0..=9) => 9,
        Version(10..=26) => 11,
        _ => 13,
    };
    let mut count = ds.take_bits(nbits);
    let mut buf = [0; 2];

    while count >= 2 {
        alpha_tuple(&mut buf, ds, 11, 2)?;
        writer.write_all(&buf[..]).map_err(|_| DeQRError::IoError)?;
        count -= 2;
    }

    if count == 1 {
        alpha_tuple(&mut buf, ds, 6, 1)?;
        writer
            .write_all(&buf[..1])
            .map_err(|_| DeQRError::IoError)?;
    }

    Ok(())
}

fn alpha_tuple(
    buf: &mut [u8; 2],
    ds: &mut CorrectedDataStream,
    nbits: usize,
    digits: usize,
) -> DeQRResult<()> {
    if ds.bits_remaining() < nbits {
        Err(DeQRError::DataUnderflow)
    } else {
        let mut tuple = ds.take_bits(nbits);
        for i in (0..digits).rev() {
            const ALPHA_MAP: &[u8; 46] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:\x00";
            buf[i] = ALPHA_MAP[tuple % 45];
            tuple /= 45;
        }
        Ok(())
    }
}

fn decode_numeric<W>(meta: &MetaData, ds: &mut CorrectedDataStream, mut writer: W) -> DeQRResult<()>
where
    W: Write,
{
    let nbits = match meta.version {
        Version(0..=9) => 10,
        Version(10..=26) => 12,
        _ => 14,
    };

    let mut count = ds.take_bits(nbits);
    let mut buf = [0; 3];
    while count >= 3 {
        numeric_tuple(&mut buf, ds, 10, 3)?;
        writer.write_all(&buf[..]).map_err(|_| DeQRError::IoError)?;
        count -= 3;
    }

    if count == 2 {
        numeric_tuple(&mut buf, ds, 7, 2)?;
        writer
            .write_all(&buf[..2])
            .map_err(|_| DeQRError::IoError)?;
        count -= 2;
    }
    if count == 1 {
        numeric_tuple(&mut buf, ds, 4, 1)?;
        writer
            .write_all(&buf[..1])
            .map_err(|_| DeQRError::IoError)?;
    }

    Ok(())
}

fn numeric_tuple(
    buf: &mut [u8; 3],
    ds: &mut CorrectedDataStream,
    nbits: usize,
    digits: usize,
) -> DeQRResult<()> {
    if ds.bits_remaining() < nbits {
        Err(DeQRError::DataUnderflow)
    } else {
        let mut tuple = ds.take_bits(nbits);
        for i in (0..digits).rev() {
            buf[i] = (tuple % 10) as u8 + b'0';
            tuple /= 10;
        }
        Ok(())
    }
}

fn codestream_ecc(meta: &MetaData, ds: RawData) -> DeQRResult<CorrectedDataStream> {
    let mut out = CorrectedDataStream {
        data: [0; MAX_PAYLOAD_SIZE],
        ptr: 0,
        bit_len: 0,
    };

    let ver = &VERSION_DATA_BASE[meta.version.0];
    let sb_ecc = &ver.ecc[meta.ecc_level as usize];
    let lb_ecc = RSParameters {
        bs: sb_ecc.bs + 1,
        dw: sb_ecc.dw + 1,
        ns: sb_ecc.ns,
    };

    let lb_count = (ver.data_bytes - sb_ecc.bs * sb_ecc.ns) / (sb_ecc.bs + 1);
    let bc = lb_count + sb_ecc.ns;
    let ecc_offset = sb_ecc.dw * bc + lb_count;

    let mut total_corrected = 0;
    let mut failed_blocks = 0;
    let mut dst_offset = 0;
    for i in 0..bc {
        let ecc = if i < sb_ecc.ns { sb_ecc } else { &lb_ecc };
        let dst = &mut out.data[dst_offset..(dst_offset + ecc.bs)];
        let num_ec = ecc.bs - ecc.dw;
        #[allow(clippy::needless_range_loop)]
        for j in 0..ecc.dw {
            dst[j] = ds.data[j * bc + i];
        }
        for j in 0..num_ec {
            dst[ecc.dw + j] = ds.data[ecc_offset + j * bc + i];
        }
        match correct_block(dst, ecc) {
            Ok(count) => total_corrected += count,
            Err(_) => {
                failed_blocks += 1;
                println!(
                    "  ECC failure in block {}/{}: capacity {} errors",
                    i + 1,
                    bc,
                    (ecc.bs - ecc.dw) / 2
                );
            }
        }

        dst_offset += ecc.dw;
    }

    if total_corrected > 0 || failed_blocks > 0 {
        println!(
            "  ECC Summary: corrected {} bytes, {}/{} blocks failed",
            total_corrected, failed_blocks, bc
        );
    }

    if failed_blocks > bc / 2 {
        return Err(DeQRError::DataEcc);
    }

    out.bit_len = dst_offset * 8;
    Ok(out)
}

fn correct_block(block: &mut [u8], ecc: &RSParameters) -> DeQRResult<usize> {
    assert!(ecc.bs > ecc.dw);

    let npar = ecc.bs - ecc.dw;
    let mut sigma_deriv = [GF256::ZERO; 64];

    // Calculate syndromes. If all 0 there is nothing to do.
    let s = match block_syndromes(&block[..ecc.bs], npar) {
        Ok(_) => return Ok(0),
        Err(s) => s,
    };

    let sigma = berlekamp_massey(&s, npar);
    /* Compute derivative of sigma */
    for i in (1..64).step_by(2) {
        sigma_deriv[i - 1] = sigma[i];
    }

    /* Compute error evaluator polynomial */
    let omega = eloc_poly(&s, &sigma, npar - 1);

    let mut corrected_count = 0;
    /* Find error locations and magnitudes */
    for i in 0..ecc.bs {
        let xinv = GF256::GENERATOR.pow(255 - i);
        if poly_eval(&sigma, xinv) == GF256::ZERO {
            let sd_x = poly_eval(&sigma_deriv, xinv);
            let omega_x = poly_eval(&omega, xinv);
            if sd_x == GF256::ZERO {
                return Err(DeQRError::DataEcc);
            }
            let error = omega_x / sd_x;
            block[ecc.bs - i - 1] = (GF256(block[ecc.bs - i - 1]) + error).0;
            corrected_count += 1;
        }
    }

    match block_syndromes(&block[..ecc.bs], npar) {
        Ok(_) => Ok(corrected_count),
        Err(_) => Err(DeQRError::DataEcc),
    }
}

/* ***********************************************************************
 * Code stream error correction
 *
 * Generator polynomial for GF(2^8) is x^8 + x^4 + x^3 + x^2 + 1
 */
fn block_syndromes(block: &[u8], npar: usize) -> Result<[GF256; 64], [GF256; 64]> {
    let mut nonzero: bool = false;
    let mut s = [GF256::ZERO; 64];

    #[allow(clippy::needless_range_loop)]
    for i in 0..npar {
        for j in 0..block.len() {
            let c = GF256(block[block.len() - 1 - j]);
            s[i] += c * GF256::GENERATOR.pow(i * j);
        }
        if s[i] != GF256::ZERO {
            nonzero = true;
        }
    }
    if nonzero { Err(s) } else { Ok(s) }
}

fn poly_eval<G>(s: &[G; 64], x: G) -> G
where
    G: GaloisField,
{
    let mut sum = G::ZERO;
    let mut x_pow = G::ONE;

    #[allow(clippy::needless_range_loop)]
    for i in 0..64 {
        sum += s[i] * x_pow;
        x_pow *= x;
    }
    sum
}

fn eloc_poly(s: &[GF256; 64], sigma: &[GF256; 64], npar: usize) -> [GF256; 64] {
    let mut omega = [GF256::ZERO; 64];
    for i in 0..npar {
        let a = sigma[i];
        for j in 0..(npar - i) {
            let b = s[j + 1];
            omega[i + j] += a * b;
        }
    }
    omega
}

fn berlekamp_massey<G>(s: &[G; 64], n: usize) -> [G; 64]
where
    G: GaloisField,
{
    let mut ts: [G; 64] = [G::ZERO; 64];
    let mut cs: [G; 64] = [G::ZERO; 64];
    let mut bs: [G; 64] = [G::ZERO; 64];
    let mut l: usize = 0;
    let mut m: usize = 1;
    let mut b = G::ONE;
    bs[0] = G::ONE;
    cs[0] = G::ONE;

    for n in 0..n {
        let mut d = s[n];

        // Calculate in GF(p):
        // d = s[n] + \Sum_{i=1}^{l} c[i] * s[n - i]
        for i in 1..=l {
            d += cs[i] * s[n - i];
        }
        // Pre-calculate d * b^-1 in GF(p)
        let mult = d / b;

        if d == G::ZERO {
            m += 1
        } else if l * 2 <= n {
            ts.copy_from_slice(&cs);
            poly_add(&mut cs, &bs, mult, m);
            bs.copy_from_slice(&ts);
            l = n + 1 - l;
            b = d;
            m = 1
        } else {
            poly_add(&mut cs, &bs, mult, m);
            m += 1
        }
    }
    cs
}
/* ***********************************************************************
 * Polynomial operations
 */
fn poly_add<G>(dst: &mut [G; 64], src: &[G; 64], c: G, shift: usize)
where
    G: GaloisField,
{
    if c == G::ZERO {
        return;
    }

    #[allow(clippy::needless_range_loop)]
    for i in 0..64 {
        let p = i + shift;
        if p >= 64 {
            break;
        }
        let v = src[i];
        dst[p] += v * c;
    }
}

fn read_data(code: &dyn BitGrid, meta: &MetaData, remove_mask: bool) -> RawData {
    let mut ds = RawData {
        data: [0; MAX_PAYLOAD_SIZE],
        len: 0,
    };

    let mut y = code.size() - 1;
    let mut x = code.size() - 1;
    let mut neg_dir = true;

    while x > 0 {
        if x == 6 {
            x -= 1;
        }
        if !reserved_cell(meta.version, y, x) {
            ds.push(read_bit(code, meta, y, x, remove_mask));
        }
        if !reserved_cell(meta.version, y, x - 1) {
            ds.push(read_bit(code, meta, y, x - 1, remove_mask));
        }

        let (new_y, new_neg_dir) = match (y, neg_dir) {
            (0, true) => {
                x = x.saturating_sub(2);
                (0, false)
            }
            (y, false) if y == code.size() - 1 => {
                x = x.saturating_sub(2);
                (code.size() - 1, true)
            }
            (y, true) => (y - 1, true),
            (y, false) => (y + 1, false),
        };

        y = new_y;
        neg_dir = new_neg_dir;
    }

    ds
}

fn read_bit(code: &dyn BitGrid, meta: &MetaData, y: usize, x: usize, remove_mask: bool) -> bool {
    let mut v = code.bit(y, x) as u8;
    if remove_mask && mask_bit(meta.mask, y, x) {
        v ^= 1
    }

    v != 0
}

fn mask_bit(mask: u16, y: usize, x: usize) -> bool {
    match mask {
        0 => (y + x).is_multiple_of(2),
        1 => y.is_multiple_of(2),
        2 => x.is_multiple_of(3),
        3 => (y + x).is_multiple_of(3),
        4 => ((y / 2) + (x / 3)).is_multiple_of(2),
        5 => 0 == ((y * x) % 2 + (y * x) % 3),
        6 => ((y * x) % 2 + (y * x) % 3).is_multiple_of(2),
        7 => ((y * x) % 3 + (y + x) % 2).is_multiple_of(2),
        _ => panic!("Unknown mask value"),
    }
}

fn reserved_cell(version: Version, i: usize, j: usize) -> bool {
    let ver = &VERSION_DATA_BASE[version.0];
    let size = version.0 * 4 + 17;

    if i < 9 && j < 9 {
        return true;
    }

    if i + 8 >= size && j < 9 {
        return true;
    }

    if i < 9 && j + 8 >= size {
        return true;
    }

    if i == 6 || j == 6 {
        return true;
    }

    if version.0 >= 7 {
        #[allow(clippy::if_same_then_else)]
        if i < 6 && j + 11 >= size {
            return true;
        } else if i + 11 >= size && j < 6 {
            return true;
        }
    }

    let mut ai = None;
    let mut aj = None;

    let mut len = 0;
    for (a, &pattern) in ver.apat.iter().take_while(|&&x| x != 0).enumerate() {
        len = a;
        if pattern.abs_diff(i) < 3 {
            ai = Some(a)
        }
        if pattern.abs_diff(j) < 3 {
            aj = Some(a)
        }
    }

    match (ai, aj) {
        (Some(x), Some(y)) if x == len && y == len => true,
        (Some(x), Some(_)) if 0 < x && x < len => true,
        (Some(_), Some(x)) if 0 < x && x < len => true,
        _ => false,
    }
}

/// 与 `reserved_cell` 相同，但不跳过版本信息区域。
/// 用于我们自定义的编码器：encoder::setup_type_number 是空函数，
/// 版本信息区域的格子被 map_data 填了真实数据位，不能按保留格跳过。
fn reserved_cell_no_version_info(version: Version, i: usize, j: usize) -> bool {
    let ver = &VERSION_DATA_BASE[version.0];
    let size = version.0 * 4 + 17;

    if i < 9 && j < 9 {
        return true;
    }

    if i + 8 >= size && j < 9 {
        return true;
    }

    if i < 9 && j + 8 >= size {
        return true;
    }

    if i == 6 || j == 6 {
        return true;
    }

    // 故意不检查 version >= 7 的版本信息块，因为编码器未写入版本信息

    let mut ai = None;
    let mut aj = None;

    let mut len = 0;
    for (a, &pattern) in ver.apat.iter().take_while(|&&x| x != 0).enumerate() {
        len = a;
        if pattern.abs_diff(i) < 3 {
            ai = Some(a)
        }
        if pattern.abs_diff(j) < 3 {
            aj = Some(a)
        }
    }

    match (ai, aj) {
        (Some(x), Some(y)) if x == len && y == len => true,
        (Some(x), Some(_)) if 0 < x && x < len => true,
        (Some(_), Some(x)) if 0 < x && x < len => true,
        _ => false,
    }
}

fn read_data_no_version_info(code: &dyn BitGrid, meta: &MetaData, remove_mask: bool) -> RawData {
    let mut ds = RawData {
        data: [0; MAX_PAYLOAD_SIZE],
        len: 0,
    };

    let mut y = code.size() - 1;
    let mut x = code.size() - 1;
    let mut neg_dir = true;

    while x > 0 {
        if x == 6 {
            x -= 1;
        }
        if !reserved_cell_no_version_info(meta.version, y, x) {
            ds.push(read_bit(code, meta, y, x, remove_mask));
        }
        if !reserved_cell_no_version_info(meta.version, y, x - 1) {
            ds.push(read_bit(code, meta, y, x - 1, remove_mask));
        }

        let (new_y, new_neg_dir) = match (y, neg_dir) {
            (0, true) => {
                x = x.saturating_sub(2);
                (0, false)
            }
            (y, false) if y == code.size() - 1 => {
                x = x.saturating_sub(2);
                (code.size() - 1, true)
            }
            (y, true) => (y - 1, true),
            (y, false) => (y + 1, false),
        };

        y = new_y;
        neg_dir = new_neg_dir;
    }

    ds
}

fn read_format(code: &dyn BitGrid) -> DeQRResult<MetaData> {
    let version = Version::from_size(code.size())?;

    let mut format1 = 0;
    const XS1: [usize; 15] = [8, 8, 8, 8, 8, 8, 8, 8, 7, 5, 4, 3, 2, 1, 0];
    const YS1: [usize; 15] = [0, 1, 2, 3, 4, 5, 7, 8, 8, 8, 8, 8, 8, 8, 8];
    for i in (0..15).rev() {
        format1 = (format1 << 1) | code.bit(YS1[i], XS1[i]) as u16;
    }

    let mut format2 = 0;
    for i in 0..7 {
        format2 = (format2 << 1) | code.bit(code.size() - 1 - i, 8) as u16;
    }
    for i in 0..8 {
        format2 = (format2 << 1) | code.bit(8, code.size() - 8 + i) as u16;
    }

    let mut best_mask = 0u16;
    let mut best_ecc = 0u16;
    let mut best_score = -1;

    for ecc in 0..4u16 {
        for mask in 0..8u16 {
            let fdata = (ecc << 3) | mask;
            let g15 = (1 << 10) | (1 << 8) | (1 << 5) | (1 << 4) | (1 << 2) | (1 << 1) | (1 << 0);
            let mut d = fdata << 10;
            for i in (0..5).rev() {
                if (d >> (i + 10)) & 1 != 0 {
                    d ^= g15 << i;
                }
            }
            let theoretical_unmasked = (fdata << 10) | d;
            let theoretical_masked = theoretical_unmasked ^ 0x5412;

            let score1 = 15 - (format1 ^ theoretical_masked).count_ones();
            let score2 = 15 - (format2 ^ theoretical_masked).count_ones();
            let score = score1.max(score2) as i32;

            if score > best_score {
                best_score = score;
                best_mask = mask;
                best_ecc = ecc;
            }
        }
    }

    let ecc_char = match best_ecc {
        0 => 'M',
        1 => 'L',
        2 => 'H',
        3 => 'Q',
        _ => '?',
    };

    println!(
        "  Extracted Metadata: Version {}, ECC {}, Mask {} (Match score: {}/15)",
        version.0, ecc_char, best_mask, best_score
    );
    println!(
        "  Raw Format 1: {:015b}, Format 2: {:015b}",
        format1, format2
    );

    Ok(MetaData {
        version,
        ecc_level: best_ecc,
        mask: best_mask,
    })
}

pub fn decode_color_blocks(
    blocks: &[Vec<crate::shared::QRCodeBlock>],
) -> DeQRResult<(Vec<u8>, Vec<u8>)> {
    use super::SimpleGrid;
    let size = blocks.len();
    if size == 0 {
        return Err(DeQRError::InvalidGridSize);
    }

    let is_finder_pattern_dark = |x: usize, y: usize| -> bool {
        let is_in_finder = |lx: usize, ly: usize| -> bool {
            (lx == 0 || lx == 6 || ly == 0 || ly == 6)
                || ((2..=4).contains(&lx) && (2..=4).contains(&ly))
        };

        if x < 7 && y < 7 {
            return is_in_finder(x, y);
        }
        if x >= size - 7 && y < 7 {
            return is_in_finder(x - (size - 7), y);
        }
        if x < 7 && y >= size - 7 {
            return is_in_finder(x, y - (size - 7));
        }
        false
    };

    let grid_a = SimpleGrid::from_func(size, |x, y| {
        if is_finder_pattern_dark(x, y) {
            return true;
        }
        matches!(
            blocks[y][x],
            crate::shared::QRCodeBlock::Red | crate::shared::QRCodeBlock::Blue
        )
    });

    let grid_b = SimpleGrid::from_func(size, |x, y| {
        if is_finder_pattern_dark(x, y) {
            return true;
        }
        matches!(
            blocks[y][x],
            crate::shared::QRCodeBlock::Green | crate::shared::QRCodeBlock::Blue
        )
    });

    let mut out_a = Vec::new();
    decode(&grid_a, &mut out_a)?;

    let mut out_b = Vec::new();
    decode(&grid_b, &mut out_b)?;

    Ok((out_a, out_b))
}

/// 使用固化的Version 40和M等级来解码颜色块
/// 这个函数假设QR码始终是Version 40且使用M等级纠错
pub fn decode_color_blocks_v40m(
    blocks: &[Vec<crate::shared::QRCodeBlock>],
) -> DeQRResult<(Vec<u8>, Vec<u8>)> {
    use super::SimpleGrid;
    let size = blocks.len();
    if size == 0 {
        return Err(DeQRError::InvalidGridSize);
    }

    // 验证大小是否符合Version 40
    // Version 40的QR码大小是 40*4+17 = 177
    if size != 177 {
        return Err(DeQRError::InvalidGridSize);
    }

    let is_finder_pattern_dark = |x: usize, y: usize| -> bool {
        let is_in_finder = |lx: usize, ly: usize| -> bool {
            (lx == 0 || lx == 6 || ly == 0 || ly == 6)
                || ((2..=4).contains(&lx) && (2..=4).contains(&ly))
        };

        if x < 7 && y < 7 {
            return is_in_finder(x, y);
        }
        if x >= size - 7 && y < 7 {
            return is_in_finder(x - (size - 7), y);
        }
        if x < 7 && y >= size - 7 {
            return is_in_finder(x, y - (size - 7));
        }
        false
    };

    let grid_a = SimpleGrid::from_func(size, |x, y| {
        if is_finder_pattern_dark(x, y) {
            return true;
        }
        matches!(
            blocks[y][x],
            crate::shared::QRCodeBlock::Red | crate::shared::QRCodeBlock::Blue
        )
    });

    let grid_b = SimpleGrid::from_func(size, |x, y| {
        if is_finder_pattern_dark(x, y) {
            return true;
        }
        matches!(
            blocks[y][x],
            crate::shared::QRCodeBlock::Green | crate::shared::QRCodeBlock::Blue
        )
    });

    let mut out_a = Vec::new();
    decode_fixed_v40m(&grid_a, &mut out_a)?;

    let mut out_b = Vec::new();
    decode_fixed_v40m(&grid_b, &mut out_b)?;

    Ok((out_a, out_b))
}

/// 内部解码函数：使用固定的Version 40和M等级
fn decode_fixed_v40m<W>(code: &dyn BitGrid, mut writer: W) -> DeQRResult<()>
where
    W: Write,
{
    fn try_decode_single<W>(code: &dyn BitGrid, writer: &mut W) -> DeQRResult<()>
    where
        W: Write,
    {
        let version = Version(40);
        let ecc_level = 0u16; // M 对应 0

        for mask in 0..8u16 {
            let meta = MetaData {
                version,
                ecc_level,
                mask,
            };

            let raw = read_data(code, &meta, true);
            match codestream_ecc(&meta, raw) {
                Ok(stream) => match decode_payload(&meta, stream, &mut *writer) {
                    Ok(()) => {
                        println!(
                            "  Fixed Decode Success: Version {}, ECC M, Mask {} ",
                            version.0, mask
                        );
                        return Ok(());
                    }
                    Err(_) => continue,
                },
                Err(_) => continue,
            }
        }

        Err(DeQRError::DataEcc)
    }

    // 先尝试原始方向，再尝试镜像方向（与decode()保持一致）
    if let Ok(()) = try_decode_single(code, &mut writer) {
        return Ok(());
    }

    let mirrored = super::MirroredGrid(code);
    if let Ok(()) = try_decode_single(&mirrored, &mut writer) {
        return Ok(());
    }

    Err(DeQRError::DataEcc)
}
