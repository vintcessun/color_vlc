pub use self::decode::{
    MAX_PAYLOAD_SIZE, MetaData, RawData, Version, decode_color_blocks, decode_color_blocks_v40m,
};
use std::error::Error;

mod decode;
mod prepare;

pub use crate::shared::qr_version as version_db;

pub trait BitGrid {
    fn size(&self) -> usize;
    fn bit(&self, y: usize, x: usize) -> bool;
}

pub struct MirroredGrid<'a>(&'a dyn BitGrid);

impl BitGrid for MirroredGrid<'_> {
    fn size(&self) -> usize {
        self.0.size()
    }

    fn bit(&self, y: usize, x: usize) -> bool {
        self.0.bit(x, y)
    }
}

#[derive(Debug, Clone)]
pub struct SimpleGrid {
    cell_bitmap: Vec<u8>,
    size: usize,
}

impl SimpleGrid {
    pub fn from_func<F>(size: usize, fill_func: F) -> Self
    where
        F: Fn(usize, usize) -> bool,
    {
        let mut cell_bitmap = vec![0; (size * size).div_ceil(8)];
        let mut c = 0;
        for y in 0..size {
            for x in 0..size {
                if fill_func(x, y) {
                    cell_bitmap[c >> 3] |= 1 << (c & 7) as u8;
                }
                c += 1;
            }
        }

        SimpleGrid { cell_bitmap, size }
    }
}

impl BitGrid for SimpleGrid {
    fn size(&self) -> usize {
        self.size
    }

    fn bit(&self, y: usize, x: usize) -> bool {
        let c = y * self.size + x;
        self.cell_bitmap[c >> 3] & (1 << (c & 7) as u8) != 0
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum DeQRError {
    IoError,
    DataUnderflow,
    DataOverflow,
    UnknownDataType,
    DataEcc,
    FormatEcc,
    InvalidVersion,
    InvalidGridSize,
    EncodingError,
}

type DeQRResult<T> = Result<T, DeQRError>;

impl Error for DeQRError {}

impl From<::std::string::FromUtf8Error> for DeQRError {
    fn from(_: ::std::string::FromUtf8Error) -> Self {
        DeQRError::EncodingError
    }
}

impl ::std::fmt::Display for DeQRError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            DeQRError::IoError => "IoError(Could not write to output)",
            DeQRError::DataUnderflow => "DataUnderflow(Expected more bits to decode)",
            DeQRError::DataOverflow => "DataOverflow(Expected less bits to decode)",
            DeQRError::UnknownDataType => "UnknownDataType(DataType not known or not implemented)",
            DeQRError::DataEcc => "Ecc(Too many errors to correct)",
            DeQRError::FormatEcc => "Ecc(Version information corrupt)",
            DeQRError::InvalidVersion => "InvalidVersion(Invalid version or corrupt)",
            DeQRError::InvalidGridSize => "InvalidGridSize(Invalid version or corrupt)",
            DeQRError::EncodingError => "Encoding(Not UTF8)",
        };
        write!(f, "{msg}")
    }
}
