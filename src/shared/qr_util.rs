pub fn get_bch_digit(data: i32) -> i32 {
    let mut digit = 0;
    let mut data = data;
    while data != 0 {
        digit += 1;
        data >>= 1;
    }
    digit
}

pub fn get_length_in_bits(mode: i32, type_num: i32) -> i32 {
    if mode != 4 {
        panic!("Invalid mode");
    }

    if (1..10).contains(&type_num) { 8 } else { 16 }
}
