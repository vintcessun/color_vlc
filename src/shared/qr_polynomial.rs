use crate::shared::qr_math::QRMath;

pub struct Polynomial {
    pub num: Vec<i32>,
}

impl Polynomial {
    pub fn new(num: Vec<i32>, _shift: i32) -> Self {
        let mut start = 0;
        let len = num.len();
        while start < len - 1 && num[start] == 0 {
            start += 1;
        }

        let num = if start > 0 {
            num[start..].to_vec()
        } else {
            num
        };

        Polynomial { num }
    }

    pub fn get(&self, index: usize) -> i32 {
        self.num.get(index).copied().unwrap_or(0)
    }

    pub fn len(&self) -> usize {
        self.num.len()
    }

    pub fn is_empty(&self) -> bool {
        self.num.is_empty()
    }

    pub fn multiply(&self, e: &Polynomial) -> Polynomial {
        let mut num = vec![0; self.len() + e.len() - 1];

        for i in 0..self.len() {
            for j in 0..e.len() {
                num[i + j] ^= QRMath::gexp(QRMath::glog(self.get(i)) + QRMath::glog(e.get(j)));
            }
        }

        Polynomial::new(num, 0)
    }

    pub fn generate_rs_poly(ec_count: i32) -> Polynomial {
        let mut result = Polynomial::new(vec![1], 0);

        for i in 0..ec_count {
            let factor = Polynomial::new(vec![1, QRMath::gexp(i)], 0);
            result = result.multiply(&factor);
        }

        result
    }

    pub fn r#mod(&self, e: &Polynomial) -> Polynomial {
        if (self.len() as i32 - e.len() as i32) < 0 {
            return Polynomial::new(self.num.clone(), 0);
        }

        let ratio = QRMath::glog(self.get(0)) - QRMath::glog(e.get(0));

        let mut num = self.num.clone();
        for (i, item) in num.iter_mut().enumerate().take(e.len()) {
            *item ^= QRMath::gexp(QRMath::glog(e.get(i)) + ratio);
        }

        Polynomial::new(num, 0).r#mod(e)
    }

    pub fn r#mod_with_shift(&self, e: &Polynomial, shift: i32) -> Polynomial {
        let mut extended = self.num.clone();
        extended.extend(std::iter::repeat_n(0, shift as usize));

        let shifted_poly = Polynomial::new(extended, 0);
        shifted_poly.r#mod(e)
    }
}
