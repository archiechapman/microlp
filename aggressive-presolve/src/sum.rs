//! Compensated arithmetic: Neumaier sums for activities and double-double
//! accumulators for row sides that many folds update.

/// `2^-53`, the unit round-off of `f64`.
pub const UNIT_ROUNDOFF: f64 = 1.0 / 9_007_199_254_740_992.0;

/// Round-off allowance of a sum of `n` rounded products of total magnitude
/// `m`, compensated or not: `(n + 2)·2^-53·m`.
pub fn allowance(n: usize, m: f64) -> f64 {
    (n as f64 + 2.0) * UNIT_ROUNDOFF * m
}

/// `a + b = s + e` exactly (Knuth's TwoSum).
#[inline]
pub fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let e = (a - (s - bb)) + (b - bb);
    (s, e)
}

/// `a·b = p + e` exactly (TwoProduct through a fused multiply-add).
#[inline]
pub fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let e = a.mul_add(b, -p);
    (p, e)
}

/// Neumaier's compensated sum, with the magnitude of its terms.
#[derive(Clone, Copy, Debug, Default)]
pub struct Neumaier {
    sum: f64,
    comp: f64,
    /// `Σ|terms|`.
    pub mag: f64,
    /// Number of terms.
    pub n: usize,
}

impl Neumaier {
    #[inline]
    pub fn add(&mut self, x: f64) {
        let t = self.sum + x;
        if self.sum.abs() >= x.abs() {
            self.comp += (self.sum - t) + x;
        } else {
            self.comp += (x - t) + self.sum;
        }
        self.sum = t;
        self.mag += x.abs();
        self.n += 1;
    }

    #[inline]
    pub fn value(&self) -> f64 {
        self.sum + self.comp
    }
}

/// A double-double value `hi + lo`, for quantities updated many times.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Dd {
    pub hi: f64,
    pub lo: f64,
}

impl Dd {
    pub fn new(x: f64) -> Self {
        Dd { hi: x, lo: 0.0 }
    }

    pub fn value(self) -> f64 {
        self.hi + self.lo
    }

    pub fn is_finite(self) -> bool {
        self.hi.is_finite()
    }

    /// `self += x`.
    pub fn add(&mut self, x: f64) {
        if !self.hi.is_finite() || !x.is_finite() {
            self.hi += x;
            self.lo = 0.0;
            return;
        }
        let (s, e) = two_sum(self.hi, x);
        let (hi, lo) = two_sum(s, e + self.lo);
        self.hi = hi;
        self.lo = lo;
    }

    /// `self -= a·b`, the product carried exactly.
    pub fn sub_prod(&mut self, a: f64, b: f64) {
        self.add_prod(-a, b);
    }

    /// `self += a·b`, the product carried exactly.
    pub fn add_prod(&mut self, a: f64, b: f64) {
        if !self.hi.is_finite() {
            return;
        }
        let (p, e) = two_prod(a, b);
        self.add(p);
        self.add(e);
    }

    /// `self -= a·d`, for a double-double `d`.
    pub fn sub_prod_dd(&mut self, a: f64, d: Dd) {
        self.sub_prod(a, d.hi);
        self.sub_prod(a, d.lo);
    }
}

/// `Σ a_j·x_j` in double-double, with `Σ|a_j·x_j|`.
pub fn dot(terms: impl IntoIterator<Item = (f64, f64)>) -> (f64, f64) {
    let mut acc = Dd::default();
    let mut mag = 0.0;
    for (a, x) in terms {
        let (p, e) = two_prod(a, x);
        acc.add(p);
        acc.add(e);
        mag += p.abs();
    }
    (acc.value(), mag)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn double_double_folds_keep_small_remainders() {
        let mut d = Dd::new(1e16);
        for _ in 0..1000 {
            d.sub_prod(1e12, 1.0);
        }
        d.add(0.5);
        assert_eq!(d.hi, 1e16 - 1e15);
        assert_eq!(d.lo, 0.5);
    }

    #[test]
    fn neumaier_recovers_cancellation() {
        let mut s = Neumaier::default();
        for x in [1e16, 1.0, -1e16] {
            s.add(x);
        }
        assert_eq!(s.value(), 1.0);
    }
}
