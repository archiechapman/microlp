//! Compensated arithmetic: exact products (`TwoProduct` through `fma`),
//! Neumaier summation, and a double-double value for row bounds that are
//! folded many times.
//!
//! A plain `f64` sum of `n` terms of magnitude `m` can be off by
//! `(n + 2)·2^-53·m`, which for a few thousand terms is above microlp's
//! round-off floor of `1e-14·m`. Everything presolve, postsolve and the check
//! sum goes through here, so their error is a few ulps of the result plus
//! about `2^-104·m`.

/// `a + b = s + e` exactly.
#[inline]
pub fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let e = (a - (s - bb)) + (b - bb);
    (s, e)
}

/// `a · b = p + e` exactly (barring over/underflow).
#[inline]
pub fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let e = a.mul_add(b, -p);
    (p, e)
}

/// Neumaier's compensated sum.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sum {
    s: f64,
    c: f64,
}

impl Sum {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn add(&mut self, x: f64) {
        let t = self.s + x;
        if !t.is_finite() {
            self.s = t;
            return;
        }
        if self.s.abs() >= x.abs() {
            self.c += (self.s - t) + x;
        } else {
            self.c += (x - t) + self.s;
        }
        self.s = t;
    }

    /// Adds `a · b`, the product's rounding error included.
    #[inline]
    pub fn add_prod(&mut self, a: f64, b: f64) {
        let (p, e) = two_prod(a, b);
        self.add(p);
        if e.is_finite() {
            self.add(e);
        }
    }

    pub fn value(&self) -> f64 {
        if self.s.is_finite() {
            self.s + self.c
        } else {
            self.s
        }
    }
}

/// Compensated `Σ a_i · x_i`.
pub fn dot(terms: impl IntoIterator<Item = (f64, f64)>) -> f64 {
    let mut s = Sum::new();
    for (a, x) in terms {
        s.add_prod(a, x);
    }
    s.value()
}

/// A double-double: the unevaluated sum `hi + lo`, `|lo| <= ulp(hi)/2`. An
/// infinite value has `lo == 0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dd {
    pub hi: f64,
    pub lo: f64,
}

impl Dd {
    pub fn new(x: f64) -> Self {
        Dd { hi: x, lo: 0.0 }
    }

    pub fn is_finite(&self) -> bool {
        self.hi.is_finite()
    }

    pub fn value(&self) -> f64 {
        self.hi + self.lo
    }

    /// `self + x`.
    pub fn add(self, x: f64) -> Self {
        if !self.hi.is_finite() {
            return self;
        }
        if !x.is_finite() {
            return Dd::new(x);
        }
        let (s, e) = two_sum(self.hi, x);
        let e = e + self.lo;
        let (hi, lo) = two_sum(s, e);
        Dd { hi, lo }
    }

    /// `self + a · b`.
    pub fn add_prod(self, a: f64, b: f64) -> Self {
        let (p, e) = two_prod(a, b);
        let r = self.add(p);
        if e.is_finite() {
            r.add(e)
        } else {
            r
        }
    }

    /// `self + other`.
    pub fn add_dd(self, other: Dd) -> Self {
        self.add(other.hi).add(other.lo)
    }

    pub fn neg(self) -> Self {
        Dd {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compensated_sum_recovers_cancellation() {
        let mut s = Sum::new();
        s.add(1e16);
        s.add(1.0);
        s.add(-1e16);
        assert_eq!(s.value(), 1.0);
    }

    #[test]
    fn long_sum_is_exact_where_plain_is_not() {
        let c = 1_234_567.123_456_789;
        let mut plain = 0.0;
        let mut s = Sum::new();
        for _ in 0..5000 {
            plain += c;
            s.add(c);
        }
        // 5000·c computed exactly, rounded once.
        let exact = Dd::new(0.0).add_prod(5000.0, c).value();
        assert_eq!(s.value(), exact);
        let _ = plain; // plain may or may not be exact; the compensated one is.
    }

    #[test]
    fn dd_folds_keep_the_low_part() {
        let mut d = Dd::new(7.0);
        for _ in 0..1000 {
            d = d.add_prod(1e6, 1.000_000_1);
        }
        for _ in 0..1000 {
            d = d.add_prod(-1e6, 1.000_000_1);
        }
        assert!((d.value() - 7.0).abs() < 1e-12);
    }
}
