//! Runtime generation of the fixed-base window table.
//!
//! The table stores `j * 2^(window * bits) * B` in the `(y+x, y-x, 2dxy)`
//! form used by the mixed-addition kernel.  Each coordinate is split into
//! ten signed 32-bit limbs with the same radix-2^25.5 layout as the kernel.
//!
//! Generating the table at startup keeps the executable small; the tradeoff
//! is a one-time CPU cost and a table-sized allocation.

use rayon::prelude::*;

const MASK51: u64 = (1u64 << 51) - 1;
const P: [u64; 5] = [MASK51 - 18, MASK51, MASK51, MASK51, MASK51];

const SQRT_EXP: [u64; 4] = [
    0xffff_ffff_ffff_fffe,
    u64::MAX,
    u64::MAX,
    0x0fff_ffff_ffff_ffff,
];
const SQRT_M1_EXP: [u64; 4] = [
    0xffff_ffff_ffff_fffb,
    u64::MAX,
    u64::MAX,
    0x1fff_ffff_ffff_ffff,
];
const INVERT_EXP: [u64; 4] = [
    0xffff_ffff_ffff_ffeb,
    u64::MAX,
    u64::MAX,
    0x7fff_ffff_ffff_ffff,
];

/// Number of windows needed to cover a 255-bit clamped Ed25519 scalar.
pub fn window_count(bits: usize) -> usize {
    255usize.div_ceil(bits)
}

/// Generate the table for `bits`-bit windows.
///
/// The result is `window_count(bits) * (1 << bits) * 30` i32 values.
pub fn generate(bits: usize) -> Vec<i32> {
    assert!((4..=16).contains(&bits), "window bits must be in 4..=16");

    let windows = window_count(bits);
    let size = 1usize << bits;
    let entries = windows
        .checked_mul(size)
        .expect("window table entry count overflow");
    let mut table = vec![0i32; entries * 30];

    let d = d_const();
    let two_d = d.add(d);
    let base = base_point(d);

    // The generator for window w is 2^(w * bits) * B.
    let mut generators = Vec::with_capacity(windows);
    let mut generator = base;
    for _ in 0..windows {
        generators.push(generator);
        for _ in 0..bits {
            generator = add(generator, generator, two_d);
        }
    }

    table
        .par_chunks_mut(size * 30)
        .zip(generators.into_par_iter())
        .for_each(|(chunk, generator)| fill_window(chunk, size, generator, two_d));

    table
}

fn d_const() -> Fe {
    Fe::from_u64(121_665)
        .neg()
        .mul(Fe::from_u64(121_666).invert())
}

fn base_point(d: Fe) -> Point {
    // y = 4/5, x = sqrt((y^2 - 1) / (d*y^2 + 1)), choosing the even x.
    let y = Fe::from_u64(4).mul(Fe::from_u64(5).invert());
    let yy = y.square();
    let xx = yy.sub(Fe::ONE).mul(d.mul(yy).add(Fe::ONE).invert());

    let mut x = xx.pow(SQRT_EXP, 252);
    if x.square().to_bytes() != xx.to_bytes() {
        x = x.mul(Fe::from_u64(2).pow(SQRT_M1_EXP, 253));
    }
    if x.is_odd() {
        x = x.neg();
    }

    Point {
        x,
        y,
        z: Fe::ONE,
        t: x.mul(y),
    }
}

fn fill_window(chunk: &mut [i32], size: usize, generator: Point, two_d: Fe) {
    let mut points = Vec::with_capacity(size);
    let mut point = Point::identity();
    for _ in 0..size {
        points.push(point);
        point = add(point, generator, two_d);
    }

    // Montgomery batch inversion: one inversion for the whole window.
    let mut products = Vec::with_capacity(size);
    let mut running = Fe::ONE;
    for point in &points {
        running = running.mul(point.z);
        products.push(running);
    }

    let mut inverse = running.invert();
    for j in (0..size).rev() {
        let z_inverse = if j == 0 {
            inverse
        } else {
            inverse.mul(products[j - 1])
        };
        inverse = inverse.mul(points[j].z);

        let x = points[j].x.mul(z_inverse);
        let y = points[j].y.mul(z_inverse);
        let q0 = y.add(x);
        let q1 = y.sub(x);
        let q2 = two_d.mul(x).mul(y);

        let base = j * 30;
        put_limbs(&mut chunk[base..base + 10], q0);
        put_limbs(&mut chunk[base + 10..base + 20], q1);
        put_limbs(&mut chunk[base + 20..base + 30], q2);
    }
}

#[derive(Clone, Copy)]
struct Point {
    x: Fe,
    y: Fe,
    z: Fe,
    t: Fe,
}

impl Point {
    fn identity() -> Self {
        Self {
            x: Fe::ZERO,
            y: Fe::ONE,
            z: Fe::ONE,
            t: Fe::ZERO,
        }
    }
}

fn add(p: Point, q: Point, two_d: Fe) -> Point {
    let a = p.y.sub(p.x).mul(q.y.sub(q.x));
    let b = p.y.add(p.x).mul(q.y.add(q.x));
    let c = p.t.mul(two_d).mul(q.t);
    let d = p.z.add(p.z).mul(q.z);
    let e = b.sub(a);
    let f = d.sub(c);
    let g = d.add(c);
    let h = b.add(a);

    Point {
        x: e.mul(f),
        y: g.mul(h),
        z: f.mul(g),
        t: e.mul(h),
    }
}

/// Field element modulo 2^255 - 19, five 51-bit limbs.
#[derive(Clone, Copy)]
struct Fe([u64; 5]);

impl Fe {
    const ZERO: Self = Self([0; 5]);
    const ONE: Self = Self([1, 0, 0, 0, 0]);

    fn from_u64(value: u64) -> Self {
        Self([value, 0, 0, 0, 0])
    }

    fn weak_reduce(mut self) -> Self {
        for _ in 0..2 {
            let c0 = self.0[0] >> 51;
            self.0[0] &= MASK51;
            self.0[1] += c0;

            let c1 = self.0[1] >> 51;
            self.0[1] &= MASK51;
            self.0[2] += c1;

            let c2 = self.0[2] >> 51;
            self.0[2] &= MASK51;
            self.0[3] += c2;

            let c3 = self.0[3] >> 51;
            self.0[3] &= MASK51;
            self.0[4] += c3;

            let c4 = self.0[4] >> 51;
            self.0[4] &= MASK51;
            self.0[0] += c4 * 19;
        }
        self
    }

    fn canonical(self) -> Self {
        let mut reduced = self.weak_reduce();

        let mut at_least_p = true;
        for i in (0..5).rev() {
            if reduced.0[i] > P[i] {
                break;
            }
            if reduced.0[i] < P[i] {
                at_least_p = false;
                break;
            }
        }

        if at_least_p {
            let mut borrow = 0i128;
            for i in 0..5 {
                let mut value = i128::from(reduced.0[i]) - i128::from(P[i]) - borrow;
                if value < 0 {
                    value += 1i128 << 51;
                    borrow = 1;
                } else {
                    borrow = 0;
                }
                reduced.0[i] = value as u64;
            }
            debug_assert_eq!(borrow, 0);
        }

        reduced
    }

    fn add(self, other: Self) -> Self {
        let mut out = [0u64; 5];
        for i in 0..5 {
            out[i] = self.0[i] + other.0[i];
        }
        Self(out).weak_reduce()
    }

    fn sub(self, other: Self) -> Self {
        let mut out = [0u64; 5];
        for i in 0..5 {
            // Add 2p to keep every limb positive, then reduce.
            out[i] = self.0[i] + 2 * P[i] - other.0[i];
        }
        Self(out).weak_reduce()
    }

    fn neg(self) -> Self {
        Self::ZERO.sub(self)
    }

    fn mul(self, other: Self) -> Self {
        let a = self.0;
        let b = other.0;
        let mut product = [0u128; 5];

        for i in 0..5 {
            for j in 0..5 {
                let value = u128::from(a[i]) * u128::from(b[j]);
                let k = i + j;
                if k < 5 {
                    product[k] += value;
                } else {
                    product[k - 5] += value * 19;
                }
            }
        }

        let mut out = [0u64; 5];
        let mut carry = 0u128;
        for i in 0..5 {
            let value = product[i] + carry;
            out[i] = (value as u64) & MASK51;
            carry = value >> 51;
        }
        out[0] += (carry as u64) * 19;

        let c0 = out[0] >> 51;
        out[0] &= MASK51;
        out[1] += c0;

        Self(out).weak_reduce()
    }

    fn square(self) -> Self {
        self.mul(self)
    }

    fn pow(self, exponent: [u64; 4], bits: usize) -> Self {
        let mut result = Self::ONE;
        for bit in (0..bits).rev() {
            result = result.square();
            if (exponent[bit / 64] >> (bit % 64)) & 1 == 1 {
                result = result.mul(self);
            }
        }
        result
    }

    fn invert(self) -> Self {
        // p - 2 = 2^255 - 21
        self.pow(INVERT_EXP, 255)
    }

    fn to_bytes(self) -> [u8; 32] {
        let limbs = self.canonical().0;
        let mut out = [0u8; 32];
        let mut accumulator = 0u128;
        let mut bits = 0usize;
        let mut index = 0usize;

        for limb in limbs {
            accumulator |= u128::from(limb) << bits;
            bits += 51;
            while bits >= 8 {
                out[index] = (accumulator & 0xff) as u8;
                accumulator >>= 8;
                bits -= 8;
                index += 1;
            }
        }

        debug_assert_eq!(index, 31);
        out[index] = (accumulator & 0xff) as u8;
        out
    }

    fn is_odd(self) -> bool {
        self.to_bytes()[0] & 1 == 1
    }
}

fn put_limbs(out: &mut [i32], value: Fe) {
    out.copy_from_slice(&limbs(value));
}

fn limbs(value: Fe) -> [i32; 10] {
    let bytes = value.to_bytes();
    let mut padded = [0u8; 40];
    padded[..32].copy_from_slice(&bytes);

    let mut out = [0i32; 10];
    let mut bit = 0usize;
    for (i, limb) in out.iter_mut().enumerate() {
        let width = if i & 1 == 0 { 26 } else { 25 };
        let byte = bit / 8;
        let shift = bit % 8;
        let mut word =
            u64::from_le_bytes(padded[byte..byte + 8].try_into().expect("8 bytes")) >> shift;
        if shift + width > 64 {
            word |= u64::from(padded[byte + 8]) << (64 - shift);
        }
        *limb = (word & ((1u64 << width) - 1)) as i32;
        bit += width;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_w8_table_matches_reference() {
        // Reference hash of the table generated by the original Python
        // implementation (and verified byte-for-byte against its .npy file).
        const REFERENCE: u64 = 0xa970_1f03_1e5d_2c0f;

        let table = generate(8);
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for value in table {
            for byte in value.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x100_0000_01b3);
            }
        }
        assert_eq!(hash, REFERENCE);
    }
}
