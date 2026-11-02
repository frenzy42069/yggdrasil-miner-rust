// Fast Yggdrasil key search.
//
// This is the OpenCL core from yggminer (the Python/OpenCL miner): instead of
// doing a full ref10 scalar multiplication and packing the point for every
// key, it uses a fixed-base window table, computes only the affine Y
// coordinate the address depends on, and inverts all Z coordinates of a batch
// with one Montgomery trick.  The host recomputes every reported key with
// ed25519-dalek before using it.
//
// Built with -DWINDOW_BITS=<bits> -DWINDOWS=<count> -DBATCH=<keys per work item>.
// Field arithmetic is the radix 2^25.5 representation from ref10.

typedef int fe[10];

// Filter flags shared with src/gpu.rs.
#define F_FORCE_ALL 1
#define F_HIGHEST 2
#define F_SHORTEST 4
#define F_LOWEST 8
#define F_COLON_REQUIRED 16

static inline void fe_carry(__private int *h, long h0, long h1, long h2, long h3, long h4,
                            long h5, long h6, long h7, long h8, long h9)
{
    long c;
    c = (h0 + (1L << 25)) >> 26; h1 += c; h0 -= c << 26;
    c = (h4 + (1L << 25)) >> 26; h5 += c; h4 -= c << 26;
    c = (h1 + (1L << 24)) >> 25; h2 += c; h1 -= c << 25;
    c = (h5 + (1L << 24)) >> 25; h6 += c; h5 -= c << 25;
    c = (h2 + (1L << 25)) >> 26; h3 += c; h2 -= c << 26;
    c = (h6 + (1L << 25)) >> 26; h7 += c; h6 -= c << 26;
    c = (h3 + (1L << 24)) >> 25; h4 += c; h3 -= c << 25;
    c = (h7 + (1L << 24)) >> 25; h8 += c; h7 -= c << 25;
    c = (h4 + (1L << 25)) >> 26; h5 += c; h4 -= c << 26;
    c = (h8 + (1L << 25)) >> 26; h9 += c; h8 -= c << 26;
    c = (h9 + (1L << 24)) >> 25; h0 += c * 19; h9 -= c << 25;
    c = (h0 + (1L << 25)) >> 26; h1 += c; h0 -= c << 26;
    h[0] = (int)h0; h[1] = (int)h1; h[2] = (int)h2; h[3] = (int)h3; h[4] = (int)h4;
    h[5] = (int)h5; h[6] = (int)h6; h[7] = (int)h7; h[8] = (int)h8; h[9] = (int)h9;
}

static inline void fe_mul(__private int *h, __private const int *f, __private const int *g)
{
    int f0 = f[0], f1 = f[1], f2 = f[2], f3 = f[3], f4 = f[4];
    int f5 = f[5], f6 = f[6], f7 = f[7], f8 = f[8], f9 = f[9];
    int g0 = g[0], g1 = g[1], g2 = g[2], g3 = g[3], g4 = g[4];
    int g5 = g[5], g6 = g[6], g7 = g[7], g8 = g[8], g9 = g[9];
    int g1_19 = 19 * g1, g2_19 = 19 * g2, g3_19 = 19 * g3, g4_19 = 19 * g4, g5_19 = 19 * g5;
    int g6_19 = 19 * g6, g7_19 = 19 * g7, g8_19 = 19 * g8, g9_19 = 19 * g9;
    int f1_2 = 2 * f1, f3_2 = 2 * f3, f5_2 = 2 * f5, f7_2 = 2 * f7, f9_2 = 2 * f9;
#define M(a, b) ((long)(a) * (long)(b))
    long h0 = M(f0, g0) + M(f1_2, g9_19) + M(f2, g8_19) + M(f3_2, g7_19) + M(f4, g6_19) + M(f5_2, g5_19) + M(f6, g4_19) + M(f7_2, g3_19) + M(f8, g2_19) + M(f9_2, g1_19);
    long h1 = M(f0, g1) + M(f1, g0) + M(f2, g9_19) + M(f3, g8_19) + M(f4, g7_19) + M(f5, g6_19) + M(f6, g5_19) + M(f7, g4_19) + M(f8, g3_19) + M(f9, g2_19);
    long h2 = M(f0, g2) + M(f1_2, g1) + M(f2, g0) + M(f3_2, g9_19) + M(f4, g8_19) + M(f5_2, g7_19) + M(f6, g6_19) + M(f7_2, g5_19) + M(f8, g4_19) + M(f9_2, g3_19);
    long h3 = M(f0, g3) + M(f1, g2) + M(f2, g1) + M(f3, g0) + M(f4, g9_19) + M(f5, g8_19) + M(f6, g7_19) + M(f7, g6_19) + M(f8, g5_19) + M(f9, g4_19);
    long h4 = M(f0, g4) + M(f1_2, g3) + M(f2, g2) + M(f3_2, g1) + M(f4, g0) + M(f5_2, g9_19) + M(f6, g8_19) + M(f7_2, g7_19) + M(f8, g6_19) + M(f9_2, g5_19);
    long h5 = M(f0, g5) + M(f1, g4) + M(f2, g3) + M(f3, g2) + M(f4, g1) + M(f5, g0) + M(f6, g9_19) + M(f7, g8_19) + M(f8, g7_19) + M(f9, g6_19);
    long h6 = M(f0, g6) + M(f1_2, g5) + M(f2, g4) + M(f3_2, g3) + M(f4, g2) + M(f5_2, g1) + M(f6, g0) + M(f7_2, g9_19) + M(f8, g8_19) + M(f9_2, g7_19);
    long h7 = M(f0, g7) + M(f1, g6) + M(f2, g5) + M(f3, g4) + M(f4, g3) + M(f5, g2) + M(f6, g1) + M(f7, g0) + M(f8, g9_19) + M(f9, g8_19);
    long h8 = M(f0, g8) + M(f1_2, g7) + M(f2, g6) + M(f3_2, g5) + M(f4, g4) + M(f5_2, g3) + M(f6, g2) + M(f7_2, g1) + M(f8, g0) + M(f9_2, g9_19);
    long h9 = M(f0, g9) + M(f1, g8) + M(f2, g7) + M(f3, g6) + M(f4, g5) + M(f5, g4) + M(f6, g3) + M(f7, g2) + M(f8, g1) + M(f9, g0);
#undef M
    fe_carry(h, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9);
}

static inline void fe_sq(__private int *h, __private const int *f)
{
    int f0 = f[0], f1 = f[1], f2 = f[2], f3 = f[3], f4 = f[4];
    int f5 = f[5], f6 = f[6], f7 = f[7], f8 = f[8], f9 = f[9];
    int f0_2 = 2 * f0, f1_2 = 2 * f1, f2_2 = 2 * f2, f3_2 = 2 * f3, f4_2 = 2 * f4;
    int f5_2 = 2 * f5, f6_2 = 2 * f6, f7_2 = 2 * f7;
    int f5_38 = 38 * f5, f6_19 = 19 * f6, f7_38 = 38 * f7, f8_19 = 19 * f8, f9_38 = 38 * f9;
#define M(a, b) ((long)(a) * (long)(b))
    long h0 = M(f0, f0) + M(f1_2, f9_38) + M(f2_2, f8_19) + M(f3_2, f7_38) + M(f4_2, f6_19) + M(f5, f5_38);
    long h1 = M(f0_2, f1) + M(f2, f9_38) + M(f3_2, f8_19) + M(f4, f7_38) + M(f5_2, f6_19);
    long h2 = M(f0_2, f2) + M(f1_2, f1) + M(f3_2, f9_38) + M(f4_2, f8_19) + M(f5_2, f7_38) + M(f6, f6_19);
    long h3 = M(f0_2, f3) + M(f1_2, f2) + M(f4, f9_38) + M(f5_2, f8_19) + M(f6, f7_38);
    long h4 = M(f0_2, f4) + M(f1_2, f3_2) + M(f2, f2) + M(f5_2, f9_38) + M(f6_2, f8_19) + M(f7, f7_38);
    long h5 = M(f0_2, f5) + M(f1_2, f4) + M(f2_2, f3) + M(f6, f9_38) + M(f7_2, f8_19);
    long h6 = M(f0_2, f6) + M(f1_2, f5_2) + M(f2_2, f4) + M(f3_2, f3) + M(f7_2, f9_38) + M(f8, f8_19);
    long h7 = M(f0_2, f7) + M(f1_2, f6) + M(f2_2, f5) + M(f3_2, f4) + M(f8, f9_38);
    long h8 = M(f0_2, f8) + M(f1_2, f7_2) + M(f2_2, f6) + M(f3_2, f5_2) + M(f4, f4) + M(f9, f9_38);
    long h9 = M(f0_2, f9) + M(f1_2, f8) + M(f2_2, f7) + M(f3_2, f6) + M(f4_2, f5);
#undef M
    fe_carry(h, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9);
}

static inline void fe_copy(__private int *h, __private const int *f)
{
    for (int i = 0; i < 10; i++) h[i] = f[i];
}

static inline void fe_add(__private int *h, __private const int *f, __private const int *g)
{
    for (int i = 0; i < 10; i++) h[i] = f[i] + g[i];
}

static inline void fe_sub(__private int *h, __private const int *f, __private const int *g)
{
    for (int i = 0; i < 10; i++) h[i] = f[i] - g[i];
}

static inline void fe_sq_n(__private int *h, __private const int *f, int n)
{
    fe_sq(h, f);
    for (int i = 1; i < n; i++) fe_sq(h, h);
}

// z^(p-2)
static void fe_invert(__private int *out, __private const int *z)
{
    fe t0, t1, t2, t3;
    fe_sq(t0, z);
    fe_sq_n(t1, t0, 2);
    fe_mul(t1, z, t1);
    fe_mul(t0, t0, t1);
    fe_sq(t2, t0);
    fe_mul(t1, t1, t2);
    fe_sq_n(t2, t1, 5);
    fe_mul(t1, t2, t1);
    fe_sq_n(t2, t1, 10);
    fe_mul(t2, t2, t1);
    fe_sq_n(t3, t2, 20);
    fe_mul(t2, t3, t2);
    fe_sq_n(t2, t2, 10);
    fe_mul(t1, t2, t1);
    fe_sq_n(t2, t1, 50);
    fe_mul(t2, t2, t1);
    fe_sq_n(t3, t2, 100);
    fe_mul(t2, t3, t2);
    fe_sq_n(t2, t2, 50);
    fe_mul(t1, t2, t1);
    fe_sq_n(t1, t1, 5);
    fe_mul(out, t1, t0);
}

// Full reduction, returning the low 192 bits as little-endian 64-bit words.
// These are bytes 0..23 of the packed public key, which is all the address
// needs (the remaining bits are the high Y bits and the X sign bit).
static inline void fe_low_words(__private const int *f, __private ulong *w)
{
    int h[10];
    for (int i = 0; i < 10; i++) h[i] = f[i];
    int q = (19 * h[9] + (1 << 24)) >> 25;
    for (int i = 0; i < 10; i++) q = (h[i] + q) >> ((i & 1) ? 25 : 26);
    h[0] += 19 * q;
    for (int i = 0; i < 9; i++) {
        int bits = (i & 1) ? 25 : 26;
        int carry = h[i] >> bits;
        h[i + 1] += carry;
        h[i] -= carry << bits;
    }
    h[9] &= (1 << 25) - 1;
    w[0] = (ulong)(uint)h[0] | ((ulong)(uint)h[1] << 26) | ((ulong)(uint)h[2] << 51);
    w[1] = ((ulong)(uint)h[2] >> 13) | ((ulong)(uint)h[3] << 13) | ((ulong)(uint)h[4] << 38);
    w[2] = (ulong)(uint)h[5] | ((ulong)(uint)h[6] << 25) | ((ulong)(uint)h[7] << 51);
}

static inline ulong bswap64(ulong x)
{
    x = ((x & 0x00FF00FF00FF00FFUL) << 8) | ((x >> 8) & 0x00FF00FF00FF00FFUL);
    x = ((x & 0x0000FFFF0000FFFFUL) << 16) | ((x >> 16) & 0x0000FFFF0000FFFFUL);
    return (x << 32) | (x >> 32);
}

#define ROR(x, n) rotate((ulong)(x), (ulong)(64 - (n)))
#define SHA_ROUND(a, b, c, d, e, f, g, h, k, w) \
    t = h + (ROR(e, 14) ^ ROR(e, 18) ^ ROR(e, 41)) + bitselect(g, f, e) + k + w; \
    d += t; \
    h = t + (ROR(a, 28) ^ ROR(a, 34) ^ ROR(a, 39)) + bitselect(a, b, a ^ c);
#define SHA_NEXT(w, w1, w9, w14) \
    w += (ROR(w1, 1) ^ ROR(w1, 8) ^ (w1 >> 7)) + w9 + (ROR(w14, 19) ^ ROR(w14, 61) ^ (w14 >> 6));

// SHA-512 of a 32-byte seed; out is the first 32 digest bytes as little-endian words.
// Unrolled so round constants are immediates and the zero message words fold away.
static inline void sha512_seed(ulong s0, ulong s1, ulong s2, ulong s3, __private ulong *out)
{
    ulong w0 = s0, w1 = s1, w2 = s2, w3 = s3, w4 = 0x8000000000000000UL, w5 = 0, w6 = 0, w7 = 0;
    ulong w8 = 0, w9 = 0, w10 = 0, w11 = 0, w12 = 0, w13 = 0, w14 = 0, w15 = 256;
    ulong a = 0x6a09e667f3bcc908UL, b = 0xbb67ae8584caa73bUL, c = 0x3c6ef372fe94f82bUL, d = 0xa54ff53a5f1d36f1UL;
    ulong e = 0x510e527fade682d1UL, f = 0x9b05688c2b3e6c1fUL, g = 0x1f83d9abfb41bd6bUL, h = 0x5be0cd19137e2179UL, t;
    SHA_ROUND(a, b, c, d, e, f, g, h, 0x428a2f98d728ae22UL, w0)
    SHA_ROUND(h, a, b, c, d, e, f, g, 0x7137449123ef65cdUL, w1)
    SHA_ROUND(g, h, a, b, c, d, e, f, 0xb5c0fbcfec4d3b2fUL, w2)
    SHA_ROUND(f, g, h, a, b, c, d, e, 0xe9b5dba58189dbbcUL, w3)
    SHA_ROUND(e, f, g, h, a, b, c, d, 0x3956c25bf348b538UL, w4)
    SHA_ROUND(d, e, f, g, h, a, b, c, 0x59f111f1b605d019UL, w5)
    SHA_ROUND(c, d, e, f, g, h, a, b, 0x923f82a4af194f9bUL, w6)
    SHA_ROUND(b, c, d, e, f, g, h, a, 0xab1c5ed5da6d8118UL, w7)
    SHA_ROUND(a, b, c, d, e, f, g, h, 0xd807aa98a3030242UL, w8)
    SHA_ROUND(h, a, b, c, d, e, f, g, 0x12835b0145706fbeUL, w9)
    SHA_ROUND(g, h, a, b, c, d, e, f, 0x243185be4ee4b28cUL, w10)
    SHA_ROUND(f, g, h, a, b, c, d, e, 0x550c7dc3d5ffb4e2UL, w11)
    SHA_ROUND(e, f, g, h, a, b, c, d, 0x72be5d74f27b896fUL, w12)
    SHA_ROUND(d, e, f, g, h, a, b, c, 0x80deb1fe3b1696b1UL, w13)
    SHA_ROUND(c, d, e, f, g, h, a, b, 0x9bdc06a725c71235UL, w14)
    SHA_ROUND(b, c, d, e, f, g, h, a, 0xc19bf174cf692694UL, w15)
    SHA_NEXT(w0, w1, w9, w14)
    SHA_ROUND(a, b, c, d, e, f, g, h, 0xe49b69c19ef14ad2UL, w0)
    SHA_NEXT(w1, w2, w10, w15)
    SHA_ROUND(h, a, b, c, d, e, f, g, 0xefbe4786384f25e3UL, w1)
    SHA_NEXT(w2, w3, w11, w0)
    SHA_ROUND(g, h, a, b, c, d, e, f, 0x0fc19dc68b8cd5b5UL, w2)
    SHA_NEXT(w3, w4, w12, w1)
    SHA_ROUND(f, g, h, a, b, c, d, e, 0x240ca1cc77ac9c65UL, w3)
    SHA_NEXT(w4, w5, w13, w2)
    SHA_ROUND(e, f, g, h, a, b, c, d, 0x2de92c6f592b0275UL, w4)
    SHA_NEXT(w5, w6, w14, w3)
    SHA_ROUND(d, e, f, g, h, a, b, c, 0x4a7484aa6ea6e483UL, w5)
    SHA_NEXT(w6, w7, w15, w4)
    SHA_ROUND(c, d, e, f, g, h, a, b, 0x5cb0a9dcbd41fbd4UL, w6)
    SHA_NEXT(w7, w8, w0, w5)
    SHA_ROUND(b, c, d, e, f, g, h, a, 0x76f988da831153b5UL, w7)
    SHA_NEXT(w8, w9, w1, w6)
    SHA_ROUND(a, b, c, d, e, f, g, h, 0x983e5152ee66dfabUL, w8)
    SHA_NEXT(w9, w10, w2, w7)
    SHA_ROUND(h, a, b, c, d, e, f, g, 0xa831c66d2db43210UL, w9)
    SHA_NEXT(w10, w11, w3, w8)
    SHA_ROUND(g, h, a, b, c, d, e, f, 0xb00327c898fb213fUL, w10)
    SHA_NEXT(w11, w12, w4, w9)
    SHA_ROUND(f, g, h, a, b, c, d, e, 0xbf597fc7beef0ee4UL, w11)
    SHA_NEXT(w12, w13, w5, w10)
    SHA_ROUND(e, f, g, h, a, b, c, d, 0xc6e00bf33da88fc2UL, w12)
    SHA_NEXT(w13, w14, w6, w11)
    SHA_ROUND(d, e, f, g, h, a, b, c, 0xd5a79147930aa725UL, w13)
    SHA_NEXT(w14, w15, w7, w12)
    SHA_ROUND(c, d, e, f, g, h, a, b, 0x06ca6351e003826fUL, w14)
    SHA_NEXT(w15, w0, w8, w13)
    SHA_ROUND(b, c, d, e, f, g, h, a, 0x142929670a0e6e70UL, w15)
    SHA_NEXT(w0, w1, w9, w14)
    SHA_ROUND(a, b, c, d, e, f, g, h, 0x27b70a8546d22ffcUL, w0)
    SHA_NEXT(w1, w2, w10, w15)
    SHA_ROUND(h, a, b, c, d, e, f, g, 0x2e1b21385c26c926UL, w1)
    SHA_NEXT(w2, w3, w11, w0)
    SHA_ROUND(g, h, a, b, c, d, e, f, 0x4d2c6dfc5ac42aedUL, w2)
    SHA_NEXT(w3, w4, w12, w1)
    SHA_ROUND(f, g, h, a, b, c, d, e, 0x53380d139d95b3dfUL, w3)
    SHA_NEXT(w4, w5, w13, w2)
    SHA_ROUND(e, f, g, h, a, b, c, d, 0x650a73548baf63deUL, w4)
    SHA_NEXT(w5, w6, w14, w3)
    SHA_ROUND(d, e, f, g, h, a, b, c, 0x766a0abb3c77b2a8UL, w5)
    SHA_NEXT(w6, w7, w15, w4)
    SHA_ROUND(c, d, e, f, g, h, a, b, 0x81c2c92e47edaee6UL, w6)
    SHA_NEXT(w7, w8, w0, w5)
    SHA_ROUND(b, c, d, e, f, g, h, a, 0x92722c851482353bUL, w7)
    SHA_NEXT(w8, w9, w1, w6)
    SHA_ROUND(a, b, c, d, e, f, g, h, 0xa2bfe8a14cf10364UL, w8)
    SHA_NEXT(w9, w10, w2, w7)
    SHA_ROUND(h, a, b, c, d, e, f, g, 0xa81a664bbc423001UL, w9)
    SHA_NEXT(w10, w11, w3, w8)
    SHA_ROUND(g, h, a, b, c, d, e, f, 0xc24b8b70d0f89791UL, w10)
    SHA_NEXT(w11, w12, w4, w9)
    SHA_ROUND(f, g, h, a, b, c, d, e, 0xc76c51a30654be30UL, w11)
    SHA_NEXT(w12, w13, w5, w10)
    SHA_ROUND(e, f, g, h, a, b, c, d, 0xd192e819d6ef5218UL, w12)
    SHA_NEXT(w13, w14, w6, w11)
    SHA_ROUND(d, e, f, g, h, a, b, c, 0xd69906245565a910UL, w13)
    SHA_NEXT(w14, w15, w7, w12)
    SHA_ROUND(c, d, e, f, g, h, a, b, 0xf40e35855771202aUL, w14)
    SHA_NEXT(w15, w0, w8, w13)
    SHA_ROUND(b, c, d, e, f, g, h, a, 0x106aa07032bbd1b8UL, w15)
    SHA_NEXT(w0, w1, w9, w14)
    SHA_ROUND(a, b, c, d, e, f, g, h, 0x19a4c116b8d2d0c8UL, w0)
    SHA_NEXT(w1, w2, w10, w15)
    SHA_ROUND(h, a, b, c, d, e, f, g, 0x1e376c085141ab53UL, w1)
    SHA_NEXT(w2, w3, w11, w0)
    SHA_ROUND(g, h, a, b, c, d, e, f, 0x2748774cdf8eeb99UL, w2)
    SHA_NEXT(w3, w4, w12, w1)
    SHA_ROUND(f, g, h, a, b, c, d, e, 0x34b0bcb5e19b48a8UL, w3)
    SHA_NEXT(w4, w5, w13, w2)
    SHA_ROUND(e, f, g, h, a, b, c, d, 0x391c0cb3c5c95a63UL, w4)
    SHA_NEXT(w5, w6, w14, w3)
    SHA_ROUND(d, e, f, g, h, a, b, c, 0x4ed8aa4ae3418acbUL, w5)
    SHA_NEXT(w6, w7, w15, w4)
    SHA_ROUND(c, d, e, f, g, h, a, b, 0x5b9cca4f7763e373UL, w6)
    SHA_NEXT(w7, w8, w0, w5)
    SHA_ROUND(b, c, d, e, f, g, h, a, 0x682e6ff3d6b2b8a3UL, w7)
    SHA_NEXT(w8, w9, w1, w6)
    SHA_ROUND(a, b, c, d, e, f, g, h, 0x748f82ee5defb2fcUL, w8)
    SHA_NEXT(w9, w10, w2, w7)
    SHA_ROUND(h, a, b, c, d, e, f, g, 0x78a5636f43172f60UL, w9)
    SHA_NEXT(w10, w11, w3, w8)
    SHA_ROUND(g, h, a, b, c, d, e, f, 0x84c87814a1f0ab72UL, w10)
    SHA_NEXT(w11, w12, w4, w9)
    SHA_ROUND(f, g, h, a, b, c, d, e, 0x8cc702081a6439ecUL, w11)
    SHA_NEXT(w12, w13, w5, w10)
    SHA_ROUND(e, f, g, h, a, b, c, d, 0x90befffa23631e28UL, w12)
    SHA_NEXT(w13, w14, w6, w11)
    SHA_ROUND(d, e, f, g, h, a, b, c, 0xa4506cebde82bde9UL, w13)
    SHA_NEXT(w14, w15, w7, w12)
    SHA_ROUND(c, d, e, f, g, h, a, b, 0xbef9a3f7b2c67915UL, w14)
    SHA_NEXT(w15, w0, w8, w13)
    SHA_ROUND(b, c, d, e, f, g, h, a, 0xc67178f2e372532bUL, w15)
    SHA_NEXT(w0, w1, w9, w14)
    SHA_ROUND(a, b, c, d, e, f, g, h, 0xca273eceea26619cUL, w0)
    SHA_NEXT(w1, w2, w10, w15)
    SHA_ROUND(h, a, b, c, d, e, f, g, 0xd186b8c721c0c207UL, w1)
    SHA_NEXT(w2, w3, w11, w0)
    SHA_ROUND(g, h, a, b, c, d, e, f, 0xeada7dd6cde0eb1eUL, w2)
    SHA_NEXT(w3, w4, w12, w1)
    SHA_ROUND(f, g, h, a, b, c, d, e, 0xf57d4f7fee6ed178UL, w3)
    SHA_NEXT(w4, w5, w13, w2)
    SHA_ROUND(e, f, g, h, a, b, c, d, 0x06f067aa72176fbaUL, w4)
    SHA_NEXT(w5, w6, w14, w3)
    SHA_ROUND(d, e, f, g, h, a, b, c, 0x0a637dc5a2c898a6UL, w5)
    SHA_NEXT(w6, w7, w15, w4)
    SHA_ROUND(c, d, e, f, g, h, a, b, 0x113f9804bef90daeUL, w6)
    SHA_NEXT(w7, w8, w0, w5)
    SHA_ROUND(b, c, d, e, f, g, h, a, 0x1b710b35131c471bUL, w7)
    SHA_NEXT(w8, w9, w1, w6)
    SHA_ROUND(a, b, c, d, e, f, g, h, 0x28db77f523047d84UL, w8)
    SHA_NEXT(w9, w10, w2, w7)
    SHA_ROUND(h, a, b, c, d, e, f, g, 0x32caab7b40c72493UL, w9)
    SHA_NEXT(w10, w11, w3, w8)
    SHA_ROUND(g, h, a, b, c, d, e, f, 0x3c9ebe0a15c9bebcUL, w10)
    SHA_NEXT(w11, w12, w4, w9)
    SHA_ROUND(f, g, h, a, b, c, d, e, 0x431d67c49c100d4cUL, w11)
    SHA_NEXT(w12, w13, w5, w10)
    SHA_ROUND(e, f, g, h, a, b, c, d, 0x4cc5d4becb3e42b6UL, w12)
    SHA_NEXT(w13, w14, w6, w11)
    SHA_ROUND(d, e, f, g, h, a, b, c, 0x597f299cfc657e2aUL, w13)
    SHA_NEXT(w14, w15, w7, w12)
    SHA_ROUND(c, d, e, f, g, h, a, b, 0x5fcb6fab3ad6faecUL, w14)
    SHA_NEXT(w15, w0, w8, w13)
    SHA_ROUND(b, c, d, e, f, g, h, a, 0x6c44198c4a475817UL, w15)
    out[0] = bswap64(a + 0x6a09e667f3bcc908UL);
    out[1] = bswap64(b + 0xbb67ae8584caa73bUL);
    out[2] = bswap64(c + 0x3c6ef372fe94f82bUL);
    out[3] = bswap64(d + 0xa54ff53a5f1d36f1UL);
}

static inline void load_entry(__global const int *table, __private const ulong *scalar, int window,
                              __private int *q0, __private int *q1, __private int *q2)
{
    int bit = window * WINDOW_BITS, word = bit >> 6, shift = bit & 63;
    ulong v = scalar[word] >> shift;
    if (shift + WINDOW_BITS > 64 && word < 3) v |= scalar[word + 1] << (64 - shift);
    v &= (1UL << WINDOW_BITS) - 1;

    __global const int *entry = table + 30 * (((ulong)window << WINDOW_BITS) + v);
    for (int i = 0; i < 10; i++) {
        q0[i] = entry[i];
        q1[i] = entry[i + 10];
        q2[i] = entry[i + 20];
    }
}

// Y and Z of scalar * B; the affine y is all the address depends on.
static inline void scalar_base(__global const int *table, __private const ulong *scalar,
                               __private int *out_y, __private int *out_z)
{
    fe x, y, z, t, a, b, c, d, e, f, g, h, q0, q1, q2;

    // The first addition starts from the identity, where a = q1, b = q0, c = 0 and
    // d = 2 leave one multiplication out of seven. Operand order matters: 19 * g
    // must fit in an int, so the larger value h goes first.
    load_entry(table, scalar, 0, q0, q1, q2);
    fe_sub(e, q0, q1);
    fe_add(h, q0, q1);
    fe_mul(t, h, e);
    fe_add(x, e, e);
    fe_add(y, h, h);
    for (int i = 1; i < 10; i++) z[i] = 0;
    z[0] = 4;

    for (int window = 1; window < WINDOWS; window++) {
        load_entry(table, scalar, window, q0, q1, q2);
        fe_sub(a, y, x);
        fe_mul(a, a, q1);
        fe_add(b, y, x);
        fe_mul(b, b, q0);
        fe_mul(c, t, q2);
        fe_add(d, z, z);
        fe_sub(e, b, a);
        fe_sub(f, d, c);
        fe_add(g, d, c);
        fe_add(h, b, a);
        fe_mul(y, g, h);
        fe_mul(z, f, g);
        // Only y and z of the final point are used.
        if (window < WINDOWS - 1) {
            fe_mul(x, e, f);
            fe_mul(t, e, h);
        }
    }
    fe_copy(out_y, y);
    fe_copy(out_z, z);
}

// Canonical IPv6 text length and the longest zero-group run.  The first group
// is always "2xx", so only the seven groups after it are inspected.
static inline void address_metrics(ulong a0, ulong a1, int *length, int *zero_run)
{
    uint groups[7] = {
        (uint)(a0 >> 48), (uint)(a0 >> 32) & 0xFFFF, (uint)(a0 >> 16) & 0xFFFF, (uint)a0 & 0xFFFF,
        (uint)(a1 >> 48), (uint)(a1 >> 32) & 0xFFFF, (uint)(a1 >> 16) & 0xFFFF
    };
    int total = 3 + 7, run = 0, best = 0, best_at_end = 0;
    for (int i = 0; i < 7; i++) {
        uint value = groups[i];
        total += 1 + (value >= 0x10) + (value >= 0x100) + (value >= 0x1000);
        run = value ? 0 : run + 1;
        if (run > best) {
            best = run;
            best_at_end = i == 6;
        }
    }
    if (best >= 2) total -= 2 * best - (best_at_end ? 2 : 1);
    *length = total;
    *zero_run = best;
}

__kernel void mine(__global const int *table,
                   ulong seed0, ulong seed1, ulong seed2, ulong seed3, ulong first_id,
                   int flags, int min_height, int max_length, ulong lowest_packed,
                   __global uint *hits, __global volatile uint *hit_count, uint capacity)
{
    fe ys[BATCH], zs[BATCH], products[BATCH], inverse, z_inverse, affine_y;
    ulong base = first_id + (ulong)get_global_id(0) * BATCH;

    for (int i = 0; i < BATCH; i++) {
        ulong scalar[4];
        sha512_seed(seed0 ^ (base + i), seed1, seed2, seed3, scalar);
        scalar[0] &= ~7UL;
        scalar[3] = (scalar[3] & 0x3FFFFFFFFFFFFFFFUL) | 0x4000000000000000UL;
        scalar_base(table, scalar, ys[i], zs[i]);
        if (i) fe_mul(products[i], products[i - 1], zs[i]);
        else fe_copy(products[0], zs[0]);
    }

    fe_invert(inverse, products[BATCH - 1]);

    for (int i = BATCH - 1; i >= 0; i--) {
        if (i) {
            fe_mul(z_inverse, inverse, products[i - 1]);
            fe_mul(inverse, inverse, zs[i]);
        } else {
            fe_copy(z_inverse, inverse);
        }
        fe_mul(affine_y, ys[i], z_inverse);

        ulong words[3];
        fe_low_words(affine_y, words);
        ulong k0 = bswap64(words[0]), k1 = bswap64(words[1]), k2 = bswap64(words[2]);

        int hit;
        if (k0 == 0) {
            // 64+ leading zeros: vanishingly rare.  Let the host decide.
            hit = 1;
        } else {
            int zeros = (int)clz(k0), keep = zeros + 1;
            ulong a0 = ~(keep == 64 ? k1 : (k0 << keep) | (k1 >> (64 - keep)));
            ulong a1 = ~(keep == 64 ? k2 : (k1 << keep) | (k2 >> (64 - keep)));
            a1 &= 0xFFFFFFFFFFFF0000UL;

            if (flags & F_FORCE_ALL) {
                hit = 1;
            } else {
                int length = 0, zero_run = 0;
                if (flags & (F_SHORTEST | F_COLON_REQUIRED))
                    address_metrics(a0, a1, &length, &zero_run);

                hit = 0;
                if (!(flags & F_COLON_REQUIRED) || zero_run >= 2) {
                    if ((flags & F_HIGHEST) && zeros > min_height) hit = 1;
                    if ((flags & F_SHORTEST) && length < max_length) hit = 1;
                    if (flags & F_LOWEST) {
                        // Same ranking key as handler.rs: height + first 56 tail bits.
                        ulong lv = ((ulong)zeros << 56) | (a0 >> 8);
                        if (lv < lowest_packed) hit = 1;
                    }
                }
            }
        }

        if (hit) {
            uint slot = atomic_inc(hit_count);
            if (slot < capacity) hits[slot] = (uint)(base + i - first_id);
        }
    }
}
