__kernel void generate_pubkey(
    __global uchar *results __attribute__((aligned(32))),
    __global uchar *heights,
    __global uchar *keys __attribute__((aligned(32))),
    const uchar min_height)
{
  size_t const thread = get_global_id(0);
  uchar key[32];
  for (size_t i = 0; i < 32; i++) key[i] = keys[thread * 32 + i];

  uchar hash[64];
  sha512_hash(key, hash);
  hash[0] &= 248; hash[31] &= 63; hash[31] |= 64;

  bignum256modm a;
  ge25519 ALIGN(16) A;
  expand256_modm(a, hash, 32);
  ge25519_scalarmult_base_niels(&A, a);

  uchar pubkey[32];
  ge25519_pack(pubkey, &A);

  /* compute leading-zero-bit count */
  uchar h = 0;
  uchar stopped = 0;
  for (int i = 0; i < 32; i++) {
    uchar b = pubkey[i];
    for (int j = 7; j >= 0; j--) {
      if ((b >> j) & 1) { stopped = 1; break; }
      h++;
    }
    if (stopped) break;
  }
  heights[thread] = h;

  if (h >= min_height) {
    for (size_t i = 0; i < 32; i++) results[thread * 32 + i] = pubkey[i];
  }
}
