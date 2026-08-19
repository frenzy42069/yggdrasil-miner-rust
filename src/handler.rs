use chrono::Utc;
use ed25519_dalek::SigningKey;
use hex::ToHex;

use std::fs;
use std::io::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, Ordering};

pub fn handle_keypair(
    seed: &[u8],
    pk: &[u8],
    most_high: &AtomicU8,
    most_short: &AtomicU8,
    highest: bool,
    shortest: bool,
    csv_file: Option<&Mutex<fs::File>>,
) {
    let height = height_of_pubkey(pk);

    if !highest || most_high.load(Ordering::Relaxed) < height {
        let addr = address_for_pubkey(pk);
        let str_addr = addr.to_string();
        #[allow(clippy::cast_possible_truncation)]
        let length = str_addr.len() as u8;

        if let segs = addr.segments()
            && segs.contains(&0x0)
            && (!highest || most_high.fetch_max(height, Ordering::SeqCst) < height)
            && (!shortest || most_short.fetch_min(length, Ordering::SeqCst) >= length)
        {
            {
                let mut fixed_seed = [0; 32];
                fixed_seed.copy_from_slice(seed);
                let signing_key = SigningKey::from_bytes(&fixed_seed);
                let verifying_key_bytes = signing_key.verifying_key().to_bytes();
                assert_eq!(verifying_key_bytes, pk);
            }

            let mut sk = [0u8; 64];
            sk[..32].copy_from_slice(seed);
            sk[32..].copy_from_slice(pk);
            let privkey_str: String = sk.encode_hex();

            {
                let mut lock = std::io::stdout().lock();
                writeln!(lock, "=======================================").unwrap();
                writeln!(lock, "PrivateKey: {privkey_str}").unwrap();
                writeln!(lock, "Address: {str_addr}").unwrap();
                writeln!(lock, "Height: {height}").unwrap();
                if shortest {
                    writeln!(lock, "Size: {length}").unwrap();
                }
                writeln!(lock, "=======================================").unwrap();
            }

            if let Some(csv_file) = csv_file {
                writeln!(
                    csv_file.lock().unwrap(),
                    "{},{str_addr},{height},{privkey_str}",
                    Utc::now().timestamp()
                )
                .unwrap();
            }
        }
    }
}

/// Height is a number of leading zeroes of a public key
fn height_of_pubkey(pk: &[u8]) -> u8 {
    let mut zeros = 0;
    for b in pk {
        let z = b.leading_zeros();
        zeros += u8::try_from(z).unwrap();
        if z != 8 {
            break;
        }
    }
    zeros
}

fn address_for_pubkey(pk: &[u8]) -> std::net::Ipv6Addr {
    let zeros = height_of_pubkey(pk);
    let mut buf = [0u8; 16];
    buf[0] = 0x02;
    buf[1] = zeros;
    for (src, trg) in pk[((zeros / 8) as usize)..]
        .windows(2)
        .zip(buf[2..].iter_mut())
    {
        *trg = src[0].wrapping_shl(u32::from((zeros + 1) % 8))
            ^ src[1].wrapping_shr(8 - u32::from((zeros + 1) % 8))
            ^ 0xFF;
    }
    std::net::Ipv6Addr::from(buf)
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::address_for_pubkey;

    #[test]
    fn test_address_for_pubkey() {
        assert_eq!(
            address_for_pubkey(
                hex::decode("000000000c4f58e09d19592f242951e6aa3185bd5ec6b95c0d56c93ae1268cbd")
                    .unwrap()
                    .as_slice()
            ),
            std::net::Ipv6Addr::from_str("224:7614:e3ec:5cd4:da1b:7ad5:c32a:b9cf").unwrap()
        );
    }
}
