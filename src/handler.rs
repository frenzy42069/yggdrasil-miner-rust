use chrono::Utc;
use ed25519_dalek::SigningKey;
use hex::ToHex;

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

#[derive(Clone, Copy, Debug, Default)]
pub struct SearchConfig {
    pub highest: bool,
    pub lowest: bool,
    pub shortest: bool,
    pub colon: bool,
    pub both_colon: bool,
}

impl SearchConfig {
    pub fn needs_segs(&self) -> bool {
        self.shortest || self.lowest || self.colon
    }
    pub fn any(&self) -> bool {
        self.highest || self.lowest || self.shortest
    }
}

/* packed thresholds, one byte per category:
0: shortest (any)
1: highest  (any)
2: shortest (colon)
3: highest  (colon) */
const INIT_PACKED: u64 = 0xFF | (0x00u64 << 8) | (0xFFu64 << 16) | (0x00u64 << 24);

const I_SHORTEST: u32 = 0;
const I_HIGHEST: u32 = 1;
const I_SHORT_C: u32 = 2;
const I_HIGH_C: u32 = 3;

#[inline]
fn get_byte(p: u64, i: u32) -> u8 {
    ((p >> (i * 8)) & 0xFF) as u8
}

#[inline]
fn set_min(p: u64, i: u32, v: u8) -> u64 {
    let s = i * 8;
    let o = ((p >> s) & 0xFF) as u8;
    if o <= v {
        p
    } else {
        (p & !(0xFFu64 << s)) | ((v as u64) << s)
    }
}

#[inline]
fn set_max(p: u64, i: u32, v: u8) -> u64 {
    let s = i * 8;
    let o = ((p >> s) & 0xFF) as u8;
    if o >= v {
        p
    } else {
        (p & !(0xFFu64 << s)) | ((v as u64) << s)
    }
}

pub struct ParetoPoint {
    pub height: u8,
    pub length: u8,
    pub addr: String,
    pub pubkey: String,
}

#[derive(Default)]
pub struct ParetoFront {
    points: Vec<ParetoPoint>,
    max_points: Option<usize>,
}

impl ParetoFront {
    pub fn new(max_points: Option<usize>) -> Self {
        Self { points: Vec::new(), max_points }
    }

    pub fn try_add(&mut self, h: u8, l: u8, addr: &str, pk: &str) -> bool {
        for p in &self.points {
            if p.height >= h && p.length <= l && (p.height > h || p.length < l) {
                return false;
            }
        }
        self.points
            .retain(|p| !(h >= p.height && l <= p.length && (h > p.height || l < p.length)));
        if self.points.iter().any(|p| p.height == h && p.length == l) {
            return false;
        }
        self.points.push(ParetoPoint {
            height: h,
            length: l,
            addr: addr.into(),
            pubkey: pk.into(),
        });
        if let Some(max) = self.max_points {
            if self.points.len() > max {
                // Sort and keep best max points
                self.points.sort_by_key(|p| (std::cmp::Reverse(p.height), p.length));
                self.points.truncate(max);
            }
        }
        true
    }

    pub fn render(&self) -> String {
        let mut v: Vec<&ParetoPoint> = self.points.iter().collect();
        v.sort_by_key(|p| (p.height, p.length));
        let mut s = String::from("pareto_frontier:\n");
        for p in v {
            s.push_str(&format!(
                "  height={:3} length={:2}\n    addr={}\n    pubkey={}\n",
                p.height, p.length, p.addr, p.pubkey
            ));
        }
        s
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

#[derive(Clone, Debug)]
pub struct TopNEntry {
    pub key: u64,
    pub addr_val: u128,
    pub addr_str: String,
    pub pubkey_str: String,
    pub privkey_str: String,
}

pub struct TopN {
    n: usize,
    items: Mutex<Vec<TopNEntry>>,
    threshold: AtomicU64,
}

impl TopN {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            items: Mutex::new(Vec::new()),
            threshold: AtomicU64::new(0),
        }
    }

    pub fn try_insert(&self, key: u64, addr_val: u128, addr_str: String, pubkey_str: String, privkey_str: String) -> bool {
        if self.n == 0 {
            return false;
        }
        let thr = self.threshold.load(Ordering::Relaxed);
        if thr > 0 && key <= thr {
            return false;
        }

        let mut v = self.items.lock().unwrap();
        if v.iter().any(|e| e.addr_val == addr_val) {
            return false;
        }

        let idx = v.partition_point(|e| e.key > key);
        v.insert(idx, TopNEntry { key, addr_val, addr_str, pubkey_str, privkey_str });
        if v.len() > self.n {
            v.truncate(self.n);
        }
        if v.len() >= self.n {
            self.threshold.store(v[self.n - 1].key, Ordering::Relaxed);
        }
        true
    }

    pub fn get_items(&self) -> Vec<TopNEntry> {
        self.items.lock().unwrap().clone()
    }

    pub fn load_from(&self, entries: Vec<TopNEntry>) {
        let mut v = self.items.lock().unwrap();
        v.clear();
        v.extend(entries);
        v.sort_by_key(|e| std::cmp::Reverse(e.key));
        if v.len() > self.n {
            v.truncate(self.n);
        }
        if v.len() >= self.n {
            self.threshold.store(v[self.n - 1].key, Ordering::Relaxed);
        } else if !v.is_empty() {
            self.threshold.store(v.last().unwrap().key, Ordering::Relaxed);
        }
    }
}

pub struct Records {
    pub packed: AtomicU64,
    pub lowest_packed: AtomicU64,
    pub lowest_colon_packed: AtomicU64,
    pub lowest_addr: Mutex<u128>,
    pub lowest_colon_addr: Mutex<u128>,
    pub pareto: Mutex<ParetoFront>,
    pub pareto_empty: AtomicBool,
    pub highest_top_n: TopN,
    pub lowest_top_n: TopN,
    pub shortest_top_n: TopN,
}

impl Records {
    pub fn new() -> Self {
        Self::with_top_n(1, 1, 1, None)
    }

    pub fn with_top_n(highest_n: usize, lowest_n: usize, shortest_n: usize, pareto_max: Option<usize>) -> Self {
        Self {
            packed: AtomicU64::new(INIT_PACKED),
            lowest_packed: AtomicU64::new(u64::MAX),
            lowest_colon_packed: AtomicU64::new(u64::MAX),
            lowest_addr: Mutex::new(u128::MAX),
            lowest_colon_addr: Mutex::new(u128::MAX),
            pareto: Mutex::new(ParetoFront::new(pareto_max)),
            pareto_empty: AtomicBool::new(true),
            highest_top_n: TopN::new(highest_n),
            lowest_top_n: TopN::new(lowest_n),
            shortest_top_n: TopN::new(shortest_n),
        }
    }

    pub fn load_from_files(&self, files: &RecordFiles, cfg: &SearchConfig) {
        let mut pareto_empty = true;

        // Load top N files first (new format)
        if cfg.highest {
            if let Some(p) = &files.highest {
                if let Ok(content) = fs::read_to_string(p) {
                    let entries = parse_top_n_file(&content);
                    if !entries.is_empty() {
                        self.highest_top_n.load_from(entries);
                        // Also update packed threshold from best entry
                        if let Some(best) = self.highest_top_n.get_items().first() {
                            let idx = if cfg.colon { I_HIGH_C } else { I_HIGHEST };
                            let mut p = self.packed.load(Ordering::Relaxed);
                            p = set_max(p, idx, best.key as u8);
                            self.packed.store(p, Ordering::Relaxed);
                        }
                        pareto_empty = false;
                    }
                }
            }
            if cfg.colon {
                if let Some(p) = &files.highest_colon {
                    if let Ok(content) = fs::read_to_string(p) {
                        let entries = parse_top_n_file(&content);
                        if !entries.is_empty() {
                            self.highest_top_n.load_from(entries);
                            if let Some(best) = self.highest_top_n.get_items().first() {
                                let mut p = self.packed.load(Ordering::Relaxed);
                                p = set_max(p, I_HIGH_C, best.key as u8);
                                self.packed.store(p, Ordering::Relaxed);
                            }
                            pareto_empty = false;
                        }
                    }
                }
            }
        }

        if cfg.shortest {
            if let Some(p) = &files.shortest {
                if let Ok(content) = fs::read_to_string(p) {
                    let entries = parse_top_n_file(&content);
                    if !entries.is_empty() {
                        self.shortest_top_n.load_from(entries);
                        if let Some(best) = self.shortest_top_n.get_items().first() {
                            let idx = if cfg.colon { I_SHORT_C } else { I_SHORTEST };
                            let mut p = self.packed.load(Ordering::Relaxed);
                            p = set_min(p, idx, (u64::MAX - best.key) as u8);
                            self.packed.store(p, Ordering::Relaxed);
                        }
                        pareto_empty = false;
                    }
                }
            }
            if cfg.colon {
                if let Some(p) = &files.shortest_colon {
                    if let Ok(content) = fs::read_to_string(p) {
                        let entries = parse_top_n_file(&content);
                        if !entries.is_empty() {
                            self.shortest_top_n.load_from(entries);
                            if let Some(best) = self.shortest_top_n.get_items().first() {
                                let mut p = self.packed.load(Ordering::Relaxed);
                                p = set_min(p, I_SHORT_C, (u64::MAX - best.key) as u8);
                                self.packed.store(p, Ordering::Relaxed);
                            }
                            pareto_empty = false;
                        }
                    }
                }
            }
        }

        if cfg.lowest {
            if let Some(p) = &files.lowest {
                if let Ok(content) = fs::read_to_string(p) {
                    let entries = parse_top_n_file(&content);
                    if !entries.is_empty() {
                        self.lowest_top_n.load_from(entries);
                        if let Some(best) = self.lowest_top_n.get_items().first() {
                            *self.lowest_addr.lock().unwrap() = best.addr_val;
                            self.lowest_packed.store(u64::MAX - best.key, Ordering::Relaxed);
                        }
                        pareto_empty = false;
                    }
                }
            }
            if cfg.colon {
                if let Some(p) = &files.lowest_colon {
                    if let Ok(content) = fs::read_to_string(p) {
                        let entries = parse_top_n_file(&content);
                        if !entries.is_empty() {
                            self.lowest_top_n.load_from(entries);
                            if let Some(best) = self.lowest_top_n.get_items().first() {
                                *self.lowest_colon_addr.lock().unwrap() = best.addr_val;
                                self.lowest_colon_packed.store(u64::MAX - best.key, Ordering::Relaxed);
                            }
                            pareto_empty = false;
                        }
                    }
                }
            }
        }

        // Fallback: load old single-record format if top N file empty/missing
        if cfg.highest {
            if let Some(p) = &files.highest {
                if let Ok(content) = fs::read_to_string(p) {
                    if let Some(height) = parse_height(&content) {
                        let idx = if cfg.colon { I_HIGH_C } else { I_HIGHEST };
                        let mut p = self.packed.load(Ordering::Relaxed);
                        p = set_max(p, idx, height);
                        self.packed.store(p, Ordering::Relaxed);
                        pareto_empty = false;
                    }
                }
            }
            if cfg.colon {
                if let Some(p) = &files.highest_colon {
                    if let Ok(content) = fs::read_to_string(p) {
                        if let Some(height) = parse_height(&content) {
                            let mut p = self.packed.load(Ordering::Relaxed);
                            p = set_max(p, I_HIGH_C, height);
                            self.packed.store(p, Ordering::Relaxed);
                            pareto_empty = false;
                        }
                    }
                }
            }
        }

        if cfg.shortest {
            if let Some(p) = &files.shortest {
                if let Ok(content) = fs::read_to_string(p) {
                    if let Some(len) = parse_length(&content) {
                        let idx = if cfg.colon { I_SHORT_C } else { I_SHORTEST };
                        let mut p = self.packed.load(Ordering::Relaxed);
                        p = set_min(p, idx, len);
                        self.packed.store(p, Ordering::Relaxed);
                        pareto_empty = false;
                    }
                }
            }
            if cfg.colon {
                if let Some(p) = &files.shortest_colon {
                    if let Ok(content) = fs::read_to_string(p) {
                        if let Some(len) = parse_length(&content) {
                            let mut p = self.packed.load(Ordering::Relaxed);
                            p = set_min(p, I_SHORT_C, len);
                            self.packed.store(p, Ordering::Relaxed);
                            pareto_empty = false;
                        }
                    }
                }
            }
        }

        if cfg.lowest {
            if let Some(p) = &files.lowest {
                if let Ok(content) = fs::read_to_string(p) {
                    if let Some(addr_val) = parse_address(&content) {
                        *self.lowest_addr.lock().unwrap() = addr_val;
                        if let Some(lv) = parse_lowest_packed(&content) {
                            self.lowest_packed.store(lv, Ordering::Relaxed);
                        }
                        pareto_empty = false;
                    }
                }
            }
            if cfg.colon {
                if let Some(p) = &files.lowest_colon {
                    if let Ok(content) = fs::read_to_string(p) {
                        if let Some(addr_val) = parse_address(&content) {
                            *self.lowest_colon_addr.lock().unwrap() = addr_val;
                            if let Some(lv) = parse_lowest_packed(&content) {
                                self.lowest_colon_packed.store(lv, Ordering::Relaxed);
                            }
                            pareto_empty = false;
                        }
                    }
                }
            }
        }

        if let Ok(content) = fs::read_to_string(&files.pareto) {
            let mut pareto = self.pareto.lock().unwrap();
            for line in content.lines() {
                if line.starts_with("  height=") {
                    if let Some((h, l, addr, pk)) = parse_pareto_line(line, &content) {
                        pareto.try_add(h, l, &addr, &pk);
                        pareto_empty = false;
                    }
                }
            }
        }

        self.pareto_empty.store(pareto_empty, Ordering::Relaxed);
    }
}

fn parse_top_n_file(content: &str) -> Vec<TopNEntry> {
    let mut entries = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Format: key=... addr=... pubkey=... privkey=...
        let mut key = None;
        let mut addr = None;
        let mut pubkey = None;
        let mut privkey = None;
        for part in line.split_whitespace() {
            if let Some(v) = part.strip_prefix("key=") {
                key = v.parse().ok();
            } else if let Some(v) = part.strip_prefix("addr=") {
                addr = v.parse::<std::net::Ipv6Addr>().ok().map(|a| u128::from_be_bytes(a.octets()));
            } else if let Some(v) = part.strip_prefix("pubkey=") {
                pubkey = Some(v.to_string());
            } else if let Some(v) = part.strip_prefix("privkey=") {
                privkey = Some(v.to_string());
            }
        }
        if let (Some(key), Some(addr_val), Some(pubkey_str), Some(privkey_str)) = (key, addr, pubkey, privkey) {
            // addr_str not in new format, reconstruct from addr_val
            let addr_str = std::net::Ipv6Addr::from(u128::to_be_bytes(addr_val)).to_string();
            entries.push(TopNEntry { key, addr_val, addr_str, pubkey_str, privkey_str });
        }
    }
    entries
}

fn parse_height(content: &str) -> Option<u8> {
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("height:") {
            return rest.trim().parse().ok();
        }
    }
    None
}

fn parse_length(content: &str) -> Option<u8> {
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("length:") {
            return rest.trim().parse().ok();
        }
    }
    None
}

fn parse_address(content: &str) -> Option<u128> {
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("address:") {
            let addr_str = rest.trim();
            if let Ok(addr) = addr_str.parse::<std::net::Ipv6Addr>() {
                return Some(u128::from_be_bytes(addr.octets()));
            }
        }
    }
    None
}

fn parse_lowest_packed(content: &str) -> Option<u64> {
    let height = parse_height(content)?;
    let addr_val = parse_address(content)?;
    let segs = addr_to_segs(addr_val);
    let tail56 = ((segs[1] as u64) << 40)
        | ((segs[2] as u64) << 24)
        | ((segs[3] as u64) << 8)
        | ((segs[4] >> 8) as u64);
    Some(((height as u64) << 56) | tail56)
}

fn addr_to_segs(addr_val: u128) -> [u16; 8] {
    let octets = addr_val.to_be_bytes();
    let mut segs = [0u16; 8];
    for i in 0..8 {
        segs[i] = u16::from_be_bytes([octets[i * 2], octets[i * 2 + 1]]);
    }
    segs
}

fn parse_pareto_line(line: &str, full_content: &str) -> Option<(u8, u8, String, String)> {
    let height_str = line.strip_prefix("  height=")?;
    let height_end = height_str.find(' ')?;
    let height: u8 = height_str[..height_end].parse().ok()?;

    let length_str = height_str[height_end..].trim_start().strip_prefix("length=")?;
    let length: u8 = length_str.trim().parse().ok()?;

    let mut addr = None;
    let mut pk = None;
    let mut in_block = false;
    for l in full_content.lines() {
        if l == line {
            in_block = true;
            continue;
        }
        if in_block {
            if let Some(rest) = l.strip_prefix("    addr=") {
                addr = Some(rest.to_string());
            } else if let Some(rest) = l.strip_prefix("    pubkey=") {
                pk = Some(rest.to_string());
            } else if l.starts_with("  height=") && l != line {
                break;
            }
        }
    }
    Some((height, length, addr?, pk?))
}

pub struct RecordFiles {
    pub highest: Option<PathBuf>,
    pub lowest: Option<PathBuf>,
    pub shortest: Option<PathBuf>,
    pub highest_colon: Option<PathBuf>,
    pub lowest_colon: Option<PathBuf>,
    pub shortest_colon: Option<PathBuf>,
    pub pareto: PathBuf,
    pub log: Option<Mutex<fs::File>>,
}

pub fn handle_keypair(
    seed: &[u8],
    pk: &[u8],
    height: u8,
    records: &Records,
    files: &RecordFiles,
    cfg: &SearchConfig,
) {
    // segments only if needed
    let (segs_opt, run, has_colon, len) = if cfg.needs_segs() {
        let s = segments_for_pubkey(pk, height);
        let r = longest_zero_run(&s);
        let hc = r >= 2;
        let l = addr_length(&s, r);
        (Some(s), r, hc, l)
    } else {
        (None, 0, false, 0)
    };

    if cfg.colon && !cfg.both_colon && !has_colon {
        return;
    }

    let packed = records.packed.load(Ordering::Relaxed);

    let mut highest_fired = false;
    let mut shortest_fired = false;
    let mut lowest_fired = false;

    if cfg.highest {
        let idx = if cfg.colon { I_HIGH_C } else { I_HIGHEST };
        if height > get_byte(packed, idx) {
            highest_fired = true;
        }
    }

    if cfg.shortest {
        let idx = if cfg.colon { I_SHORT_C } else { I_SHORTEST };
        if len < get_byte(packed, idx) {
            shortest_fired = true;
        }
    }

    let s_for_low = if cfg.lowest { segs_opt.as_ref() } else { None };
    let mut lv = 0u64;
    if cfg.lowest {
        let s = s_for_low.unwrap();
        let tail56 = ((s[1] as u64) << 40)
            | ((s[2] as u64) << 24)
            | ((s[3] as u64) << 8)
            | ((s[4] >> 8) as u64);
        lv = ((height as u64) << 56) | tail56;
        let at = if cfg.colon {
            &records.lowest_colon_packed
        } else {
            &records.lowest_packed
        };
        let current = at.load(Ordering::Relaxed);
        if lv < current && at.compare_exchange(current, lv, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
            lowest_fired = true;
        }
    }

    // Pareto: only try when we already reached the slow path or the frontier is empty.
    let mut pareto_fired = false;
    if segs_opt.is_some() {
        if highest_fired || shortest_fired || lowest_fired || records.pareto_empty.load(Ordering::Relaxed) {
            pareto_fired = true;
        }
    }

    if !(highest_fired || shortest_fired || lowest_fired || pareto_fired) {
        return;
    }

    // ---------- SLOW PATH ----------
    let addr = address_for_pubkey(pk);
    let str_addr = addr.to_string();
    let addr_val = u128::from_be_bytes(addr.octets());

    // Verify kernel output against Ed25519 — only on records.
    {
        let mut s = [0u8; 32];
        s.copy_from_slice(seed);
        let signing_key = SigningKey::from_bytes(&s);
        assert_eq!(signing_key.verifying_key().to_bytes(), pk);
    }

    let mut sk64 = [0u8; 64];
    sk64[..32].copy_from_slice(seed);
    sk64[32..].copy_from_slice(pk);
    let pubkey_str: String = pk.encode_hex();
    let ts = Utc::now().to_rfc3339();

    let mut any_top_n_updated = false;

    let privkey_hex = sk64.encode_hex::<String>();

    if cfg.highest {
        let idx = if cfg.colon { I_HIGH_C } else { I_HIGHEST };
        let key = height as u64;
        if records.highest_top_n.try_insert(key, addr_val, str_addr.clone(), pubkey_str.clone(), privkey_hex.clone()) {
            any_top_n_updated = true;
            records
                .packed
                .try_update(Ordering::SeqCst, Ordering::Relaxed, |mut p| {
                    let before = p;
                    p = set_max(p, idx, height);
                    if p == before { None } else { Some(p) }
                })
                .ok();
        }
    }

    if cfg.shortest {
        let idx = if cfg.colon { I_SHORT_C } else { I_SHORTEST };
        let key = u64::MAX - len as u64;
        if records.shortest_top_n.try_insert(key, addr_val, str_addr.clone(), pubkey_str.clone(), privkey_hex.clone()) {
            any_top_n_updated = true;
            records
                .packed
                .try_update(Ordering::SeqCst, Ordering::Relaxed, |mut p| {
                    let before = p;
                    p = set_min(p, idx, len);
                    if p == before { None } else { Some(p) }
                })
                .ok();
        }
    }

    if cfg.lowest {
        let key = u64::MAX - lv;
        if records.lowest_top_n.try_insert(key, addr_val, str_addr.clone(), pubkey_str.clone(), privkey_hex.clone()) {
            any_top_n_updated = true;
            let mut l = if cfg.colon {
                records.lowest_colon_addr.lock().unwrap()
            } else {
                records.lowest_addr.lock().unwrap()
            };
            *l = addr_val;
        }
    }

    let pareto_new = if pareto_fired {
        let mut pareto = records.pareto.lock().unwrap();
        let added = pareto.try_add(height, len, &str_addr, &pubkey_str);
        if added {
            records.pareto_empty.store(false, Ordering::Relaxed);
        }
        added
    } else {
        false
    };

    let body = format!(
        "height:      {height}\nlength:      {len}\nzero_run:    {run}\n\
         address:     {str_addr}\npublic_key:  {pubkey_str}\n\
         private_key: {}\ntimestamp:   {ts}\n",
        sk64.encode_hex::<String>()
    );

    if let Some(ref l) = files.log {
        let _ = writeln!(
            l.lock().unwrap(),
            "=== HEIGHT {height} LEN {len} ===\n{body}"
        );
    }

    // Write top N files if any updated
    if any_top_n_updated {
        if cfg.highest {
            write_top_n_file(&files.highest, &records.highest_top_n);
            if cfg.colon {
                write_top_n_file(&files.highest_colon, &records.highest_top_n);
            }
        }
        if cfg.shortest {
            write_top_n_file(&files.shortest, &records.shortest_top_n);
            if cfg.colon {
                write_top_n_file(&files.shortest_colon, &records.shortest_top_n);
            }
        }
        if cfg.lowest {
            write_top_n_file(&files.lowest, &records.lowest_top_n);
            if cfg.colon {
                write_top_n_file(&files.lowest_colon, &records.lowest_top_n);
            }
        }
    }

    if pareto_new {
        let r = records.pareto.lock().unwrap().render();
        let _ = fs::write(&files.pareto, r);
    }

    eprintln!("height={height} len={len} addr={str_addr}");
}

fn write_top_n_file(path_opt: &Option<PathBuf>, top_n: &TopN) {
    if let Some(path) = path_opt {
        let entries = top_n.get_items();
        let mut content = String::new();
        for e in entries {
            content.push_str(&format!("key={} addr={} pubkey={} privkey={}\n", e.key, e.addr_str, e.pubkey_str, e.privkey_str));
        }
        let _ = fs::write(path, content);
    }
}

pub fn height_of_pubkey(pk: &[u8]) -> u8 {
    let mut z = 0u8;
    for b in pk {
        let lz = b.leading_zeros() as u8;
        z += lz;
        if lz != 8 {
            break;
        }
    }
    z
}

fn segments_for_pubkey(pk: &[u8], height: u8) -> [u16; 8] {
    let mut segs = [0u16; 8];
    segs[0] = u16::from_be_bytes([0x02, height]);

    let skip = usize::from(height) + 1;
    let base = (skip / 8).min(pk.len());
    let shift = (skip % 8) as u32;
    let src = &pk[base..];

    let mut tail = [0u8; 14];
    if shift == 0 {
        for (i, b) in src.iter().take(14).enumerate() {
            tail[i] = b ^ 0xFF;
        }
    } else {
        for (i, w) in src.windows(2).take(14).enumerate() {
            tail[i] = w[0].wrapping_shl(shift) ^ w[1].wrapping_shr(8 - shift) ^ 0xFF;
        }
    }
    for i in 0..7 {
        segs[i + 1] = u16::from_be_bytes([tail[2 * i], tail[2 * i + 1]]);
    }
    segs
}

fn longest_zero_run(segs: &[u16; 8]) -> usize {
    let mut best = 0;
    let mut cur = 0;
    for &s in segs.iter().skip(1) {
        if s == 0 {
            cur += 1;
            best = best.max(cur);
        } else {
            cur = 0;
        }
    }
    best
}

fn hex_len(x: u16) -> u8 {
    match x {
        0..=0x000f => 1,
        0x0010..=0x00ff => 2,
        0x0100..=0x0fff => 3,
        _ => 4,
    }
}

fn addr_length(segs: &[u16; 8], run: usize) -> u8 {
    let mut t: u16 = 7;
    for &s in segs {
        t += u16::from(hex_len(s));
    }
    if run >= 2 {
        let end = segs.iter().rposition(|&s| s != 0).map_or(8, |i| i + 1);
        let save = if end == 8 { 2 * run - 2 } else { 2 * run - 1 };
        t -= save as u16;
    }
    t as u8
}

fn address_for_pubkey(pk: &[u8]) -> std::net::Ipv6Addr {
    let z = height_of_pubkey(pk);
    let mut b = [0u8; 16];
    b[0] = 0x02;
    b[1] = z;
    let skip = usize::from(z) + 1;
    let base = (skip / 8).min(pk.len());
    let shift = (skip % 8) as u32;
    let src = &pk[base..];
    if shift == 0 {
        for (b_src, trg) in src.iter().take(14).zip(b[2..].iter_mut()) {
            *trg = b_src ^ 0xFF;
        }
    } else {
        for (w, trg) in src.windows(2).take(14).zip(b[2..].iter_mut()) {
            *trg = w[0].wrapping_shl(shift) ^ w[1].wrapping_shr(8 - shift) ^ 0xFF;
        }
    }
    std::net::Ipv6Addr::from(b)
}

/// CPU-side reference metrics used by GPU filter regression tests.
#[cfg(test)]
pub(crate) fn metrics_for_test(pk: &[u8]) -> (u8, u8, usize, u64) {
    let height = height_of_pubkey(pk);
    let segs = segments_for_pubkey(pk, height);
    let run = longest_zero_run(&segs);
    let length = addr_length(&segs, run);
    let tail56 = ((segs[1] as u64) << 40)
        | ((segs[2] as u64) << 24)
        | ((segs[3] as u64) << 8)
        | ((segs[4] >> 8) as u64);
    (height, length, run, ((height as u64) << 56) | tail56)
}
