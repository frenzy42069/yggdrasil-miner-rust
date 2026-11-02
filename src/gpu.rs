use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use chrono::SecondsFormat;
use chrono::Utc;
use ed25519_dalek::SigningKey;
use ocl::builders::{DeviceSpecifier, ProgramBuilder};
use ocl::enums::{KernelWorkGroupInfo, KernelWorkGroupInfoResult};
use ocl::flags::MemFlags;
use ocl::{Buffer, Device, Kernel, Platform, ProQue, Result};
use rand::Rng;
use rayon::prelude::*;

use crate::Args;
use crate::handler::{RecordFiles, Records, SearchConfig, handle_keypair, height_of_pubkey};
use crate::table::{generate as generate_table, window_count};
use crate::thermal::{Guard, Monitor};

const BATCH: usize = 64;
const HIT_CAPACITY: usize = 1 << 20;
const WARMUP_KEYS: usize = 4096;

const F_FORCE_ALL: i32 = 1;
const F_HIGHEST: i32 = 2;
const F_SHORTEST: i32 = 4;
const F_LOWEST: i32 = 8;
const F_COLON_REQUIRED: i32 = 16;

const TARGET_BATCH_SECONDS: f64 = 0.25;
const MIN_WORK_ITEMS: usize = 64;
const MAX_WORK_ITEMS: usize = 1 << 20;

const THROTTLE_PERIOD: Duration = Duration::from_secs(15);
const MIN_BURST: Duration = Duration::from_secs(3);

#[derive(Clone, Copy, Default)]
struct GpuFilter {
    flags: i32,
    min_height: i32,
    max_length: i32,
    lowest_packed: u64,
}

impl GpuFilter {
    fn for_records(records: &Records, cfg: &SearchConfig) -> Self {
        let packed = records.packed.load(Ordering::Relaxed);
        let highest_idx = if cfg.colon { 3 } else { 1 };
        let shortest_idx = if cfg.colon { 2 } else { 0 };
        let lowest = if cfg.colon {
            records.lowest_colon_packed.load(Ordering::Relaxed)
        } else {
            records.lowest_packed.load(Ordering::Relaxed)
        };

        let mut flags = 0;
        if cfg.highest {
            flags |= F_HIGHEST;
        }
        if cfg.shortest {
            flags |= F_SHORTEST;
        }
        if cfg.lowest {
            flags |= F_LOWEST;
        }
        if cfg.colon && !cfg.both_colon {
            flags |= F_COLON_REQUIRED;
        }

        Self {
            flags,
            min_height: ((packed >> (highest_idx * 8)) & 0xFF) as i32,
            max_length: ((packed >> (shortest_idx * 8)) & 0xFF) as i32,
            lowest_packed: lowest,
        }
    }

    fn force_all() -> Self {
        Self {
            flags: F_FORCE_ALL,
            ..Self::default()
        }
    }
}

struct Gpu {
    kernel: Kernel,
    table: Buffer<i32>,
    hits: Buffer<u32>,
    hit_count: Buffer<u32>,
    local_work_size: Option<usize>,
    device_name: String,
}

impl Gpu {
    fn new(args: &Args) -> Result<Self> {
        let window_bits = args.window_bits;
        let windows = window_count(window_bits);
        let table_size = windows * (1 << window_bits) * 30 * 4;

        eprintln!(
            "Generating w{window_bits} table ({windows} windows, {:.1} MiB)...",
            table_size as f64 / (1024.0 * 1024.0)
        );
        let started = Instant::now();
        let table_values = generate_table(window_bits);
        eprintln!(
            "Table generated in {:.2}s",
            started.elapsed().as_secs_f64()
        );

        let mut program_builder = ProgramBuilder::new();
        program_builder.src(include_str!("../kernel/fixedbase.cl"));
        program_builder.cmplr_opt(format!(
            "-DWINDOW_BITS={window_bits} -DWINDOWS={windows} -DBATCH={BATCH}"
        ));

        let platforms = Platform::list();
        if platforms.is_empty() {
            return Err("no OpenCL platforms".into());
        }
        if args.platform_idx >= platforms.len() {
            return Err("OpenCL platform index out of range".into());
        }

        let pq = ProQue::builder()
            .prog_bldr(program_builder)
            .platform(platforms[args.platform_idx])
            .device(DeviceSpecifier::Indices(vec![args.device_idx]))
            .dims(1)
            .build()?;

        eprintln!("GPU {} {}", pq.device().vendor()?, pq.device().name()?);

        let table = pq
            .buffer_builder::<i32>()
            .flags(MemFlags::new().read_only().host_write_only())
            .len(table_values.len())
            .copy_host_slice(&table_values)
            .build()?;
        let hits = pq
            .buffer_builder::<u32>()
            .flags(MemFlags::new().write_only().host_read_only())
            .len(HIT_CAPACITY)
            .build()?;
        let hit_count = pq
            .buffer_builder::<u32>()
            .flags(MemFlags::new().read_write())
            .len(1)
            .build()?;

        let mut kb = pq.kernel_builder("mine");
        kb.arg(&table)
            .arg(&0u64)
            .arg(&0u64)
            .arg(&0u64)
            .arg(&0u64)
            .arg(&0u64)
            .arg(&0i32)
            .arg(&0i32)
            .arg(&0i32)
            .arg(&0u64)
            .arg(&hits)
            .arg(&hit_count)
            .arg(&(HIT_CAPACITY as u32));
        let kernel = kb.build()?;
        let local_work_size = match args.local_work_size {
            Some(local) => local,
            None => preferred_local_size(&kernel, pq.device())?,
        };

        Ok(Self {
            kernel,
            table,
            hits,
            hit_count,
            local_work_size: Some(local_work_size),
            device_name: pq.device().name()?,
        })
    }

    fn enqueue(
        &self,
        work_items: usize,
        base: &[u64; 4],
        first_id: u64,
        filter: GpuFilter,
    ) -> Result<()> {
        self.hit_count.write(&[0u32][..]).enq()?;
        self.kernel.set_arg(0, &self.table)?;
        self.kernel.set_arg(1, base[0])?;
        self.kernel.set_arg(2, base[1])?;
        self.kernel.set_arg(3, base[2])?;
        self.kernel.set_arg(4, base[3])?;
        self.kernel.set_arg(5, first_id)?;
        self.kernel.set_arg(6, filter.flags)?;
        self.kernel.set_arg(7, filter.min_height)?;
        self.kernel.set_arg(8, filter.max_length)?;
        self.kernel.set_arg(9, filter.lowest_packed)?;
        self.kernel.set_arg(10, &self.hits)?;
        self.kernel.set_arg(11, &self.hit_count)?;
        self.kernel.set_arg(12, HIT_CAPACITY as u32)?;

        let mut cmd = self.kernel.cmd();
        cmd = cmd.global_work_size(work_items);
        if let Some(local) = self.local_work_size {
            cmd = cmd.local_work_size(local);
        }
        unsafe {
            cmd.enq()?;
        }
        Ok(())
    }

    fn take_hits(&self) -> Result<Vec<u32>> {
        let mut count = [0u32; 1];
        self.hit_count.read(&mut count[..]).enq()?;
        let found = (count[0] as usize).min(HIT_CAPACITY);
        let mut ids = vec![0u32; found];
        if found > 0 {
            self.hits.read(&mut ids[..]).enq()?;
        }
        Ok(ids)
    }
}

/// The hardware's own group size: smaller groups waste lanes, larger ones can fail
/// with `OUT_OF_RESOURCES` because every work item holds BATCH keys in private memory.
fn preferred_local_size(kernel: &Kernel, device: Device) -> Result<usize> {
    let preferred = kernel.wg_info(device, KernelWorkGroupInfo::PreferredWorkGroupSizeMultiple)?;
    let maximum = kernel.wg_info(device, KernelWorkGroupInfo::WorkGroupSize)?;
    match (preferred, maximum) {
        (
            KernelWorkGroupInfoResult::PreferredWorkGroupSizeMultiple(preferred),
            KernelWorkGroupInfoResult::WorkGroupSize(maximum),
        ) => Ok(preferred.min(maximum).max(1)),
        _ => Err("unexpected work group info".into()),
    }
}

fn seed_for(base: &[u64; 4], id: u64) -> [u8; 32] {
    let words = [base[0] ^ id, base[1], base[2], base[3]];
    let mut seed = [0u8; 32];
    for (i, word) in words.iter().enumerate() {
        seed[i * 8..i * 8 + 8].copy_from_slice(&word.to_be_bytes());
    }
    seed
}

fn handle_hits(
    ids: &[u32],
    base: &[u64; 4],
    first_id: u64,
    records: &Records,
    files: &RecordFiles,
    cfg: &SearchConfig,
) {
    ids.par_iter().for_each(|&offset| {
        let seed = seed_for(base, first_id + u64::from(offset));
        let signing_key = SigningKey::from_bytes(&seed);
        let pubkey = signing_key.verifying_key().to_bytes();
        let height = height_of_pubkey(&pubkey);
        handle_keypair(&seed, &pubkey, height, records, files, cfg);
    });
}

fn initial_work_items(args: &Args) -> usize {
    if let Some(global) = args.global_work_size {
        return global.max(1);
    }
    (args.threads / BATCH).max(MIN_WORK_ITEMS)
}

fn align_work_items(work_items: usize, local: Option<usize>) -> usize {
    match local {
        Some(local) if local > 1 => work_items.div_ceil(local) * local,
        _ => work_items,
    }
}

/// Holds the GPU at `throttle` percent of the time with long bursts of batches and
/// long rests.  Measured on an RTX 3070 Laptop: during short rests the GPU stays in
/// its high power state, and after longer ones it needs about a second to clock back
/// up, so rapid on/off cycling loses speed and saves little power.
struct Pacer {
    throttle: u32,
    burst: Duration,
    busy: Duration,
}

impl Pacer {
    fn new(throttle: u8) -> Self {
        let throttle = u32::from(throttle.clamp(1, 100));
        Self {
            throttle,
            burst: (THROTTLE_PERIOD * throttle / 100).max(MIN_BURST),
            busy: Duration::ZERO,
        }
    }

    /// How long to rest after a batch, if the current burst is over.
    fn rest_after(&mut self, elapsed: Duration) -> Option<Duration> {
        if self.throttle == 100 {
            return None;
        }
        self.busy += elapsed;
        if self.busy < self.burst {
            return None;
        }
        let rest = self.busy * (100 - self.throttle) / self.throttle;
        self.busy = Duration::ZERO;
        Some(rest)
    }
}

pub fn run_opencl(args: &Args, records: &Records, files: &RecordFiles, cfg: SearchConfig) {
    let gpu = Gpu::new(args).unwrap();

    let mut base_bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut base_bytes);
    let base = [
        u64::from_be_bytes(base_bytes[0..8].try_into().unwrap()),
        u64::from_be_bytes(base_bytes[8..16].try_into().unwrap()),
        u64::from_be_bytes(base_bytes[16..24].try_into().unwrap()),
        u64::from_be_bytes(base_bytes[24..32].try_into().unwrap()),
    ];

    // Seed the thresholds with a real batch first.  Without this the initial
    // "everything is a new record" thresholds would make the GPU report every
    // key in the first filtered batch.
    let warmup_items = align_work_items(WARMUP_KEYS.div_ceil(BATCH), gpu.local_work_size);
    gpu.enqueue(warmup_items, &base, 0, GpuFilter::force_all())
        .unwrap();
    let warmup_hits = gpu.take_hits().unwrap();
    handle_hits(&warmup_hits, &base, 0, records, files, &cfg);

    let mut first_id = (warmup_items * BATCH) as u64;
    let mut work_items = align_work_items(initial_work_items(args), gpu.local_work_size);
    let fixed_work_items = args.global_work_size.is_some();

    let mut start = Instant::now();
    let mut iters = 0u64;
    let mut first = true;
    let mut pacer = Pacer::new(args.throttle);
    let mut woke_up = false;

    let monitor = (args.temp_limit > 0).then(|| Monitor::start(gpu.device_name.clone()));
    if monitor.as_ref().is_some_and(|m| m.temperature().is_none()) {
        eprintln!("GPU temperature is not available, --temp-limit has no effect");
    }
    let mut guard = Guard::new(args.temp_limit);

    loop {
        let temperature = monitor.as_ref().and_then(Monitor::temperature);
        let was_cooling = guard.cooling();
        if guard.must_wait(temperature) {
            if !was_cooling {
                eprintln!(
                    "GPU at {} C, pausing until it is below {} C",
                    temperature.unwrap_or_default(),
                    guard.resume_at()
                );
            }
            std::thread::sleep(Duration::from_secs(1));
            continue;
        }
        if was_cooling {
            eprintln!("GPU cooled down, resuming");
            woke_up = true;
        }

        let filter = GpuFilter::for_records(records, &cfg);
        let work_items_now = align_work_items(work_items, gpu.local_work_size);

        let batch_start = Instant::now();
        gpu.enqueue(work_items_now, &base, first_id, filter)
            .unwrap();
        let hits = gpu.take_hits().unwrap();
        let elapsed = batch_start.elapsed();

        handle_hits(&hits, &base, first_id, records, files, &cfg);

        let tried = (work_items_now * BATCH) as u64;
        first_id += tried;
        iters += tried;

        // A batch right after a sleep runs on low clocks.  Shrinking the next one
        // because of it would keep the GPU from ever clocking back up.
        if !fixed_work_items && !woke_up {
            if elapsed.as_secs_f64() < TARGET_BATCH_SECONDS * 0.6 {
                work_items = (work_items * 2).min(MAX_WORK_ITEMS);
            } else if elapsed.as_secs_f64() > TARGET_BATCH_SECONDS * 1.6 {
                work_items = (work_items / 2).max(MIN_WORK_ITEMS);
            }
        }

        let rest = pacer.rest_after(elapsed);
        if let Some(rest) = rest {
            std::thread::sleep(rest);
        }
        woke_up = rest.is_some();

        // While throttling, report whole burst + rest cycles so the rate is not
        // skewed by where the interval happens to end.
        let el = start.elapsed();
        if !first && el.as_secs() >= args.log_interval && (pacer.throttle == 100 || woke_up) {
            #[allow(clippy::cast_precision_loss)]
            let hr = iters as f64 / el.as_secs_f64() / 1e6;
            eprintln!(
                "{} Hashrate: {:.2} MH/s",
                Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
                hr
            );
            start = Instant::now();
            iters = 0;
        }
        first = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    const TEST_BASE: [u64; 4] = [
        0x0123_4567_89ab_cdef,
        0xfedc_ba98_7654_3210,
        0x0f1e_2d3c_4b5a_6978,
        0x8877_6655_4433_2211,
    ];
    const ALL_KEYS: usize = 65536;

    fn cycle(pacer: &mut Pacer, batch: Duration) -> (Duration, Duration) {
        let mut busy = Duration::ZERO;
        loop {
            busy += batch;
            if let Some(rest) = pacer.rest_after(batch) {
                return (busy, rest);
            }
        }
    }

    #[test]
    fn full_speed_never_rests() {
        let mut pacer = Pacer::new(100);
        assert_eq!(pacer.rest_after(Duration::from_secs(60)), None);
    }

    #[test]
    fn rest_keeps_the_requested_share() {
        let batch = Duration::from_millis(250);
        for (throttle, burst, rest) in [
            (75, 11_250, 3_750),
            (50, 7_500, 7_500),
            (10, 3_000, 27_000), // short bursts would never reach full clocks
        ] {
            let mut pacer = Pacer::new(throttle);
            for _ in 0..3 {
                let (busy, slept) = cycle(&mut pacer, batch);
                assert_eq!(busy, Duration::from_millis(burst), "throttle {throttle}");
                assert_eq!(slept, Duration::from_millis(rest), "throttle {throttle}");
            }
        }
    }

    #[test]
    fn slow_batches_count_towards_the_share() {
        let mut pacer = Pacer::new(25);
        assert_eq!(
            pacer.rest_after(Duration::from_secs(8)),
            Some(Duration::from_secs(24))
        );
    }

    fn compare(
        gpu: &Gpu,
        metrics: &[(u8, u8, usize, u64)],
        keys: usize,
        filter: GpuFilter,
        expected: impl Fn(&(u8, u8, usize, u64)) -> bool,
    ) {
        assert_eq!(keys % BATCH, 0);
        gpu.enqueue(keys / BATCH, &TEST_BASE, 0, filter).unwrap();
        let mut got = gpu.take_hits().unwrap();
        got.sort_unstable();

        let want: Vec<u32> = (0..keys)
            .filter(|&i| expected(&metrics[i]))
            .map(|i| i as u32)
            .collect();
        assert!(
            want.len() < HIT_CAPACITY,
            "test case has too many hits: {}",
            want.len()
        );
        assert_eq!(got, want, "GPU filter mismatch (flags={})", filter.flags);
    }

    #[test]
    #[ignore = "requires a GPU"]
    fn gpu_filters_match_cpu() {
        let args = Args::parse_from(["yggdrasil-miner", "--highest", "-w", "8"]);
        let gpu = Gpu::new(&args).unwrap();

        let metrics: Vec<(u8, u8, usize, u64)> = (0..ALL_KEYS)
            .into_par_iter()
            .map(|i| {
                let seed = seed_for(&TEST_BASE, i as u64);
                let pk = SigningKey::from_bytes(&seed).verifying_key().to_bytes();
                crate::handler::metrics_for_test(&pk)
            })
            .collect();

        compare(
            &gpu,
            &metrics,
            ALL_KEYS,
            GpuFilter {
                flags: F_HIGHEST,
                min_height: 8,
                ..GpuFilter::default()
            },
            |m| i32::from(m.0) > 8,
        );

        compare(
            &gpu,
            &metrics,
            ALL_KEYS,
            GpuFilter {
                flags: F_SHORTEST,
                max_length: 37,
                ..GpuFilter::default()
            },
            |m| m.1 < 37,
        );

        compare(
            &gpu,
            &metrics,
            16384,
            GpuFilter {
                flags: F_LOWEST,
                lowest_packed: 0x0100_1234_5678_9abc,
                ..GpuFilter::default()
            },
            |m| m.3 < 0x0100_1234_5678_9abc,
        );

        compare(
            &gpu,
            &metrics,
            ALL_KEYS,
            GpuFilter {
                flags: F_HIGHEST | F_COLON_REQUIRED,
                min_height: 0,
                ..GpuFilter::default()
            },
            |m| m.0 > 0 && m.2 >= 2,
        );

        compare(
            &gpu,
            &metrics,
            ALL_KEYS,
            GpuFilter {
                flags: F_SHORTEST | F_COLON_REQUIRED,
                max_length: 40,
                ..GpuFilter::default()
            },
            |m| m.1 < 40 && m.2 >= 2,
        );

        let filter = GpuFilter {
            flags: F_HIGHEST | F_SHORTEST | F_LOWEST,
            min_height: 6,
            max_length: 38,
            lowest_packed: 0x0180_0000_0000_0000,
        };
        compare(&gpu, &metrics, 16384, filter, |m| {
            m.0 > 6 || m.1 < 38 || m.3 < 0x0180_0000_0000_0000
        });
    }

    const METRICS_SRC: &str = concat!(
        include_str!("../kernel/fixedbase.cl"),
        r#"
__kernel void metrics(__global const int *table,
                      ulong seed0, ulong seed1, ulong seed2, ulong seed3, ulong first_id,
                      __global ulong *out)
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
        int zeros = (int)clz(k0), keep = zeros + 1;
        ulong a0 = ~(keep == 64 ? k1 : (k0 << keep) | (k1 >> (64 - keep)));
        ulong a1 = ~(keep == 64 ? k2 : (k1 << keep) | (k2 >> (64 - keep)));
        a1 &= 0xFFFFFFFFFFFF0000UL;

        int length = 0, zero_run = 0;
        address_metrics(a0, a1, &length, &zero_run);
        ulong lv = ((ulong)zeros << 56) | (a0 >> 8);
        __global ulong *o = out + (ulong)(base + i - first_id) * 2;
        o[0] = (ulong)zeros | ((ulong)length << 8) | ((ulong)zero_run << 16);
        o[1] = lv;
    }
}
"#
    );

    #[test]
    #[ignore = "requires a GPU"]
    fn gpu_metrics_match_cpu() {
        let args = Args::parse_from(["yggdrasil-miner", "--highest", "-w", "16"]);
        let window_bits = args.window_bits;
        let windows = window_count(window_bits);
        let table_values = generate_table(window_bits);
        let platforms = Platform::list();
        let mut pb = ProgramBuilder::new();
        pb.src(METRICS_SRC);
        pb.cmplr_opt(format!(
            "-DWINDOW_BITS={window_bits} -DWINDOWS={windows} -DBATCH={BATCH}"
        ));
        let pq = ProQue::builder()
            .prog_bldr(pb)
            .platform(platforms[args.platform_idx])
            .device(DeviceSpecifier::Indices(vec![args.device_idx]))
            .dims(1)
            .build()
            .unwrap();

        let table = pq
            .buffer_builder::<i32>()
            .flags(MemFlags::new().read_only().host_write_only())
            .len(table_values.len())
            .copy_host_slice(&table_values)
            .build()
            .unwrap();
        let out = pq
            .buffer_builder::<u64>()
            .flags(MemFlags::new().write_only().host_read_only())
            .len(ALL_KEYS * 2)
            .build()
            .unwrap();

        let mut kb = pq.kernel_builder("metrics");
        kb.arg(&table)
            .arg(&TEST_BASE[0])
            .arg(&TEST_BASE[1])
            .arg(&TEST_BASE[2])
            .arg(&TEST_BASE[3])
            .arg(&0u64)
            .arg(&out);
        let kernel = kb.build().unwrap();

        let mut cmd = kernel.cmd();
        cmd = cmd.global_work_size(ALL_KEYS / BATCH);
        unsafe { cmd.enq().unwrap() };
        let mut actual = vec![0u64; ALL_KEYS * 2];
        out.read(&mut actual[..]).enq().unwrap();

        for i in 0..ALL_KEYS {
            let seed = seed_for(&TEST_BASE, i as u64);
            let pk = SigningKey::from_bytes(&seed).verifying_key().to_bytes();
            let (height, length, run, lv) = crate::handler::metrics_for_test(&pk);
            let expected_word = u64::from(height)
                | (u64::from(length) << 8)
                | ((run as u64) << 16);
            assert_eq!(
                actual[i * 2],
                expected_word,
                "metric mismatch at key {i}: height={height} length={length} run={run}"
            );
            assert_eq!(actual[i * 2 + 1], lv, "lowest key mismatch at key {i}");
        }
    }
}
