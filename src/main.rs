use clap::Parser;
use std::fs;
use std::sync::Mutex;

use crate::handler::{RecordFiles, Records, SearchConfig};
use gpu::run_opencl;

mod gpu;
mod handler;
mod table;
mod thermal;

#[derive(Parser, Debug)]
#[command(author, version, about)]
pub struct Args {
    /// Search for highest leading-zero-bit count
    #[arg(short = 'h', long)]
    highest: bool,

    /// Search for numerically lowest address
    #[arg(short = 'l', long)]
    lowest: bool,

    /// Search for shortest RFC-5952 string
    #[arg(short = 's', long)]
    shortest: bool,

    /// Require a `::` run (RFC 5952 compressed) in matched addresses
    #[arg(short = 'c', long)]
    colon: bool,

    /// Also search non-colon addresses when -c is set
    #[arg(short = 'C', long)]
    both_colon: bool,

    /// Append-only log of every record hit
    #[arg(long)]
    log: Option<String>,

    /// Fixed-base window size in bits (4-16)
    #[arg(short = 'w', long, default_value_t = 16)]
    window_bits: usize,

    /// OpenCL thread count
    #[arg(short = 't', long, default_value_t = 1024 * 1024)]
    threads: usize,

    /// Log hashrate every N seconds
    #[arg(short = 'i', long, default_value_t = 10)]
    log_interval: u64,

    /// Target GPU usage percentage (1-100), held by alternating work and rest over about 15 s
    #[arg(long, default_value_t = 100)]
    throttle: u8,

    /// Pause at this GPU temperature in Celsius and resume 5 degrees below it (0 disables)
    #[arg(long, default_value_t = 85)]
    temp_limit: u8,

    /// Track top N results per category
    #[arg(long, default_value_t = 10)]
    top_n: usize,

    /// Override top N for highest
    #[arg(long)]
    top_n_highest: Option<usize>,

    /// Override top N for lowest
    #[arg(long)]
    top_n_lowest: Option<usize>,

    /// Override top N for shortest
    #[arg(long)]
    top_n_shortest: Option<usize>,

    /// Cap pareto frontier size
    #[arg(long)]
    top_n_pareto: Option<usize>,

    #[arg(long)]
    local_work_size: Option<usize>,
    #[arg(long)]
    global_work_size: Option<usize>,
    #[arg(long, default_value_t = 0)]
    platform_idx: usize,
    #[arg(long, default_value_t = 0)]
    device_idx: usize,
}

fn main() {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(main_)
        .unwrap()
        .join()
        .unwrap();
}

fn main_() {
    let args = Args::parse();

    if !(4..=16).contains(&args.window_bits) {
        eprintln!("--window-bits must be between 4 and 16");
        return;
    }

    let cfg = SearchConfig {
        highest: args.highest,
        lowest: args.lowest,
        shortest: args.shortest,
        colon: args.colon,
        both_colon: args.both_colon,
    };
    if !cfg.any() {
        eprintln!("specify at least one of -h -l -s");
        return;
    }

    let suffix = if args.colon { "_colon" } else { "" };

    let highest_n = args.top_n_highest.unwrap_or(args.top_n);
    let lowest_n = args.top_n_lowest.unwrap_or(args.top_n);
    let shortest_n = args.top_n_shortest.unwrap_or(args.top_n);
    let pareto_n = args.top_n_pareto;

    let files = RecordFiles {
        highest: if args.highest {
            Some(format!("highest{suffix}.txt").into())
        } else {
            None
        },
        lowest: if args.lowest {
            Some(format!("lowest{suffix}.txt").into())
        } else {
            None
        },
        shortest: if args.shortest {
            Some(format!("shortest{suffix}.txt").into())
        } else {
            None
        },
        highest_colon: if args.highest && args.colon {
            Some("highest_colon.txt".into())
        } else {
            None
        },
        lowest_colon: if args.lowest && args.colon {
            Some("lowest_colon.txt".into())
        } else {
            None
        },
        shortest_colon: if args.shortest && args.colon {
            Some("shortest_colon.txt".into())
        } else {
            None
        },
        pareto: "pareto.txt".into(),
        log: args.log.as_ref().map(|p| {
            Mutex::new(
                fs::OpenOptions::new()
                    .append(true)
                    .create(true)
                    .open(p)
                    .unwrap(),
            )
        }),
    };

    let records = Records::with_top_n(highest_n, lowest_n, shortest_n, pareto_n);
    records.load_from_files(&files, &cfg);
    run_opencl(&args, &records, &files, cfg);
}
