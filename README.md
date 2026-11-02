# Yggdrasil Miner

Address miner for the Yggdrasil network

Supports mining for high and/or short addresses

## Binary builds

Please build it yourself with 
```shell
cargo build --release
```

## Basic usage (just search high addresses)

```shell
# build and run from source
cargo run --release -- -H
# or download and run
./yggdrasil-miner -H
```

## Address matching

To change which address will be searched ou will need to rewrite the matching part of the code! `src/handler.rs:22`

## Benchmarks

- AMD Radeon RX 550: 0.52 MH/s
- AMD Radeon RX 5500 XT: 5.9 MH/s
- Nvidia GeForce RTX 4060: 18.13 MH/s
- Nvidia GeForce RTX 3070 Laptop: ~60 MH/s (16-bit windows, at its 95 W power limit)
- AMD Radeon RX 6800 XT: 25 MH/s (likely outdated)
- AMD Radeon RX 9070 XT: ~148 MH/s (16-bit windows)
- Intel Iris Xe TGL GT2 (integrated in i5-1135G7): 0.67 MH/s (likely outdated)

The OpenCL path uses a fixed-base window table generated at startup. The
window size is selectable with `-w/--window-bits` (4-16, default 16); larger
windows need fewer point additions per key but a larger table. On the RX 9070
XT, 16-bit windows use a ~120 MiB table and reach ~148 MH/s, while 8-bit
windows need only ~1 MiB and reach ~87 MH/s. The table is not embedded in the
binary; it is built once when the miner starts. One field inversion is shared
by each batch of 64 keys, and the host recomputes every reported hit with
ed25519-dalek.

`--throttle N` keeps the GPU busy N percent of the time, in bursts of 3-15 s followed
by a rest. On an RTX 3070 Laptop, `--throttle 25` gives 25% of the speed at 38 W
instead of 95 W, and the speed stays proportional down to `--throttle 10`.

`--temp-limit C` (default 85, 0 disables) pauses mining at that GPU temperature and
resumes 5 degrees below it. The temperature comes from nvidia-smi on Nvidia and from
sysfs for amdgpu on Linux. Elsewhere the miner prints a notice and runs without the guard.

Feel free to add your results and report hardware-specific issues!

## Credits
- [nano-vanity](https://github.com/PlasmaPower/nano-vanity) for ed25519 OpenCL kernel
- [opencl_brute](https://github.com/bkerler/opencl_brute) for sha512 OpenCL kernel
- yggminer (Python/OpenCL) for the fixed-base OpenCL kernel and address derivation
